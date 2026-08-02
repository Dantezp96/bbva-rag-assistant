"""Crawler BFS acotado sobre el sitio objetivo.

Recorre el sitio en anchura desde una o varias URL semilla, aplicando:

* límite de páginas (`SCRAPER_MAX_PAGES`) y de profundidad (`SCRAPER_MAX_DEPTH`);
* filtro de dominio y de patrones prohibidos (`SCRAPER_URL_DENY_PATTERNS`);
* `robots.txt` (opt-out vía `SCRAPER_RESPECT_ROBOTS=false`);
* deduplicación por URL normalizada y por hash de contenido;
* retardo entre peticiones para no saturar el servidor.

El fetcher se inyecta (Strategy), así que el crawler no sabe si detrás hay
httpx o un Chromium completo.
"""

from __future__ import annotations

import asyncio
import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import (
    BlockedByTargetError,
    ExtractionError,
    FetchError,
)
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import CleanDocument
from rag_assistant.scraping.cleaner import clean_page
from rag_assistant.scraping.fetchers import Fetcher, create_fetcher
from rag_assistant.scraping.storage import ScrapeStorage

logger = get_logger(__name__)

#: Extensiones que nunca son páginas HTML.
_BINARY_SUFFIXES = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".zip", ".rar", ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx",
    ".mp4", ".mp3", ".css", ".js", ".json", ".xml",
)


@dataclass
class CrawlReport:
    """Resumen cuantitativo de una ejecución del crawler."""

    pages_fetched: int = 0
    documents_saved: int = 0
    skipped_duplicates: int = 0
    failed: int = 0
    blocked: int = 0
    extraction_failures: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "pages_fetched": self.pages_fetched,
            "documents_saved": self.documents_saved,
            "skipped_duplicates": self.skipped_duplicates,
            "failed": self.failed,
            "blocked": self.blocked,
            "extraction_failures": self.extraction_failures,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "errors": self.errors[:10],
        }


def normalize_url(url: str) -> str:
    """Normaliza para deduplicar: sin fragmento, sin query de tracking, sin `/` final."""
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    # Los parámetros de campaña generan URLs distintas con el mismo contenido.
    query = "&".join(
        part
        for part in parsed.query.split("&")
        if part and not part.split("=")[0].lower().startswith(("utm_", "gclid", "fbclid"))
    )
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


class SiteCrawler:
    """Rastreo en anchura del sitio, con guardado de crudos y limpios."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        fetcher: Fetcher | None = None,
        storage: ScrapeStorage | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._fetcher = fetcher or create_fetcher(self._settings)
        self._storage = storage or ScrapeStorage(
            self._settings.raw_data_dir, self._settings.clean_data_dir
        )
        self._robots: urllib.robotparser.RobotFileParser | None = None

    # ------------------------------------------------------------- robots --
    async def _load_robots(self) -> None:
        if not self._settings.scraper_respect_robots:
            return
        robots_url = urljoin(self._settings.scraper_base_url, "/robots.txt")
        parser = urllib.robotparser.RobotFileParser()
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                response = await client.get(
                    robots_url, headers={"User-Agent": self._settings.scraper_user_agent}
                )
            if response.status_code == 200 and "<html" not in response.text[:200].lower():
                parser.parse(response.text.splitlines())
                self._robots = parser
                logger.info("crawler.robots_loaded", url=robots_url)
            else:
                # bbva.com.co devuelve su página de bloqueo también en /robots.txt;
                # sin reglas parseables, se aplica el resto de límites de cortesía.
                logger.warning(
                    "crawler.robots_unavailable",
                    url=robots_url,
                    status=response.status_code,
                    action="se continúa respetando delay, dominio y tope de páginas",
                )
        except httpx.HTTPError as exc:
            logger.warning("crawler.robots_fetch_failed", url=robots_url, error=str(exc))

    def _robots_allows(self, url: str) -> bool:
        if self._robots is None:
            return True
        return self._robots.can_fetch(self._settings.scraper_user_agent, url)

    # ------------------------------------------------------------- filtros --
    def _is_crawlable(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if parsed.netloc.lower() not in {d.lower() for d in self._settings.scraper_allowed_domains}:
            return False
        if parsed.path.lower().endswith(_BINARY_SUFFIXES):
            return False
        if any(pattern.search(url) for pattern in self._settings.deny_regexes):
            return False
        return self._robots_allows(url)

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            candidate = normalize_url(urljoin(base_url, href))
            if self._is_crawlable(candidate):
                links.append(candidate)
        return links

    # --------------------------------------------------------------- crawl --
    async def crawl(self, seeds: list[str] | None = None) -> CrawlReport:
        """Ejecuta el rastreo completo y devuelve un informe."""
        started = time.perf_counter()
        report = CrawlReport()

        seed_urls = [normalize_url(s) for s in (seeds or [self._settings.scraper_base_url])]
        queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        for seed in seed_urls:
            queue.put_nowait((seed, 0))

        visited: set[str] = set(seed_urls)
        seen_hashes: set[str] = set()
        lock = asyncio.Lock()

        await self._load_robots()

        async with self._fetcher:

            async def worker(worker_id: int) -> None:
                while True:
                    try:
                        url, depth = await asyncio.wait_for(queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        return  # cola vacía y estable: este worker termina

                    try:
                        async with lock:
                            if report.pages_fetched >= self._settings.scraper_max_pages:
                                return
                        await self._process(url, depth, queue, visited, seen_hashes, lock, report)
                    except Exception as exc:  # noqa: BLE001 - un fallo no detiene el crawl
                        async with lock:
                            report.failed += 1
                            report.errors.append(f"{url}: {exc}")
                        logger.warning("crawler.page_failed", url=url, error=str(exc))
                    finally:
                        queue.task_done()
                        if self._settings.scraper_request_delay:
                            await asyncio.sleep(self._settings.scraper_request_delay)

            workers = [
                asyncio.create_task(worker(i)) for i in range(self._settings.scraper_concurrency)
            ]
            await asyncio.gather(*workers, return_exceptions=True)

        report.elapsed_seconds = time.perf_counter() - started
        logger.info("crawler.finished", **report.as_dict())
        return report

    async def _process(
        self,
        url: str,
        depth: int,
        queue: asyncio.Queue,
        visited: set[str],
        seen_hashes: set[str],
        lock: asyncio.Lock,
        report: CrawlReport,
    ) -> None:
        try:
            page = await self._fetcher.fetch(url, depth=depth)
        except BlockedByTargetError as exc:
            async with lock:
                report.blocked += 1
                report.errors.append(f"BLOQUEADO {url}: {exc.detail or ''}")
            logger.warning("crawler.blocked", url=url, status=exc.status_code)
            return
        except FetchError as exc:
            async with lock:
                report.failed += 1
                report.errors.append(f"{url}: {exc}")
            return

        async with lock:
            report.pages_fetched += 1
            if page.content_hash in seen_hashes:
                report.skipped_duplicates += 1
                logger.debug("crawler.duplicate_content", url=url)
                return
            seen_hashes.add(page.content_hash)

        self._storage.save_raw(page)

        try:
            document = clean_page(page)
        except ExtractionError as exc:
            async with lock:
                report.extraction_failures += 1
            logger.debug("crawler.extraction_failed", url=url, error=str(exc))
        else:
            self._storage.save_clean(document)
            async with lock:
                report.documents_saved += 1
            logger.info(
                "crawler.page_ok",
                url=url,
                depth=depth,
                words=document.word_count,
                ms=page.elapsed_ms,
                total=report.documents_saved,
            )

        if depth >= self._settings.scraper_max_depth:
            return

        for link in self._extract_links(page.html, page.url):
            async with lock:
                if link in visited or len(visited) >= self._settings.scraper_max_pages * 3:
                    continue
                visited.add(link)
            queue.put_nowait((link, depth + 1))

    # ----------------------------------------------------------- reproceso --
    def reclean(self) -> list[CleanDocument]:
        """Regenera `data/clean/` desde el HTML crudo, sin volver a rastrear."""
        self._storage.reset(raw=False, clean=True)
        documents: list[CleanDocument] = []
        for page in self._storage.iter_raw():
            try:
                document = clean_page(page)
            except ExtractionError:
                continue
            self._storage.save_clean(document)
            documents.append(document)
        logger.info("crawler.recleaned", documents=len(documents))
        return documents
