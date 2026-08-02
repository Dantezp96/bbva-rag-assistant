"""Orquestación de alto nivel: ingesta completa del sitio.

`run_ingestion()` encadena las dos mitades del sistema —scraping e
indexación— y es el punto de entrada que usan la CLI, el servicio `ingest`
de docker-compose y el endpoint `POST /ingest`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from rag_assistant.config import get_settings
from rag_assistant.core.logging import get_logger
from rag_assistant.indexing import IndexingPipeline, IndexReport
from rag_assistant.scraping import CrawlReport, ScrapeStorage, SiteCrawler

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Resultado combinado de scraping + indexación."""

    crawl: CrawlReport | None
    index: IndexReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "crawl": self.crawl.as_dict() if self.crawl else None,
            "index": self.index.as_dict(),
        }


def run_ingestion(
    *,
    seeds: list[str] | None = None,
    max_pages: int | None = None,
    recreate: bool = False,
    skip_scrape: bool = False,
    fresh: bool = False,
) -> IngestionResult:
    """Rastrea el sitio (opcional) e indexa el corpus limpio en la base vectorial.

    Args:
        seeds: URLs semilla. Por defecto, `SCRAPER_BASE_URL`.
        max_pages: Sobrescribe `SCRAPER_MAX_PAGES` para esta ejecución.
        recreate: Recrea la colección vectorial desde cero.
        skip_scrape: Reindexa el corpus ya guardado sin volver a rastrear.
        fresh: Borra los datos locales antes de rastrear.
    """
    settings = get_settings()
    settings.ensure_directories()
    storage = ScrapeStorage(settings.raw_data_dir, settings.clean_data_dir)

    if max_pages:
        # `Settings` es inmutable por diseño; se ajusta solo esta ejecución.
        object.__setattr__(settings, "scraper_max_pages", max_pages)

    crawl_report: CrawlReport | None = None
    if not skip_scrape:
        if fresh:
            storage.reset()
        logger.info(
            "ingestion.scraping_started",
            base_url=settings.scraper_base_url,
            fetcher=settings.scraper_fetcher,
            max_pages=settings.scraper_max_pages,
        )
        crawler = SiteCrawler(settings, storage=storage)
        crawl_report = asyncio.run(crawler.crawl(seeds))

        if crawl_report.documents_saved == 0:
            hint = (
                "El sitio bloqueó todas las peticiones. Prueba con "
                "SCRAPER_FETCHER=browser."
                if crawl_report.blocked
                else "Revisa SCRAPER_BASE_URL y los patrones de exclusión."
            )
            logger.error("ingestion.no_documents", hint=hint, **crawl_report.as_dict())

    documents = storage.load_clean()
    logger.info("ingestion.indexing_started", documents=len(documents))

    index_report = IndexingPipeline(settings).run(documents, recreate=recreate)

    logger.info(
        "ingestion.finished",
        documents=index_report.documents,
        chunks=index_report.chunks,
        collection=index_report.collection,
    )
    return IngestionResult(crawl=crawl_report, index=index_report)


def run_reclean_and_index(*, recreate: bool = True) -> IngestionResult:
    """Re-limpia el HTML ya descargado y reindexa, sin tocar el sitio.

    Útil al ajustar el extractor o el tamaño de chunk: itera sobre la calidad
    del corpus en segundos en vez de volver a rastrear el sitio entero.
    """
    settings = get_settings()
    crawler = SiteCrawler(settings)
    documents = crawler.reclean()
    index_report = IndexingPipeline(settings).run(documents, recreate=recreate)
    return IngestionResult(crawl=None, index=index_report)
