"""Estrategia de descarga vía HTTP plano (httpx)."""

from __future__ import annotations

import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_assistant.config import Settings
from rag_assistant.core.exceptions import BlockedByTargetError, FetchError
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import RawPage
from rag_assistant.scraping.fetchers.base import Fetcher

logger = get_logger(__name__)

#: Códigos que indican bloqueo deliberado del servidor, no un fallo transitorio.
_BLOCKING_STATUS = {401, 403, 407, 429}


class HttpFetcher(Fetcher):
    """Descarga rápida sin navegador. Válida para sitios sin anti-bot."""

    name = "http"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.scraper_timeout_seconds),
            follow_redirects=True,
            headers={
                "User-Agent": self._settings.scraper_user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
            },
            limits=httpx.Limits(max_connections=self._settings.scraper_concurrency * 2),
        )
        logger.debug("http_fetcher.started")

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _get(self, url: str) -> httpx.Response:
        assert self._client is not None, "HttpFetcher usado sin startup()"
        return await self._client.get(url)

    async def fetch(self, url: str, *, depth: int = 0) -> RawPage:
        started = time.perf_counter()
        try:
            response = await self._get(url)
        except httpx.HTTPError as exc:  # red, DNS, timeout tras los reintentos
            raise FetchError(url, detail=f"{type(exc).__name__}: {exc}") from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code in _BLOCKING_STATUS:
            raise BlockedByTargetError(
                url,
                status_code=response.status_code,
                detail=(
                    "El sitio rechazó la petición HTTP plana. "
                    "Usa SCRAPER_FETCHER=browser para renderizar con Chromium."
                ),
            )
        if response.status_code >= 400:
            raise FetchError(url, status_code=response.status_code, detail=response.reason_phrase)

        return RawPage(
            url=str(response.url),
            html=response.text,
            status_code=response.status_code,
            depth=depth,
            fetcher=self.name,
            elapsed_ms=elapsed_ms,
        )
