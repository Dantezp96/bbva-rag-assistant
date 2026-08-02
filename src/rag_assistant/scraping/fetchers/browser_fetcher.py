"""Estrategia de descarga con navegador real (Playwright + Chromium).

Necesaria para www.bbva.com.co: el sitio está detrás de una protección
anti-bot que devuelve 403 a `curl`/`httpx` pero sirve el contenido con
normalidad a un motor de navegador completo. Además ejecuta el JavaScript que
inyecta parte del contenido, así que el HTML resultante es más rico que el
HTML inicial.

Se reutiliza **una sola instancia de navegador** para todo el crawl y se abre
una pestaña por página: arrancar Chromium cuesta ~1 s, hacerlo por URL sería
inviable.
"""

from __future__ import annotations

import asyncio
import time

from rag_assistant.config import Settings
from rag_assistant.core.exceptions import BlockedByTargetError, FetchError, ScrapingError
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import RawPage
from rag_assistant.scraping.fetchers.base import Fetcher

logger = get_logger(__name__)

#: Recursos que no aportan texto: bloquearlos acelera el crawl ~3x.
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}

#: Marcador de la página de bloqueo de BBVA (Akamai "Customdeny").
_BLOCK_MARKERS = ("Customdeny", "Algo salió mal", "Access Denied", "Request unsuccessful")


class BrowserFetcher(Fetcher):
    """Descarga renderizando la página en Chromium headless."""

    name = "browser"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._playwright = None
        self._browser = None
        self._context = None
        self._semaphore = asyncio.Semaphore(settings.scraper_concurrency)

    async def startup(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - dependencia declarada
            raise ScrapingError(
                "Playwright no está instalado",
                detail="pip install playwright && playwright install chromium",
            ) from exc

        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",  # imprescindible dentro de Docker
                    "--disable-blink-features=AutomationControlled",
                ],
            )
        except Exception as exc:
            await self._playwright.stop()
            raise ScrapingError(
                "No se pudo iniciar Chromium",
                detail=f"{exc}. Ejecuta `playwright install --with-deps chromium`.",
            ) from exc

        self._context = await self._browser.new_context(
            user_agent=self._settings.scraper_user_agent,
            locale="es-CO",
            viewport={"width": 1440, "height": 900},
            extra_http_headers={"Accept-Language": "es-CO,es;q=0.9,en;q=0.8"},
        )
        # Oculta el flag `navigator.webdriver`, señal habitual de automatización.
        await self._context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        logger.info("browser_fetcher.started", concurrency=self._settings.scraper_concurrency)

    async def shutdown(self) -> None:
        for resource, closer in (
            (self._context, "close"),
            (self._browser, "close"),
            (self._playwright, "stop"),
        ):
            if resource is not None:
                try:
                    await getattr(resource, closer)()
                except Exception as exc:  # noqa: BLE001 - el cierre no debe romper el crawl
                    logger.warning("browser_fetcher.shutdown_error", error=str(exc))
        self._context = self._browser = self._playwright = None

    async def fetch(self, url: str, *, depth: int = 0) -> RawPage:
        if self._context is None:
            raise ScrapingError("BrowserFetcher usado sin startup()")

        async with self._semaphore:
            started = time.perf_counter()
            page = await self._context.new_page()
            try:
                await page.route(
                    "**/*",
                    lambda route: (
                        route.abort()
                        if route.request.resource_type in _BLOCKED_RESOURCE_TYPES
                        else route.continue_()
                    ),
                )
                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self._settings.scraper_timeout_seconds * 1000,
                )
                status = response.status if response else 0

                # Margen para el contenido inyectado por JS tras DOMContentLoaded.
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:  # noqa: BLE001 - timeout aceptable, seguimos
                    pass

                html = await page.content()
                elapsed_ms = int((time.perf_counter() - started) * 1000)

                if status in (401, 403, 429) or any(m in html for m in _BLOCK_MARKERS):
                    raise BlockedByTargetError(
                        url,
                        status_code=status,
                        detail="El sitio devolvió su página de bloqueo anti-bot.",
                    )
                if status >= 400:
                    raise FetchError(url, status_code=status, detail="Respuesta HTTP no exitosa")

                return RawPage(
                    url=page.url,
                    html=html,
                    status_code=status,
                    depth=depth,
                    fetcher=self.name,
                    elapsed_ms=elapsed_ms,
                )
            except (FetchError, BlockedByTargetError):
                raise
            except Exception as exc:  # timeouts y errores de navegación
                raise FetchError(url, detail=f"{type(exc).__name__}: {exc}") from exc
            finally:
                await page.close()
