"""Estrategia de descarga con navegador real (Playwright).

Necesaria para www.bbva.com.co, que está detrás de una protección anti-bot
(Akamai) que devuelve 403 a `curl`/`httpx`. Además el sitio inyecta parte del
contenido con JavaScript, así que el HTML renderizado es más rico que el
inicial.

**Hallazgo empírico que determina la implementación.** Se midieron cinco
configuraciones contra el sitio real:

    Chromium bundled, headless ................. 403 (bloqueado)
    Chromium bundled, headless + stealth JS ..... 403 (bloqueado)
    Chromium bundled, con ventana ............... 200 OK
    Google Chrome estable, headless ............. 200 OK   <-- se usa esta
    Google Chrome estable, con ventana .......... 200 OK

Es decir: el bloqueo no depende de los parches de JavaScript sino del binario.
El Chromium que empaqueta Playwright tiene una huella distinguible (códecs,
`userAgentData`, fingerprint TLS/HTTP2); Google Chrome estable en modo
headless nuevo pasa el filtro sin necesidad de ventana gráfica ni Xvfb.

Por eso se lanza con `channel="chrome"` (la imagen Docker instala Google
Chrome estable) y se degrada al Chromium empaquetado si no está disponible,
avisando de que ese camino puede acabar bloqueado.

Se reutiliza **una sola instancia de navegador** para todo el crawl y se abre
una pestaña por página: arrancar el navegador cuesta ~1 s, hacerlo por URL
sería inviable.
"""

from __future__ import annotations

import asyncio
import contextlib
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

#: Normaliza señales de automatización que algunos filtros inspeccionan. No es
#: lo que desbloquea BBVA (ver docstring del módulo) pero sí evita bloqueos en
#: sitios con detección más ingenua, y no tiene coste.
_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['es-CO', 'es', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = window.chrome || {runtime: {}};
"""

_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",  # imprescindible dentro de Docker
    "--disable-blink-features=AutomationControlled",
    "--disable-gpu",
]


class BrowserFetcher(Fetcher):
    """Descarga renderizando la página en Chromium headless."""

    name = "browser"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._playwright = None
        self._browser = None
        self._context = None
        self._channel = ""
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
        self._browser, self._channel = await self._launch_browser()

        self._context = await self._browser.new_context(
            user_agent=self._settings.scraper_user_agent,
            locale="es-CO",
            viewport={"width": 1440, "height": 900},
            extra_http_headers={"Accept-Language": "es-CO,es;q=0.9,en;q=0.8"},
        )
        await self._context.add_init_script(_STEALTH_SCRIPT)
        logger.info(
            "browser_fetcher.started",
            channel=self._channel,
            concurrency=self._settings.scraper_concurrency,
        )

    async def _launch_browser(self):
        """Lanza Google Chrome estable; si no está, degrada al Chromium empaquetado."""
        preferred = self._settings.scraper_browser_channel
        attempts = [preferred, None] if preferred else [None]

        errors: list[str] = []
        for channel in attempts:
            options = {"headless": True, "args": _LAUNCH_ARGS}
            if channel:
                options["channel"] = channel
            try:
                browser = await self._playwright.chromium.launch(**options)
            except Exception as exc:  # noqa: BLE001 - se intenta la siguiente opción
                errors.append(f"{channel or 'chromium'}: {exc}")
                continue

            if channel is None and preferred:
                logger.warning(
                    "browser_fetcher.channel_fallback",
                    requested=preferred,
                    using="chromium",
                    detail=(
                        "Google Chrome estable no está instalado. El Chromium empaquetado "
                        "es detectado por la protección anti-bot de bbva.com.co; si el "
                        "crawl devuelve 403, instala Chrome o usa la imagen Docker."
                    ),
                )
            return browser, channel or "chromium"

        await self._playwright.stop()
        raise ScrapingError(
            "No se pudo iniciar ningún navegador",
            detail=" | ".join(errors) + ". Ejecuta `playwright install --with-deps chromium`.",
        )

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
                # En bbva.com.co los beacons de analítica hacen que `networkidle`
                # casi nunca se cumpla, así que este timeout se paga completo en
                # cada página: es el factor que domina la duración del crawl. Se
                # deja configurable (`SCRAPER_SETTLE_MS`) para poder ajustar el
                # equilibrio entre velocidad y contenido renderizado.
                if self._settings.scraper_settle_ms > 0:
                    with contextlib.suppress(Exception):
                        await page.wait_for_load_state(
                            "networkidle", timeout=self._settings.scraper_settle_ms
                        )

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
