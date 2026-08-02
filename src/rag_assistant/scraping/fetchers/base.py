"""Contrato de descarga de páginas.

PATRÓN DE DISEÑO: **Strategy**.
`Fetcher` define *qué* hace un descargador (`fetch(url) -> RawPage`) sin fijar
*cómo*. Existen dos implementaciones intercambiables en caliente:

* `HttpFetcher`    — httpx puro: rápido y ligero.
* `BrowserFetcher` — Playwright/Chromium: lento pero ejecuta JavaScript y
  supera la protección anti-bot de www.bbva.com.co (que responde 403 a
  cualquier cliente HTTP plano).

El crawler depende únicamente de esta interfaz, así que cambiar de estrategia
es un cambio de variable de entorno (`SCRAPER_FETCHER`), no de código.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from rag_assistant.core.models import RawPage


class Fetcher(ABC):
    """Estrategia de descarga de una URL."""

    name: str = "base"

    async def __aenter__(self) -> Fetcher:
        await self.startup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.shutdown()

    # Hooks opcionales, no abstractos a propósito: una estrategia sin recursos
    # que reservar (un doble de test, por ejemplo) no debería verse obligada a
    # implementarlos.
    async def startup(self) -> None:  # noqa: B027
        """Reserva recursos costosos (navegador, pool de conexiones)."""

    async def shutdown(self) -> None:  # noqa: B027
        """Libera los recursos reservados en `startup`."""

    @abstractmethod
    async def fetch(self, url: str, *, depth: int = 0) -> RawPage:
        """Descarga `url`.

        Raises:
            FetchError: la descarga falló.
            BlockedByTargetError: el sitio bloqueó la petición (403/429).
        """
