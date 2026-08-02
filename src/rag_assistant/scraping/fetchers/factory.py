"""Fábrica de estrategias de descarga.

PATRÓN DE DISEÑO: **Factory Method**.
Concentra en un único punto la decisión de qué implementación de `Fetcher`
construir a partir de la configuración. El crawler pide "un fetcher" y recibe
uno ya configurado, sin importar ni conocer las clases concretas.
"""

from __future__ import annotations

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import ConfigurationError
from rag_assistant.scraping.fetchers.base import Fetcher
from rag_assistant.scraping.fetchers.browser_fetcher import BrowserFetcher
from rag_assistant.scraping.fetchers.http_fetcher import HttpFetcher

_REGISTRY: dict[str, type[Fetcher]] = {
    "http": HttpFetcher,
    "browser": BrowserFetcher,
}


def create_fetcher(settings: Settings | None = None, *, kind: str | None = None) -> Fetcher:
    """Instancia la estrategia de descarga indicada por configuración."""
    settings = settings or get_settings()
    resolved = (kind or settings.scraper_fetcher).lower()
    try:
        return _REGISTRY[resolved](settings)
    except KeyError as exc:
        raise ConfigurationError(
            f"Fetcher desconocido: '{resolved}'",
            detail=f"Opciones válidas: {', '.join(sorted(_REGISTRY))}",
        ) from exc


def register_fetcher(kind: str, cls: type[Fetcher]) -> None:
    """Permite añadir estrategias desde fuera (p. ej. tests o plugins)."""
    _REGISTRY[kind.lower()] = cls
