"""Fábrica de proveedores de embeddings (Factory Method + caché de instancia).

Cargar un modelo ONNX cuesta segundos y varios cientos de MB de RAM: la
fábrica cachea la instancia por proceso para que API, CLI e indexador
compartan el mismo modelo ya cargado.
"""

from __future__ import annotations

from functools import lru_cache

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import ConfigurationError
from rag_assistant.indexing.embeddings.base import EmbeddingProvider
from rag_assistant.indexing.embeddings.fastembed_provider import FastEmbedProvider

_REGISTRY: dict[str, type[EmbeddingProvider]] = {"fastembed": FastEmbedProvider}


def create_embedding_provider(
    settings: Settings | None = None, *, provider: str | None = None
) -> EmbeddingProvider:
    settings = settings or get_settings()
    resolved = (provider or settings.embedding_provider).lower()
    try:
        return _REGISTRY[resolved](settings)
    except KeyError as exc:
        raise ConfigurationError(
            f"Proveedor de embeddings desconocido: '{resolved}'",
            detail=f"Opciones válidas: {', '.join(sorted(_REGISTRY))}",
        ) from exc


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Instancia compartida por proceso (evita recargar el modelo ONNX)."""
    return create_embedding_provider()
