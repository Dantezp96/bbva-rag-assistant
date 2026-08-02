"""Fábrica de adaptadores de base vectorial (Factory Method)."""

from __future__ import annotations

from functools import lru_cache

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import ConfigurationError
from rag_assistant.indexing.vector_store.base import VectorStore
from rag_assistant.indexing.vector_store.qdrant_store import QdrantVectorStore

_REGISTRY: dict[str, type[VectorStore]] = {"qdrant": QdrantVectorStore}


def create_vector_store(
    settings: Settings | None = None, *, provider: str | None = None
) -> VectorStore:
    settings = settings or get_settings()
    resolved = (provider or settings.vector_store_provider).lower()
    try:
        return _REGISTRY[resolved](settings)
    except KeyError as exc:
        raise ConfigurationError(
            f"Base vectorial desconocida: '{resolved}'",
            detail=f"Opciones válidas: {', '.join(sorted(_REGISTRY))}",
        ) from exc


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Instancia compartida (reutiliza el pool de conexiones del cliente)."""
    return create_vector_store()
