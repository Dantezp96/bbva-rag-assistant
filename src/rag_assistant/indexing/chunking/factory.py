"""Fábrica de estrategias de troceado (Factory Method)."""

from __future__ import annotations

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import ConfigurationError
from rag_assistant.indexing.chunking.base import ChunkingStrategy
from rag_assistant.indexing.chunking.strategies import FixedChunker, RecursiveChunker

_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "recursive": RecursiveChunker,
    "fixed": FixedChunker,
}


def create_chunker(settings: Settings | None = None, *, strategy: str | None = None) -> ChunkingStrategy:
    settings = settings or get_settings()
    resolved = (strategy or settings.chunk_strategy).lower()
    try:
        cls = _REGISTRY[resolved]
    except KeyError as exc:
        raise ConfigurationError(
            f"Estrategia de chunking desconocida: '{resolved}'",
            detail=f"Opciones válidas: {', '.join(sorted(_REGISTRY))}",
        ) from exc
    return cls(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_size=settings.chunk_min_size,
    )
