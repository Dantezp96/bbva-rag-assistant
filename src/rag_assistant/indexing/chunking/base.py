"""Contrato de troceado de documentos.

PATRÓN DE DISEÑO: **Strategy**.
El tamaño y la forma de los chunks es la palanca que más afecta a la calidad
de un RAG, y la estrategia óptima depende del corpus. Aislarla tras una
interfaz permite experimentar (`CHUNK_STRATEGY=recursive|fixed`) sin tocar el
pipeline de indexación.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from rag_assistant.core.models import Chunk, CleanDocument


class ChunkingStrategy(ABC):
    """Divide un documento limpio en fragmentos indexables."""

    name: str = "base"

    def __init__(self, chunk_size: int, chunk_overlap: int, min_size: int = 0) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_size = min_size

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Devuelve los fragmentos de texto crudos."""

    def chunk_document(self, document: CleanDocument) -> list[Chunk]:
        """Aplica la estrategia y envuelve el resultado en objetos `Chunk`."""
        pieces = [p.strip() for p in self.split(document.text) if p.strip()]
        pieces = [p for p in pieces if len(p) >= self.min_size] or pieces[:1]

        total = len(pieces)
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            # ID determinista: reindexar el mismo contenido actualiza el punto
            # en vez de duplicarlo (ingesta idempotente).
            raw_id = f"{document.url}::{index}::{hashlib.sha1(piece.encode()).hexdigest()[:8]}"
            chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:32]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    url=document.url,
                    title=document.title,
                    section=document.section,
                    chunk_index=index,
                    total_chunks=total,
                    metadata={
                        "language": document.language,
                        "scraped_at": document.scraped_at.isoformat(),
                        "content_hash": document.content_hash,
                        "chunk_strategy": self.name,
                    },
                )
            )
        return chunks
