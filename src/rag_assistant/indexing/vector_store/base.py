"""Contrato de la base de datos vectorial.

PATRÓN DE DISEÑO: **Adapter**.
Cada base vectorial (Qdrant, Chroma, Milvus, pgvector…) expone una API
distinta e incompatible. `VectorStore` define el vocabulario que necesita
*nuestro* dominio —`upsert(chunks, vectors)`, `search(vector, top_k)`— y cada
adaptador traduce ese vocabulario a la API concreta del proveedor.

Consecuencia práctica: el motor RAG nunca importa `qdrant_client`. Sustituir
Qdrant por otra base es escribir un adaptador nuevo, sin tocar la lógica.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_assistant.core.models import Chunk, RetrievedChunk


class VectorStore(ABC):
    """Adaptador sobre una base de datos vectorial."""

    name: str = "base"

    @abstractmethod
    def ensure_collection(self, dimension: int, *, recreate: bool = False) -> None:
        """Crea la colección si no existe (o la recrea si `recreate`)."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Inserta o actualiza puntos. Devuelve cuántos se escribieron."""

    @abstractmethod
    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Busca los `top_k` vecinos más próximos."""

    @abstractmethod
    def count(self) -> int:
        """Número de puntos indexados."""

    @abstractmethod
    def health(self) -> bool:
        """`True` si el servicio responde."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Metadatos de la colección (para el endpoint de estado)."""
