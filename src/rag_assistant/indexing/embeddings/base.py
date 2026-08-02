"""Contrato de generación de embeddings.

PATRÓN DE DISEÑO: **Strategy**.
El modelo de embeddings es una decisión reversible: hoy fastembed/ONNX local
(gratuito, sin torch), mañana quizá una API gestionada. Todo el sistema
depende solo de esta interfaz.

`embed_documents` y `embed_query` están separados a propósito: varias familias
de modelos (E5, BGE, Instructor) exigen prefijos distintos para pasajes y
consultas, y mezclarlos degrada la recuperación de forma silenciosa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Estrategia de vectorización de texto."""

    name: str = "base"

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensión de los vectores generados."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identificador del modelo en uso."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza pasajes que se van a indexar."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Vectoriza una consulta de usuario."""

    def warmup(self) -> None:
        """Fuerza la carga del modelo para que la primera consulta real no pague el coste."""
        self.embed_query("calentamiento")
