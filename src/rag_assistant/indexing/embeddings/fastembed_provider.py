"""Embeddings locales con fastembed (ONNX Runtime).

Elección deliberada frente a `sentence-transformers`:

* **Sin coste**: el modelo corre en local, no hay llamadas a APIs de pago.
* **Imagen ligera**: fastembed usa onnxruntime (~60 MB) en vez de PyTorch
  (~800 MB en CPU). La imagen Docker baja de ~3 GB a menos de 1 GB.
* **Coherencia**: es la librería oficial de Qdrant, nuestra base vectorial.

El modelo por defecto (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims) está
entrenado en 50+ idiomas y rinde bien en español, que es el idioma del corpus.
"""

from __future__ import annotations

import threading

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import EmbeddingError
from rag_assistant.core.logging import get_logger
from rag_assistant.indexing.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

#: Prefijos que exigen los modelos de la familia E5 para separar pasaje/consulta.
_E5_DOC_PREFIX = "passage: "
_E5_QUERY_PREFIX = "query: "


class FastEmbedProvider(EmbeddingProvider):
    """Vectorización local vía ONNX. El modelo se carga de forma perezosa."""

    name = "fastembed"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()
        self._dimension = self._settings.embedding_dimension
        self._is_e5 = "e5" in self._settings.embedding_model.lower()

    # ------------------------------------------------------------ interno --
    def _ensure_model(self):
        """Carga diferida y thread-safe: arrancar la API no debe costar 20 s."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from fastembed import TextEmbedding

                logger.info(
                    "embeddings.loading",
                    model=self._settings.embedding_model,
                    cache=str(self._settings.model_cache_dir),
                )
                self._model = TextEmbedding(
                    model_name=self._settings.embedding_model,
                    cache_dir=str(self._settings.model_cache_dir),
                )
                probe = next(iter(self._model.embed(["dimension probe"])))
                self._dimension = len(probe)
                if self._dimension != self._settings.embedding_dimension:
                    # La configuración manda menos que la realidad del modelo:
                    # avisamos en vez de crear una colección con dimensión errónea.
                    logger.warning(
                        "embeddings.dimension_mismatch",
                        configured=self._settings.embedding_dimension,
                        actual=self._dimension,
                        action="se usa la dimensión real del modelo",
                    )
                logger.info("embeddings.ready", dimension=self._dimension)
            except Exception as exc:
                raise EmbeddingError(
                    f"No se pudo cargar el modelo de embeddings '{self._settings.embedding_model}'",
                    detail=str(exc),
                ) from exc
        return self._model

    # ------------------------------------------------------------- público --
    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        prepared = [f"{_E5_DOC_PREFIX}{t}" if self._is_e5 else t for t in texts]
        try:
            return [
                vector.tolist()
                for vector in model.embed(
                    prepared, batch_size=self._settings.embedding_batch_size
                )
            ]
        except Exception as exc:
            raise EmbeddingError("Fallo al vectorizar los documentos", detail=str(exc)) from exc

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        prepared = f"{_E5_QUERY_PREFIX}{text}" if self._is_e5 else text
        try:
            return next(iter(model.query_embed(prepared))).tolist()
        except Exception as exc:
            raise EmbeddingError("Fallo al vectorizar la consulta", detail=str(exc)) from exc
