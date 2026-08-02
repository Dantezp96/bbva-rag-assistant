"""Pipeline de indexación: corpus limpio → chunks → vectores → Qdrant.

Orquesta las tres estrategias (chunking, embeddings, vector store) sin
conocer sus implementaciones concretas: las recibe ya construidas por sus
respectivas fábricas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import IndexingError
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import Chunk, CleanDocument
from rag_assistant.indexing.chunking import ChunkingStrategy, create_chunker
from rag_assistant.indexing.embeddings import EmbeddingProvider, get_embedding_provider
from rag_assistant.indexing.vector_store import VectorStore, get_vector_store

logger = get_logger(__name__)


@dataclass
class IndexReport:
    """Resumen de una ejecución de indexación."""

    documents: int = 0
    chunks: int = 0
    vectors_written: int = 0
    #: Puntos que quedan en la colección al terminar. Puede ser MENOR que
    #: `vectors_written`: los IDs son deterministas, así que reescribir el
    #: mismo contenido actualiza el punto en lugar de duplicarlo. Ver la
    #: diferencia entre ambos es la señal de cuánta ingesta fue redundante.
    indexed_total: int = 0
    skipped_documents: int = 0
    elapsed_seconds: float = 0.0
    collection: str = ""
    embedding_model: str = ""
    dimension: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "vectors_written": self.vectors_written,
            "indexed_total": self.indexed_total,
            "skipped_documents": self.skipped_documents,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "collection": self.collection,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "errors": self.errors[:10],
        }


class IndexingPipeline:
    """Convierte documentos limpios en puntos consultables."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chunker: ChunkingStrategy | None = None,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._chunker = chunker or create_chunker(self._settings)
        self._embedder = embedder or get_embedding_provider()
        self._store = store or get_vector_store()

    def run(self, documents: list[CleanDocument], *, recreate: bool = False) -> IndexReport:
        started = time.perf_counter()
        report = IndexReport(
            collection=self._settings.qdrant_collection,
            embedding_model=self._embedder.model_name,
        )
        if not documents:
            raise IndexingError(
                "No hay documentos que indexar",
                detail="Ejecuta primero el scraping (`rag-assistant scrape`).",
            )

        dimension = self._embedder.dimension
        report.dimension = dimension
        self._store.ensure_collection(dimension, recreate=recreate)

        # Se procesa documento a documento: un fallo aislado (página rara) no
        # tumba toda la ingesta, y el uso de memoria se mantiene acotado.
        buffer: list[Chunk] = []
        for document in documents:
            try:
                chunks = self._chunker.chunk_document(document)
            except Exception as exc:  # noqa: BLE001
                report.skipped_documents += 1
                report.errors.append(f"{document.url}: chunking falló ({exc})")
                continue
            if not chunks:
                report.skipped_documents += 1
                continue

            report.documents += 1
            report.chunks += len(chunks)
            buffer.extend(chunks)

            if len(buffer) >= self._settings.embedding_batch_size * 4:
                report.vectors_written += self._flush(buffer)
                buffer = []

        if buffer:
            report.vectors_written += self._flush(buffer)

        report.indexed_total = self._store.count()
        if report.indexed_total < report.vectors_written:
            logger.info(
                "indexing.deduplicated_by_id",
                written=report.vectors_written,
                in_collection=report.indexed_total,
                note="IDs deterministas: contenido repetido actualizó puntos existentes",
            )
        report.elapsed_seconds = time.perf_counter() - started
        logger.info("indexing.finished", **report.as_dict())
        return report

    def _flush(self, chunks: list[Chunk]) -> int:
        """Vectoriza y escribe un lote de chunks."""
        try:
            vectors = self._embedder.embed_documents([c.text for c in chunks])
            written = self._store.upsert(chunks, vectors)
        except Exception as exc:
            raise IndexingError("Fallo al indexar un lote de chunks", detail=str(exc)) from exc
        logger.info("indexing.batch", chunks=len(chunks), written=written)
        return written
