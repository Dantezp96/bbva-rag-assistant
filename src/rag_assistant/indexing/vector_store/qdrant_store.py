"""Adaptador de Qdrant.

Qdrant se eligió por: licencia Apache-2.0, self-hosted con un solo contenedor
(sin dependencias externas ni cuenta cloud), filtrado por payload nativo, y
persistencia en disco vía volumen Docker.
"""

from __future__ import annotations

import uuid
from typing import Any

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import CollectionNotFoundError, VectorStoreError
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import Chunk, RetrievedChunk
from rag_assistant.indexing.vector_store.base import VectorStore

logger = get_logger(__name__)

_UPSERT_BATCH = 128


class QdrantVectorStore(VectorStore):
    """Traduce las operaciones del dominio a la API de `qdrant_client`."""

    name = "qdrant"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._collection = self._settings.qdrant_collection
        self._client = None

    # ------------------------------------------------------------ interno --
    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(
                    url=self._settings.qdrant_url,
                    api_key=self._settings.qdrant_api_key,
                    timeout=60,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"No se pudo conectar con Qdrant en {self._settings.qdrant_url}",
                    detail=str(exc),
                ) from exc
        return self._client

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Qdrant exige UUID o entero: derivamos un UUID estable del id del chunk."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _distance(self):
        from qdrant_client.models import Distance

        return {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
        }[self._settings.qdrant_distance]

    # ------------------------------------------------------------ público --
    def ensure_collection(self, dimension: int, *, recreate: bool = False) -> None:
        from qdrant_client.models import PayloadSchemaType, VectorParams

        try:
            exists = self.client.collection_exists(self._collection)
            if exists and recreate:
                self.client.delete_collection(self._collection)
                exists = False
                logger.info("vector_store.collection_dropped", collection=self._collection)

            if not exists:
                self.client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=dimension, distance=self._distance()),
                )
                # Índices de payload: permiten filtrar por sección/URL sin
                # escaneo completo (útil para búsquedas acotadas).
                for field in ("url", "section"):
                    self.client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                logger.info(
                    "vector_store.collection_created",
                    collection=self._collection,
                    dimension=dimension,
                    distance=self._settings.qdrant_distance,
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"No se pudo preparar la colección '{self._collection}'", detail=str(exc)
            ) from exc

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                "Desajuste entre chunks y vectores",
                detail=f"{len(chunks)} chunks vs {len(vectors)} vectores",
            )

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=self._point_id(c.id), vector=v, payload=c.payload())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        written = 0
        try:
            for start in range(0, len(points), _UPSERT_BATCH):
                batch = points[start : start + _UPSERT_BATCH]
                self.client.upsert(collection_name=self._collection, points=batch, wait=True)
                written += len(batch)
                logger.debug("vector_store.batch_upserted", written=written, total=len(points))
        except Exception as exc:
            raise VectorStoreError("Fallo al escribir en Qdrant", detail=str(exc)) from exc
        return written

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filters:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )

        try:
            response = self.client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:
            message = str(exc)
            if "doesn't exist" in message or "Not found" in message or "404" in message:
                raise CollectionNotFoundError(
                    f"La colección '{self._collection}' no existe",
                    detail="Ejecuta la ingesta: `docker compose run --rm ingest`",
                ) from exc
            raise VectorStoreError("Fallo en la búsqueda vectorial", detail=message) from exc

        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    id=str(point.id),
                    text=payload.get("text", ""),
                    url=payload.get("url", ""),
                    title=payload.get("title", ""),
                    section=payload.get("section", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    score=float(point.score),
                )
            )
        return results

    def count(self) -> int:
        try:
            return self.client.count(self._collection, exact=True).count
        except Exception:  # noqa: BLE001 - la colección puede no existir aún
            return 0

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    def info(self) -> dict[str, Any]:
        try:
            exists = self.client.collection_exists(self._collection)
        except Exception as exc:  # noqa: BLE001
            return {"provider": self.name, "reachable": False, "error": str(exc)}
        return {
            "provider": self.name,
            "reachable": True,
            "collection": self._collection,
            "exists": exists,
            "points": self.count() if exists else 0,
            "url": self._settings.qdrant_url,
        }
