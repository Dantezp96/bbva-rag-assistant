"""Reranking con cross-encoder (BONUS de la prueba).

**Por qué hace falta.** La búsqueda vectorial compara dos vectores calculados
por separado (bi-encoder): es rapidísima sobre millones de puntos, pero la
consulta y el pasaje nunca "se ven". Eso produce falsos positivos: un chunk
sobre *tarjeta débito* puntúa alto ante una pregunta sobre *tarjeta de
crédito* porque el vocabulario se solapa.

Un cross-encoder procesa el par `(consulta, pasaje)` **junto**, con atención
cruzada, y estima la relevancia real. Es órdenes de magnitud más lento, así
que se aplica solo al top-K ya reducido:

    consulta → vectorial (top_k=20) → cross-encoder → top_n=5 → LLM

Efecto neto: el LLM recibe menos contexto pero mucho mejor, lo que sube la
fidelidad de la respuesta y baja el coste en tokens.

**Degradación elegante:** si el modelo no puede cargarse (sin red en el primer
arranque, disco lleno), se registra el aviso y se continúa con el orden
vectorial. Un reranker caído nunca debe dejar el chat sin servicio.
"""

from __future__ import annotations

import threading
import time

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import RetrievedChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reordena los candidatos recuperados según relevancia consulta-pasaje."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()
        self._unavailable = False

    @property
    def enabled(self) -> bool:
        return self._settings.reranker_enabled and not self._unavailable

    @property
    def model_name(self) -> str:
        return self._settings.reranker_model

    def _ensure_model(self):
        """Carga diferida y thread-safe del cross-encoder."""
        if self._model is not None or self._unavailable:
            return self._model
        with self._lock:
            if self._model is not None or self._unavailable:
                return self._model
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info(
                    "reranker.loading",
                    model=self._settings.reranker_model,
                    threads=self._settings.effective_onnx_threads,
                )
                self._model = TextCrossEncoder(
                    model_name=self._settings.reranker_model,
                    cache_dir=str(self._settings.model_cache_dir),
                    threads=self._settings.effective_onnx_threads,
                )
                logger.info("reranker.ready", model=self._settings.reranker_model)
            except Exception as exc:  # noqa: BLE001 - degradación deliberada
                self._unavailable = True
                logger.warning(
                    "reranker.unavailable",
                    model=self._settings.reranker_model,
                    error=str(exc),
                    action="se continúa con el orden de la búsqueda vectorial",
                )
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_n: int | None = None
    ) -> tuple[list[RetrievedChunk], int, bool]:
        """Reordena y recorta los candidatos.

        Returns:
            (documentos, milisegundos, se_aplicó_reranking)
        """
        limit = top_n or self._settings.reranker_top_n
        if not candidates:
            return [], 0, False
        if not self.enabled:
            return candidates[:limit], 0, False

        model = self._ensure_model()
        if model is None:
            return candidates[:limit], 0, False

        started = time.perf_counter()
        try:
            scores = list(model.rerank(query, [c.text for c in candidates]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("reranker.scoring_failed", error=str(exc))
            return candidates[:limit], 0, False

        for chunk, score in zip(candidates, scores, strict=True):
            chunk.rerank_score = float(score)

        ordered = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[:limit]
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.debug(
            "reranker.applied",
            candidates=len(candidates),
            kept=len(ordered),
            ms=elapsed_ms,
        )
        return ordered, elapsed_ms, True

    def warmup(self) -> None:
        self._ensure_model()
