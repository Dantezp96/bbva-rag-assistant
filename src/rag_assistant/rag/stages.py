"""Etapas del pipeline RAG.

PATRÓN DE DISEÑO: **Chain of Responsibility**.
Responder una pregunta es una secuencia de pasos —recuperar, reordenar,
construir el prompt, generar— donde cada paso enriquece un contexto compartido
y decide si pasa el testigo al siguiente. Modelarlo como una cadena aporta:

* **Composición**: activar o desactivar el reranker es añadir o quitar un
  eslabón, no meter un `if` en un método de 200 líneas.
* **Cortocircuito**: si la recuperación no encuentra nada relevante, la etapa
  puede terminar la cadena y devolver una respuesta honesta sin gastar una
  llamada al LLM.
* **Observabilidad**: cada eslabón mide su propio tiempo, lo que da la
  telemetría por etapa que consume el módulo de analítica.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import Citation, Message, RetrievedChunk
from rag_assistant.indexing.embeddings import EmbeddingProvider
from rag_assistant.indexing.vector_store import VectorStore
from rag_assistant.rag.llm import LLMProvider
from rag_assistant.rag.prompt_builder import PromptBuilder
from rag_assistant.rag.reranker import CrossEncoderReranker

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "No encontré información sobre eso en el contenido indexado del sitio de "
    "BBVA Colombia. Puedes reformular la pregunta con otros términos, o "
    "consultar directamente en https://www.bbva.com.co/."
)


@dataclass
class RAGContext:
    """Estado que atraviesa la cadena, enriquecido en cada etapa."""

    question: str
    conversation_id: str = ""
    history: list[Message] = field(default_factory=list)

    candidates: list[RetrievedChunk] = field(default_factory=list)
    documents: list[RetrievedChunk] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    answer: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    retrieval_ms: int = 0
    rerank_ms: int = 0
    llm_ms: int = 0

    reranked: bool = False
    grounded: bool = True
    stop: bool = False  # cortocircuita el resto de la cadena


class Stage(ABC):
    """Eslabón de la cadena."""

    name: str = "stage"

    def __init__(self) -> None:
        self._next: Stage | None = None

    def set_next(self, stage: Stage) -> Stage:
        """Encadena y devuelve el siguiente eslabón (permite encadenado fluido)."""
        self._next = stage
        return stage

    def handle(self, context: RAGContext) -> RAGContext:
        context = self.process(context)
        if context.stop or self._next is None:
            return context
        return self._next.handle(context)

    @abstractmethod
    def process(self, context: RAGContext) -> RAGContext:
        """Trabajo propio de la etapa."""


class RetrievalStage(Stage):
    """Vectoriza la pregunta y recupera los candidatos del índice."""

    name = "retrieval"

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        top_k: int,
        score_threshold: float,
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold

    def process(self, context: RAGContext) -> RAGContext:
        started = time.perf_counter()
        vector = self._embedder.embed_query(context.question)
        candidates = self._store.search(
            vector, top_k=self._top_k, score_threshold=self._score_threshold
        )
        context.candidates = candidates
        context.retrieval_ms = int((time.perf_counter() - started) * 1000)

        if not candidates:
            # Sin evidencia no se llama al LLM: se ahorra coste y se evita que
            # el modelo rellene el hueco con conocimiento general.
            context.answer = NO_CONTEXT_ANSWER
            context.grounded = False
            context.stop = True
            logger.info("rag.no_context", question=context.question[:80])

        logger.debug("rag.retrieved", n=len(candidates), ms=context.retrieval_ms)
        return context


class RerankStage(Stage):
    """Reordena los candidatos con un cross-encoder (bonus)."""

    name = "rerank"

    def __init__(self, reranker: CrossEncoderReranker, *, top_n: int) -> None:
        super().__init__()
        self._reranker = reranker
        self._top_n = top_n

    def process(self, context: RAGContext) -> RAGContext:
        documents, elapsed_ms, applied = self._reranker.rerank(
            context.question, context.candidates, top_n=self._top_n
        )
        context.documents = documents
        context.rerank_ms = elapsed_ms
        context.reranked = applied
        return context


class PromptStage(Stage):
    """Ensambla el prompt final con el Builder."""

    name = "prompt"

    def __init__(self, *, context_max_chars: int, history_max_chars: int) -> None:
        super().__init__()
        self._context_max_chars = context_max_chars
        self._history_max_chars = history_max_chars

    def process(self, context: RAGContext) -> RAGContext:
        builder = (
            PromptBuilder(
                context_max_chars=self._context_max_chars,
                history_max_chars=self._history_max_chars,
            )
            .with_history(context.history)
            .with_context(context.documents)
            .with_question(context.question)
        )
        context.messages, context.citations = builder.build()
        return context


class GenerationStage(Stage):
    """Invoca al LLM y recoge la respuesta y el consumo de tokens."""

    name = "generation"

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__()
        self._llm = llm

    def process(self, context: RAGContext) -> RAGContext:
        started = time.perf_counter()
        response = self._llm.complete(context.messages)
        context.llm_ms = int((time.perf_counter() - started) * 1000)
        context.answer = response.content
        context.model = response.model
        context.prompt_tokens = response.prompt_tokens
        context.completion_tokens = response.completion_tokens

        # Heurística de "grounding": si el modelo no citó ninguna fuente y
        # además declaró no haber encontrado la información, la respuesta no
        # está respaldada por el corpus. Se registra para la analítica.
        lowered = context.answer.lower()
        cited = any(f"[{c.index}]" in context.answer for c in context.citations)
        context.grounded = cited or "no encontré" not in lowered

        logger.info(
            "rag.generated",
            model=context.model,
            ms=context.llm_ms,
            tokens=response.total_tokens,
            cited=cited,
        )
        return context


def build_chain(stages: list[Stage]) -> Stage:
    """Encadena las etapas y devuelve la cabeza de la cadena."""
    if not stages:
        raise ValueError("La cadena RAG necesita al menos una etapa")
    for current, following in zip(stages, stages[1:], strict=False):
        current.set_next(following)
    return stages[0]
