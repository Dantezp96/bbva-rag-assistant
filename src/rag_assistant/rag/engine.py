"""Motor RAG: punto de entrada único del sistema conversacional.

PATRÓN DE DISEÑO: **Facade**.
Responder una pregunta implica coordinar seis subsistemas —memoria, embeddings,
base vectorial, reranker, constructor de prompt, LLM— más la persistencia y la
telemetría. `RAGEngine.ask()` ofrece una fachada de un solo método sobre todo
ello: la API, la CLI y la UI comparten exactamente la misma lógica y ninguna
conoce las piezas internas.

Todas las dependencias se inyectan por constructor, con valores por defecto
resueltos por las fábricas. Eso permite testear el motor con dobles (repositorio
en memoria, LLM falso) sin levantar infraestructura.
"""

from __future__ import annotations

import time

from rag_assistant.analytics.observers import (
    InMemoryMetricsObserver,
    LoggingObserver,
    QueryEvent,
    QueryEventPublisher,
)
from rag_assistant.config import Settings, get_settings
from rag_assistant.conversation.memory import ConversationMemory
from rag_assistant.conversation.repository import (
    ConversationRepository,
    SqlConversationRepository,
)
from rag_assistant.core.exceptions import (
    CollectionNotFoundError,
    LLMError,
    RAGAssistantError,
)
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import RAGAnswer
from rag_assistant.indexing.embeddings import EmbeddingProvider, get_embedding_provider
from rag_assistant.indexing.vector_store import VectorStore, get_vector_store
from rag_assistant.rag.llm import LLMProvider, get_llm_provider
from rag_assistant.rag.reranker import CrossEncoderReranker
from rag_assistant.rag.stages import (
    GenerationStage,
    PromptStage,
    QueryRewriteStage,
    RAGContext,
    RerankStage,
    RetrievalStage,
    Stage,
    SuggestionStage,
    build_chain,
)

logger = get_logger(__name__)

_EMPTY_QUESTION = "Escribe una pregunta sobre el contenido del sitio de BBVA Colombia."


class RAGEngine:
    """Fachada del sistema RAG conversacional."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        repository: ConversationRepository | None = None,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        llm: LLMProvider | None = None,
        reranker: CrossEncoderReranker | None = None,
        publisher: QueryEventPublisher | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._repository = repository or SqlConversationRepository()
        self._memory = ConversationMemory(self._repository, self._settings)
        self._embedder = embedder or get_embedding_provider()
        self._store = store or get_vector_store()
        self._llm = llm or get_llm_provider()
        self._reranker = reranker or CrossEncoderReranker(self._settings)

        self._publisher = publisher or QueryEventPublisher()
        if publisher is None:
            self._publisher.subscribe(LoggingObserver())
            self._publisher.subscribe(InMemoryMetricsObserver())

        self._chain = self._build_chain()

    # ------------------------------------------------------------ cadena ---
    def _build_chain(self):
        stages: list[Stage] = []
        # Reescribir la consulta con el historial ANTES de buscar. Añadir esta
        # capacidad fue añadir un eslabón: ni la recuperación ni la generación
        # se enteraron. Eso es lo que compra el patrón Chain of Responsibility.
        if self._settings.query_rewrite_enabled:
            stages.append(QueryRewriteStage(self._llm))
        stages.append(
            RetrievalStage(
                self._embedder,
                self._store,
                top_k=self._settings.retrieval_top_k,
                score_threshold=self._settings.retrieval_score_threshold,
            )
        )
        # El reranker es un eslabón opcional: se añade o no según configuración.
        if self._settings.reranker_enabled:
            stages.append(RerankStage(self._reranker, top_n=self._settings.reranker_top_n))
        else:
            stages.append(_TruncateStage(limit=self._settings.reranker_top_n))
        stages.append(
            PromptStage(
                context_max_chars=self._settings.context_max_chars,
                history_max_chars=self._settings.history_max_chars,
            )
        )
        stages.append(GenerationStage(self._llm))
        # Las sugerencias van al final: necesitan la respuesta ya generada para
        # no repetir lo que se acaba de contar.
        if self._settings.suggestions_enabled and self._settings.suggestions_count:
            stages.append(
                SuggestionStage(self._llm, count=self._settings.suggestions_count)
            )
        return build_chain(stages)

    # ----------------------------------------------------------- público ---
    @property
    def repository(self) -> ConversationRepository:
        return self._repository

    @property
    def publisher(self) -> QueryEventPublisher:
        return self._publisher

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
        channel: str = "api",
        history_window: int | None = None,
    ) -> RAGAnswer:
        """Responde una pregunta manteniendo el contexto de la conversación."""
        started = time.perf_counter()
        question = (question or "").strip()
        conversation = self._repository.ensure_conversation(
            conversation_id, user_id=user_id, channel=channel
        )

        if not question:
            return RAGAnswer(answer=_EMPTY_QUESTION, conversation_id=conversation, grounded=False)

        # El historial se carga ANTES de guardar la pregunta actual: si no, la
        # pregunta aparecería duplicada (en el historial y en el prompt final).
        history = self._memory.load(conversation, window=history_window)
        self._repository.add_user_message(conversation, question)

        context = RAGContext(
            question=question, conversation_id=conversation, history=history
        )

        try:
            context = self._chain.handle(context)
        except CollectionNotFoundError as exc:
            return self._fail(
                conversation, question, started,
                message=(
                    "El índice vectorial todavía está vacío. Ejecuta la ingesta "
                    "(`docker compose run --rm ingest`) y vuelve a preguntar."
                ),
                error=str(exc),
            )
        except LLMError as exc:
            return self._fail(
                conversation, question, started,
                message=f"No pude generar la respuesta: {exc.message}. {exc.detail or ''}".strip(),
                error=str(exc),
            )
        except RAGAssistantError as exc:
            return self._fail(
                conversation, question, started,
                message=f"Ocurrió un problema procesando tu pregunta: {exc.message}",
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - red de seguridad final
            logger.exception("rag.unexpected_error", error=str(exc))
            return self._fail(
                conversation, question, started,
                message="Ocurrió un error inesperado. Revisa los logs del servicio.",
                error=f"{type(exc).__name__}: {exc}",
            )

        answer = RAGAnswer(
            answer=context.answer,
            citations=context.citations,
            conversation_id=conversation,
            retrieved=context.documents or context.candidates,
            latency_ms=int((time.perf_counter() - started) * 1000),
            retrieval_ms=context.retrieval_ms,
            rerank_ms=context.rerank_ms,
            llm_ms=context.llm_ms,
            prompt_tokens=context.prompt_tokens,
            completion_tokens=context.completion_tokens,
            model=context.model or self._llm.model,
            grounded=context.grounded,
            reranked=context.reranked,
            history_used=len(history),
            search_query=context.search_query or question,
            rewritten=context.rewritten,
            rewrite_ms=context.rewrite_ms,
            suggestions=context.suggestions,
            suggestions_ms=context.suggestions_ms,
        )

        # El ID del mensaje permite que la UI envíe feedback sobre esta respuesta.
        answer.message_id = self._repository.add_assistant_message(conversation, answer)
        self._publish(question, answer, success=True)
        answer.retrieved = answer.retrieved[: self._settings.reranker_top_n]
        return answer

    def health(self) -> dict[str, object]:
        """Estado de las dependencias externas (para `/health`)."""
        store_info = self._store.info()
        return {
            "vector_store": store_info,
            "llm": {
                "provider": self._llm.name,
                "model": self._llm.model,
                "reachable": self._llm.health(),
            },
            "embeddings": {"model": self._embedder.model_name},
            "reranker": {
                "enabled": self._settings.reranker_enabled,
                "model": self._settings.reranker_model,
            },
            "history_window_size": self._settings.history_window_size,
            "indexed_chunks": store_info.get("points", 0),
        }

    def warmup(self) -> None:
        """Precarga modelos para que la primera consulta real no pague el arranque."""
        try:
            self._embedder.warmup()
            if self._settings.reranker_enabled:
                self._reranker.warmup()
        except Exception as exc:  # noqa: BLE001 - el warmup es best-effort
            logger.warning("rag.warmup_failed", error=str(exc))

    # ------------------------------------------------------------ interno --
    def _fail(
        self, conversation: str, question: str, started: float, *, message: str, error: str
    ) -> RAGAnswer:
        """Construye, persiste y publica una respuesta de error."""
        answer = RAGAnswer(
            answer=message,
            conversation_id=conversation,
            latency_ms=int((time.perf_counter() - started) * 1000),
            grounded=False,
            model=self._llm.model,
        )
        try:
            answer.message_id = self._repository.add_error_message(conversation, message, error)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag.error_persist_failed", error=str(exc))
        self._publish(question, answer, success=False, error=error)
        logger.error("rag.failed", conversation_id=conversation, error=error)
        return answer

    def _publish(
        self, question: str, answer: RAGAnswer, *, success: bool, error: str | None = None
    ) -> None:
        self._publisher.publish(
            QueryEvent(
                conversation_id=answer.conversation_id,
                question=question,
                answer_preview=answer.answer[:200],
                latency_ms=answer.latency_ms,
                retrieval_ms=answer.retrieval_ms,
                rerank_ms=answer.rerank_ms,
                llm_ms=answer.llm_ms,
                retrieved_count=len(answer.retrieved),
                top_score=max((c.effective_score for c in answer.retrieved), default=0.0),
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                model=answer.model,
                grounded=answer.grounded,
                reranked=answer.reranked,
                history_used=answer.history_used,
                success=success,
                error=error,
            )
        )


class _TruncateStage(Stage):
    """Sustituto del reranker cuando está desactivado: solo recorta a top-N.

    Mantener un eslabón en su lugar (en vez de omitirlo) deja la cadena con la
    misma forma en ambas configuraciones: las etapas siguientes siempre
    encuentran `context.documents` poblado.
    """

    name = "truncate"

    def __init__(self, *, limit: int) -> None:
        super().__init__()
        self._limit = limit

    def process(self, context: RAGContext) -> RAGContext:
        context.documents = context.candidates[: self._limit]
        context.reranked = False
        return context
