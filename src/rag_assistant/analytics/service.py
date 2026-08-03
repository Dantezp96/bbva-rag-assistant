"""Analítica del histórico de conversaciones.

Requisito de la prueba: *"Se debe incluir una funcionalidad que me permita
recorrer el histórico de conversaciones para extraer métricas y valores de
impacto"*.

El módulo recorre `conversations` + `messages` y produce cuatro familias de
métricas, elegidas por lo que cada una permite **decidir**:

1. **Volumen y adopción** — conversaciones, mensajes, usuarios, profundidad
   media de sesión y evolución diaria. Responde: ¿se está usando?
2. **Calidad de las respuestas** — tasa de respuestas fundamentadas
   (*grounded*), tasa de "no encontré información", cobertura de recuperación,
   score medio y feedback del usuario. Responde: ¿sirve?  La tasa de
   no-respuesta es la métrica accionable más directa: cada una es un vacío
   concreto del corpus indexado.
3. **Rendimiento y coste** — latencias p50/p95 desglosadas por etapa
   (recuperación / reranking / LLM), tokens y coste estimado. Responde:
   ¿es sostenible? El desglose por etapa dice *dónde* optimizar.
4. **Contenido** — temas más consultados y páginas del sitio más citadas.
   Responde: ¿qué le importa a la gente y qué contenido rinde?

Las agregaciones se calculan en SQL (no en Python) para que el coste no crezca
con el histórico.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Float, case, cast, func, select
from sqlalchemy.exc import SQLAlchemyError

from rag_assistant.conversation.db import session_scope
from rag_assistant.conversation.entities import ConversationEntity, MessageEntity
from rag_assistant.core.exceptions import ConversationError
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import Role

logger = get_logger(__name__)

#: Precio por 1M de tokens (USD): (entrada, salida). Solo para estimar coste.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def _pricing_for(model: str | None) -> tuple[float, float]:
    """Precio del modelo, tolerando los identificadores con fecha.

    La API no devuelve el alias que se le pidió sino la instantánea concreta
    que atendió la petición (`gpt-4o-mini` -> `gpt-4o-mini-2024-07-18`). Una
    búsqueda exacta falla siempre contra la respuesta real y el coste se
    reporta como 0. Se resuelve por prefijo más largo, que además absorbe
    futuras instantáneas sin tocar la tabla.
    """
    if not model:
        return (0.0, 0.0)
    normalized = model.lower().removeprefix("models/")
    if normalized in _PRICING_USD_PER_MTOK:
        return _PRICING_USD_PER_MTOK[normalized]
    matches = [key for key in _PRICING_USD_PER_MTOK if normalized.startswith(key)]
    if not matches:
        return (0.0, 0.0)
    return _PRICING_USD_PER_MTOK[max(matches, key=len)]

#: Palabras vacías del español: sin filtrarlas, el "top de temas" son artículos.
_STOPWORDS = {
    "que", "como", "para", "por", "con", "los", "las", "del", "una", "uno", "unos",
    "unas", "sus", "sobre", "cual", "cuales", "cuanto", "cuanta", "cuando", "donde",
    "hay", "son", "ser", "esta", "este", "estos", "estas", "mas", "muy", "pero",
    "puedo", "puede", "tiene", "tengo", "tienen", "quiero", "necesito", "hacer",
    "todo", "toda", "todos", "todas", "desde", "hasta", "entre", "porque", "sin",
    "les", "año", "años", "dime", "cuál", "qué", "cómo", "cuánto", "cuáles",
}


@dataclass
class AnalyticsReport:
    """Informe consolidado del histórico de conversaciones."""

    generated_at: str = ""
    period_days: int | None = None

    # 1. Volumen y adopción
    total_conversations: int = 0
    total_messages: int = 0
    total_questions: int = 0
    total_answers: int = 0
    unique_users: int = 0
    avg_messages_per_conversation: float = 0.0
    max_messages_in_conversation: int = 0
    multi_turn_conversation_rate: float = 0.0
    conversations_per_day: list[dict[str, Any]] = field(default_factory=list)

    # 2. Calidad
    grounded_rate: float = 0.0
    no_answer_rate: float = 0.0
    error_rate: float = 0.0
    avg_retrieved_chunks: float = 0.0
    avg_top_score: float = 0.0
    reranked_rate: float = 0.0
    feedback_positive: int = 0
    feedback_negative: int = 0
    satisfaction_rate: float | None = None

    # 3. Rendimiento y coste
    avg_latency_ms: float = 0.0
    p50_latency_ms: int = 0
    p95_latency_ms: int = 0
    max_latency_ms: int = 0
    avg_retrieval_ms: float = 0.0
    avg_rerank_ms: float = 0.0
    avg_llm_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_tokens_per_answer: float = 0.0
    estimated_cost_usd: float = 0.0
    cost_per_conversation_usd: float = 0.0
    models_used: dict[str, int] = field(default_factory=dict)
    #: Modelos sin tarifa conocida: su consumo NO está contado en el coste.
    untariffed_models: list[str] = field(default_factory=list)

    # 4. Contenido
    top_topics: list[dict[str, Any]] = field(default_factory=list)
    top_cited_pages: list[dict[str, Any]] = field(default_factory=list)
    unanswered_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnalyticsService:
    """Recorre el histórico y calcula las métricas de impacto."""

    def report(self, *, days: int | None = None, limit_examples: int = 10) -> AnalyticsReport:
        """Genera el informe. `days` acota el periodo; `None` = todo el histórico."""
        try:
            return self._report(days=days, limit_examples=limit_examples)
        except SQLAlchemyError as exc:
            # Sin esto, una base caída se escapaba como un 500 opaco con la
            # traza del driver. Comprobado parando el contenedor de PostgreSQL.
            logger.error("analytics.db_error", error=str(exc))
            raise ConversationError(
                "No se pudo generar el informe de analítica",
                detail="La base de datos del historial no está disponible.",
            ) from exc

    def _report(self, *, days: int | None, limit_examples: int) -> AnalyticsReport:
        report = AnalyticsReport(
            generated_at=datetime.now(UTC).isoformat(), period_days=days
        )
        since = datetime.now(UTC) - timedelta(days=days) if days else None

        with session_scope() as session:
            msg_filter = [MessageEntity.created_at >= since] if since else []
            conv_filter = [ConversationEntity.created_at >= since] if since else []

            # ---------------------------------------------- 1. volumen ------
            report.total_conversations = (
                session.execute(
                    select(func.count()).select_from(ConversationEntity).where(*conv_filter)
                ).scalar()
                or 0
            )
            report.total_messages = (
                session.execute(
                    select(func.count()).select_from(MessageEntity).where(*msg_filter)
                ).scalar()
                or 0
            )
            report.total_questions = (
                session.execute(
                    select(func.count())
                    .select_from(MessageEntity)
                    .where(MessageEntity.role == Role.USER.value, *msg_filter)
                ).scalar()
                or 0
            )
            report.total_answers = (
                session.execute(
                    select(func.count())
                    .select_from(MessageEntity)
                    .where(MessageEntity.role == Role.ASSISTANT.value, *msg_filter)
                ).scalar()
                or 0
            )
            report.unique_users = (
                session.execute(
                    select(func.count(func.distinct(ConversationEntity.user_id)))
                    .where(ConversationEntity.user_id.is_not(None), *conv_filter)
                ).scalar()
                or 0
            )

            counts = (
                session.execute(
                    select(ConversationEntity.message_count).where(*conv_filter)
                )
                .scalars()
                .all()
            )
            if counts:
                report.avg_messages_per_conversation = round(sum(counts) / len(counts), 2)
                report.max_messages_in_conversation = max(counts)
                # >2 mensajes = el usuario repreguntó: la memoria aportó valor.
                report.multi_turn_conversation_rate = round(
                    sum(1 for c in counts if c > 2) / len(counts), 4
                )

            day = func.date(ConversationEntity.created_at)
            report.conversations_per_day = [
                {"date": str(row.day), "conversations": row.total}
                for row in session.execute(
                    select(day.label("day"), func.count().label("total"))
                    .where(*conv_filter)
                    .group_by(day)
                    .order_by(day)
                ).all()
            ]

            # ---------------------------------------------- 2. calidad ------
            quality = session.execute(
                select(
                    func.count().label("total"),
                    func.sum(case((MessageEntity.grounded.is_(True), 1), else_=0)).label(
                        "grounded"
                    ),
                    func.sum(case((MessageEntity.error.is_not(None), 1), else_=0)).label("errors"),
                    func.sum(case((MessageEntity.retrieved_count == 0, 1), else_=0)).label("empty"),
                    func.sum(case((MessageEntity.reranked.is_(True), 1), else_=0)).label(
                        "reranked"
                    ),
                    func.avg(cast(MessageEntity.retrieved_count, Float)).label("avg_chunks"),
                    func.avg(MessageEntity.top_score).label("avg_score"),
                )
                .select_from(MessageEntity)
                .where(MessageEntity.role == Role.ASSISTANT.value, *msg_filter)
            ).one()

            total_answers = quality.total or 0
            if total_answers:
                report.grounded_rate = round((quality.grounded or 0) / total_answers, 4)
                report.error_rate = round((quality.errors or 0) / total_answers, 4)
                report.no_answer_rate = round((quality.empty or 0) / total_answers, 4)
                report.reranked_rate = round((quality.reranked or 0) / total_answers, 4)
                report.avg_retrieved_chunks = round(float(quality.avg_chunks or 0), 2)
                report.avg_top_score = round(float(quality.avg_score or 0), 4)

            feedback = session.execute(
                select(MessageEntity.feedback, func.count())
                .where(MessageEntity.feedback.is_not(None), *msg_filter)
                .group_by(MessageEntity.feedback)
            ).all()
            for value, count in feedback:
                if value == 1:
                    report.feedback_positive = count
                elif value == -1:
                    report.feedback_negative = count
            rated = report.feedback_positive + report.feedback_negative
            if rated:
                report.satisfaction_rate = round(report.feedback_positive / rated, 4)

            # ------------------------------------- 3. rendimiento y coste ---
            perf = session.execute(
                select(
                    func.avg(cast(MessageEntity.latency_ms, Float)),
                    func.max(MessageEntity.latency_ms),
                    func.avg(cast(MessageEntity.retrieval_ms, Float)),
                    func.avg(cast(MessageEntity.rerank_ms, Float)),
                    func.avg(cast(MessageEntity.llm_ms, Float)),
                    func.sum(MessageEntity.prompt_tokens),
                    func.sum(MessageEntity.completion_tokens),
                )
                .select_from(MessageEntity)
                .where(MessageEntity.role == Role.ASSISTANT.value, *msg_filter)
            ).one()
            (
                avg_latency, max_latency, avg_retrieval, avg_rerank, avg_llm,
                prompt_tokens, completion_tokens,
            ) = perf

            report.avg_latency_ms = round(float(avg_latency or 0), 1)
            report.max_latency_ms = int(max_latency or 0)
            report.avg_retrieval_ms = round(float(avg_retrieval or 0), 1)
            report.avg_rerank_ms = round(float(avg_rerank or 0), 1)
            report.avg_llm_ms = round(float(avg_llm or 0), 1)
            report.total_prompt_tokens = int(prompt_tokens or 0)
            report.total_completion_tokens = int(completion_tokens or 0)
            if total_answers:
                report.avg_tokens_per_answer = round(
                    (report.total_prompt_tokens + report.total_completion_tokens) / total_answers, 1
                )

            # Los percentiles se calculan en Python: `percentile_cont` no existe
            # en SQLite y el volumen de latencias es pequeño.
            latencies = sorted(
                session.execute(
                    select(MessageEntity.latency_ms).where(
                        MessageEntity.role == Role.ASSISTANT.value,
                        MessageEntity.latency_ms > 0,
                        *msg_filter,
                    )
                )
                .scalars()
                .all()
            )
            if latencies:
                report.p50_latency_ms = latencies[len(latencies) // 2]
                report.p95_latency_ms = latencies[max(0, int(len(latencies) * 0.95) - 1)]

            model_rows = session.execute(
                select(MessageEntity.model, func.count(), func.sum(MessageEntity.prompt_tokens),
                       func.sum(MessageEntity.completion_tokens))
                .where(MessageEntity.model.is_not(None), *msg_filter)
                .group_by(MessageEntity.model)
            ).all()
            cost = 0.0
            untariffed: set[str] = set()
            for model, count, prompt_sum, completion_sum in model_rows:
                report.models_used[model] = count
                price_in, price_out = _pricing_for(model)
                if not price_in and not price_out:
                    untariffed.add(model)
                cost += (prompt_sum or 0) / 1e6 * price_in
                cost += (completion_sum or 0) / 1e6 * price_out
            report.untariffed_models = sorted(untariffed)
            report.estimated_cost_usd = round(cost, 6)
            if report.total_conversations:
                report.cost_per_conversation_usd = round(
                    cost / report.total_conversations, 6
                )

            # ---------------------------------------------- 4. contenido ----
            questions = (
                session.execute(
                    select(MessageEntity.content).where(
                        MessageEntity.role == Role.USER.value, *msg_filter
                    )
                )
                .scalars()
                .all()
            )
            report.top_topics = _top_terms(questions, limit=limit_examples)

            sources = (
                session.execute(
                    select(MessageEntity.sources).where(
                        MessageEntity.sources.is_not(None), *msg_filter
                    )
                )
                .scalars()
                .all()
            )
            report.top_cited_pages = _top_pages(sources, limit=limit_examples)

            # Preguntas sin respuesta: la lista de trabajo más accionable que
            # produce este informe (qué contenido falta por indexar).
            report.unanswered_questions = _unanswered(session, msg_filter, limit=limit_examples)

        logger.info(
            "analytics.report_generated",
            conversations=report.total_conversations,
            messages=report.total_messages,
            days=days,
        )
        return report

    def popular_questions(self, *, limit: int = 4, min_repeticiones: int = 2) -> list[str]:
        """Preguntas más repetidas que el sistema SÍ supo responder.

        Alimenta los botones de arranque del chat. Se exige que la respuesta
        quedara fundamentada: proponer una pregunta que el corpus no cubre
        llevaría al usuario directo a un "no encontré esa información".

        Devuelve lista vacía si aún no hay señal suficiente; en ese caso la UI
        recurre a las preguntas configuradas en `STARTER_QUESTIONS`.
        """
        try:
            with session_scope() as session:
                # Se toman las preguntas cuya respuesta siguiente fue fundamentada.
                respondidas = session.execute(
                    select(MessageEntity.conversation_id, MessageEntity.created_at).where(
                        MessageEntity.role == Role.ASSISTANT.value,
                        MessageEntity.grounded.is_(True),
                    )
                ).all()
                contador: Counter[str] = Counter()
                for conversation_id, momento in respondidas:
                    pregunta = session.execute(
                        select(MessageEntity.content)
                        .where(
                            MessageEntity.conversation_id == conversation_id,
                            MessageEntity.role == Role.USER.value,
                            MessageEntity.created_at <= momento,
                        )
                        .order_by(MessageEntity.created_at.desc())
                        .limit(1)
                    ).scalar()
                    if pregunta and 12 <= len(pregunta) <= 120:
                        contador[pregunta.strip()] += 1
        except SQLAlchemyError as exc:
            logger.warning("analytics.popular_questions_failed", error=str(exc))
            return []

        return [q for q, n in contador.most_common(limit) if n >= min_repeticiones]

    def conversation_detail(self, conversation_id: str) -> dict[str, Any]:
        """Traza completa de una conversación (para auditar un caso concreto)."""
        from rag_assistant.conversation.repository import SqlConversationRepository

        return SqlConversationRepository().get_conversation(conversation_id)


# ------------------------------------------------------------- utilidades ---
def _top_terms(questions: list[str], *, limit: int) -> list[dict[str, Any]]:
    """Términos más frecuentes en las preguntas, sin palabras vacías."""
    counter: Counter[str] = Counter()
    for question in questions:
        for word in re.findall(r"[a-záéíóúñü]{4,}", (question or "").lower()):
            if word not in _STOPWORDS:
                counter[word] += 1
    total = sum(counter.values()) or 1
    return [
        {"term": term, "count": count, "share": round(count / total, 4)}
        for term, count in counter.most_common(limit)
    ]


def _top_pages(sources: list[Any], *, limit: int) -> list[dict[str, Any]]:
    """Páginas del sitio citadas con más frecuencia en las respuestas."""
    counter: Counter[str] = Counter()
    titles: dict[str, str] = {}
    for entry in sources:
        for source in entry or []:
            url = source.get("url")
            if not url:
                continue
            counter[url] += 1
            titles.setdefault(url, source.get("title", ""))
    return [
        {"url": url, "title": titles.get(url, ""), "citations": count}
        for url, count in counter.most_common(limit)
    ]


def _unanswered(session, msg_filter: list, *, limit: int) -> list[str]:
    """Preguntas cuya respuesta no quedó fundamentada en el corpus."""
    answers = (
        session.execute(
            select(MessageEntity.conversation_id, MessageEntity.created_at)
            .where(
                MessageEntity.role == Role.ASSISTANT.value,
                MessageEntity.grounded.is_(False),
                *msg_filter,
            )
            .order_by(MessageEntity.created_at.desc())
            .limit(limit * 2)
        )
        .all()
    )
    questions: list[str] = []
    for conversation_id, created_at in answers:
        question = session.execute(
            select(MessageEntity.content)
            .where(
                MessageEntity.conversation_id == conversation_id,
                MessageEntity.role == Role.USER.value,
                MessageEntity.created_at <= created_at,
            )
            .order_by(MessageEntity.created_at.desc())
            .limit(1)
        ).scalar()
        if question and question not in questions:
            questions.append(question)
        if len(questions) >= limit:
            break
    return questions
