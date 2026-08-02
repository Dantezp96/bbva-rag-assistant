"""Entidades persistentes (SQLAlchemy 2.0).

Dos tablas:

* `conversations` — una sesión identificada por `id` (el ID que envía el
  cliente). Guarda contadores desnormalizados (mensajes, tokens) porque la
  analítica los consulta constantemente y recalcularlos por agregación en cada
  petición sería innecesariamente caro.
* `messages` — cada turno, con la telemetría del pipeline RAG asociada a las
  respuestas del asistente (latencias por etapa, tokens, fuentes citadas).
  Esa telemetría es exactamente la materia prima del módulo de analítica.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base declarativa del modelo de datos."""


class ConversationEntity(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(300), default="")
    user_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(String(32), default="api")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, index=True
    )

    message_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    messages: Mapped[list[MessageEntity]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageEntity.created_at",
    )


class MessageEntity(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    # --- Telemetría del pipeline (solo en mensajes del asistente) ---
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    retrieval_ms: Mapped[int] = mapped_column(Integer, default=0)
    rerank_ms: Mapped[int] = mapped_column(Integer, default=0)
    llm_ms: Mapped[int] = mapped_column(Integer, default=0)

    retrieved_count: Mapped[int] = mapped_column(Integer, default=0)
    top_score: Mapped[float] = mapped_column(Float, default=0.0)
    grounded: Mapped[bool] = mapped_column(Boolean, default=True)
    reranked: Mapped[bool] = mapped_column(Boolean, default=False)
    history_used: Mapped[int] = mapped_column(Integer, default=0)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Valoración explícita del usuario: 1 útil, -1 no útil, NULL sin valorar.
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    conversation: Mapped[ConversationEntity] = relationship(back_populates="messages")


# Consulta caliente de la memoria conversacional: "últimos N de esta sesión".
Index("ix_messages_conversation_created", MessageEntity.conversation_id, MessageEntity.created_at)
