"""Acceso al historial de conversaciones.

PATRÓN DE DISEÑO: **Repository**.
El motor RAG necesita "los últimos N mensajes de la conversación X" y
"guarda este turno". No necesita saber que existe SQLAlchemy, ni una sesión,
ni una transacción. `ConversationRepository` expone ese vocabulario de
dominio y `SqlConversationRepository` lo implementa contra SQL.

Beneficios concretos aquí:
* los tests usan `InMemoryConversationRepository` y corren sin base de datos;
* migrar de SQLite a PostgreSQL (o mañana a Redis) no toca el motor RAG;
* toda la lógica de transacciones queda confinada a un único módulo.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from sqlalchemy import desc, select

from rag_assistant.conversation.db import session_scope
from rag_assistant.conversation.entities import ConversationEntity, MessageEntity
from rag_assistant.core.exceptions import ConversationError, ConversationNotFoundError
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import Message, RAGAnswer, Role

logger = get_logger(__name__)


class ConversationRepository(ABC):
    """Contrato de persistencia del historial."""

    @abstractmethod
    def ensure_conversation(
        self,
        conversation_id: str | None = None,
        *,
        user_id: str | None = None,
        channel: str = "api",
    ) -> str:
        """Devuelve el ID de la conversación, creándola si hace falta."""

    @abstractmethod
    def get_recent_messages(self, conversation_id: str, limit: int) -> list[Message]:
        """Últimos `limit` mensajes en orden cronológico."""

    @abstractmethod
    def add_user_message(self, conversation_id: str, content: str) -> int:
        """Persiste el turno del usuario y devuelve su ID."""

    @abstractmethod
    def add_assistant_message(self, conversation_id: str, answer: RAGAnswer) -> int:
        """Persiste la respuesta con toda su telemetría."""

    @abstractmethod
    def add_error_message(self, conversation_id: str, message: str, error: str) -> int:
        """Registra un turno fallido (para no perderlo en las métricas)."""

    @abstractmethod
    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> list[dict]:
        """Listado paginado de conversaciones, más recientes primero."""

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> dict:
        """Conversación completa con todos sus mensajes."""

    @abstractmethod
    def set_feedback(self, message_id: int, value: int) -> None:
        """Valoración del usuario sobre una respuesta (1 / -1)."""

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> None:
        """Elimina una conversación y sus mensajes."""


class SqlConversationRepository(ConversationRepository):
    """Implementación sobre SQLAlchemy (PostgreSQL o SQLite)."""

    def ensure_conversation(
        self,
        conversation_id: str | None = None,
        *,
        user_id: str | None = None,
        channel: str = "api",
    ) -> str:
        resolved = (conversation_id or "").strip() or f"conv-{uuid.uuid4().hex[:12]}"
        try:
            with session_scope() as session:
                existing = session.get(ConversationEntity, resolved)
                if existing is None:
                    session.add(
                        ConversationEntity(id=resolved, user_id=user_id, channel=channel)
                    )
                    logger.info("conversation.created", conversation_id=resolved)
        except Exception as exc:
            raise ConversationError(
                "No se pudo crear/recuperar la conversación", detail=str(exc)
            ) from exc
        return resolved

    def get_recent_messages(self, conversation_id: str, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        with session_scope() as session:
            rows = (
                session.execute(
                    select(MessageEntity)
                    .where(MessageEntity.conversation_id == conversation_id)
                    .order_by(desc(MessageEntity.created_at), desc(MessageEntity.id))
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        # Se piden los N más recientes y se devuelven en orden cronológico.
        return [
            Message(
                id=row.id,
                role=Role(row.role),
                content=row.content,
                created_at=row.created_at,
            )
            for row in reversed(rows)
        ]

    def add_user_message(self, conversation_id: str, content: str) -> int:
        with session_scope() as session:
            message = MessageEntity(
                conversation_id=conversation_id, role=Role.USER.value, content=content
            )
            session.add(message)
            session.flush()
            self._touch(session, conversation_id, tokens=0, title_hint=content)
            return message.id

    def add_assistant_message(self, conversation_id: str, answer: RAGAnswer) -> int:
        with session_scope() as session:
            message = MessageEntity(
                conversation_id=conversation_id,
                role=Role.ASSISTANT.value,
                content=answer.answer,
                model=answer.model,
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                latency_ms=answer.latency_ms,
                retrieval_ms=answer.retrieval_ms,
                rerank_ms=answer.rerank_ms,
                llm_ms=answer.llm_ms,
                retrieved_count=len(answer.retrieved),
                # Se persiste la similitud vectorial, no la del reranker: es la
                # única comparable entre consultas (ver `Citation`).
                top_score=round(max((c.score for c in answer.retrieved), default=0.0), 4),
                grounded=answer.grounded,
                reranked=answer.reranked,
                history_used=answer.history_used,
                sources=[
                    {"url": c.url, "title": c.title, "score": round(c.score, 4)}
                    for c in answer.citations
                ],
            )
            session.add(message)
            session.flush()
            self._touch(session, conversation_id, tokens=answer.total_tokens)
            return message.id

    def add_error_message(self, conversation_id: str, message: str, error: str) -> int:
        with session_scope() as session:
            entity = MessageEntity(
                conversation_id=conversation_id,
                role=Role.ASSISTANT.value,
                content=message,
                grounded=False,
                error=error,
            )
            session.add(entity)
            session.flush()
            self._touch(session, conversation_id, tokens=0)
            return entity.id

    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> list[dict]:
        with session_scope() as session:
            rows = (
                session.execute(
                    select(ConversationEntity)
                    .order_by(desc(ConversationEntity.updated_at))
                    .limit(limit)
                    .offset(offset)
                )
                .scalars()
                .all()
            )
            return [
                {
                    "id": row.id,
                    "title": row.title,
                    "user_id": row.user_id,
                    "channel": row.channel,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                    "message_count": row.message_count,
                    "total_tokens": row.total_tokens,
                }
                for row in rows
            ]

    def get_conversation(self, conversation_id: str) -> dict:
        with session_scope() as session:
            conversation = session.get(ConversationEntity, conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(
                    f"No existe la conversación '{conversation_id}'"
                )
            messages = (
                session.execute(
                    select(MessageEntity)
                    .where(MessageEntity.conversation_id == conversation_id)
                    .order_by(MessageEntity.created_at, MessageEntity.id)
                )
                .scalars()
                .all()
            )
            return {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "message_count": conversation.message_count,
                "total_tokens": conversation.total_tokens,
                "messages": [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "created_at": m.created_at.isoformat(),
                        "model": m.model,
                        "latency_ms": m.latency_ms,
                        "total_tokens": m.prompt_tokens + m.completion_tokens,
                        "grounded": m.grounded,
                        "reranked": m.reranked,
                        "sources": m.sources or [],
                        "feedback": m.feedback,
                        "error": m.error,
                    }
                    for m in messages
                ],
            }

    def set_feedback(self, message_id: int, value: int) -> None:
        if value not in (-1, 1):
            raise ConversationError("El feedback debe ser 1 (útil) o -1 (no útil)")
        with session_scope() as session:
            message = session.get(MessageEntity, message_id)
            if message is None:
                raise ConversationNotFoundError(f"No existe el mensaje {message_id}")
            message.feedback = value

    def delete_conversation(self, conversation_id: str) -> None:
        with session_scope() as session:
            conversation = session.get(ConversationEntity, conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(f"No existe la conversación '{conversation_id}'")
            session.delete(conversation)

    # ------------------------------------------------------------ interno --
    @staticmethod
    def _touch(session, conversation_id: str, *, tokens: int, title_hint: str = "") -> None:
        """Actualiza contadores desnormalizados y el título de la conversación."""
        conversation = session.get(ConversationEntity, conversation_id)
        if conversation is None:
            conversation = ConversationEntity(id=conversation_id)
            session.add(conversation)
        conversation.message_count += 1
        conversation.total_tokens += tokens
        conversation.updated_at = datetime.now(UTC)
        if title_hint and not conversation.title:
            # El título es la primera pregunta: identifica la sesión de un vistazo.
            conversation.title = title_hint.strip()[:120]


class InMemoryConversationRepository(ConversationRepository):
    """Implementación en memoria para tests y ejecución sin base de datos."""

    def __init__(self) -> None:
        self._conversations: dict[str, dict] = {}
        self._messages: dict[str, list[dict]] = {}
        self._next_id = 1

    def ensure_conversation(
        self,
        conversation_id: str | None = None,
        *,
        user_id: str | None = None,
        channel: str = "api",
    ) -> str:
        resolved = (conversation_id or "").strip() or f"conv-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)
        self._conversations.setdefault(
            resolved,
            {
                "id": resolved,
                "title": "",
                "user_id": user_id,
                "channel": channel,
                "created_at": now,
                "updated_at": now,
                "message_count": 0,
                "total_tokens": 0,
            },
        )
        self._messages.setdefault(resolved, [])
        return resolved

    def get_recent_messages(self, conversation_id: str, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        stored = self._messages.get(conversation_id, [])[-limit:]
        return [
            Message(
                id=m["id"],
                role=Role(m["role"]),
                content=m["content"],
                created_at=m["created_at"],
            )
            for m in stored
        ]

    def _append(self, conversation_id: str, payload: dict, tokens: int = 0) -> int:
        self.ensure_conversation(conversation_id)
        message_id = self._next_id
        self._next_id += 1
        payload.update(id=message_id, created_at=datetime.now(UTC))
        self._messages[conversation_id].append(payload)
        conversation = self._conversations[conversation_id]
        conversation["message_count"] += 1
        conversation["total_tokens"] += tokens
        conversation["updated_at"] = datetime.now(UTC)
        if payload["role"] == Role.USER.value and not conversation["title"]:
            conversation["title"] = payload["content"][:120]
        return message_id

    def add_user_message(self, conversation_id: str, content: str) -> int:
        return self._append(conversation_id, {"role": Role.USER.value, "content": content})

    def add_assistant_message(self, conversation_id: str, answer: RAGAnswer) -> int:
        return self._append(
            conversation_id,
            {
                "role": Role.ASSISTANT.value,
                "content": answer.answer,
                "model": answer.model,
                "latency_ms": answer.latency_ms,
                "grounded": answer.grounded,
                "reranked": answer.reranked,
                "sources": [{"url": c.url, "title": c.title} for c in answer.citations],
                "feedback": None,
                "error": None,
            },
            tokens=answer.total_tokens,
        )

    def add_error_message(self, conversation_id: str, message: str, error: str) -> int:
        return self._append(
            conversation_id,
            {"role": Role.ASSISTANT.value, "content": message, "error": error, "grounded": False},
        )

    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> list[dict]:
        ordered = sorted(
            self._conversations.values(), key=lambda c: c["updated_at"], reverse=True
        )
        return [
            {
                **c,
                "created_at": c["created_at"].isoformat(),
                "updated_at": c["updated_at"].isoformat(),
            }
            for c in ordered[offset : offset + limit]
        ]

    def get_conversation(self, conversation_id: str) -> dict:
        if conversation_id not in self._conversations:
            raise ConversationNotFoundError(f"No existe la conversación '{conversation_id}'")
        conversation = self._conversations[conversation_id]
        return {
            **conversation,
            "created_at": conversation["created_at"].isoformat(),
            "updated_at": conversation["updated_at"].isoformat(),
            "messages": [
                {**m, "created_at": m["created_at"].isoformat()}
                for m in self._messages[conversation_id]
            ],
        }

    def set_feedback(self, message_id: int, value: int) -> None:
        if value not in (-1, 1):
            raise ConversationError("El feedback debe ser 1 (útil) o -1 (no útil)")
        for messages in self._messages.values():
            for message in messages:
                if message["id"] == message_id:
                    message["feedback"] = value
                    return
        raise ConversationNotFoundError(f"No existe el mensaje {message_id}")

    def delete_conversation(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)
        self._messages.pop(conversation_id, None)
