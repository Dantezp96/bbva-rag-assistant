"""Modelos de dominio compartidos entre capas.

Son objetos de transporte puros: no conocen HTTP, ni SQL, ni Qdrant. Esto
mantiene el núcleo del sistema independiente de la infraestructura y hace que
cambiar de base vectorial o de framework web no toque la lógica de negocio.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Autor de un mensaje dentro de una conversación."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(slots=True)
class RawPage:
    """HTML tal cual se descargó del sitio, sin procesar."""

    url: str
    html: str
    status_code: int
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    depth: int = 0
    fetcher: str = "unknown"
    elapsed_ms: int = 0

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.html.encode("utf-8", errors="ignore")).hexdigest()

    @property
    def slug(self) -> str:
        """Nombre de archivo estable y seguro derivado de la URL."""
        digest = hashlib.sha1(self.url.encode("utf-8")).hexdigest()[:10]
        tail = self.url.rstrip("/").rsplit("/", 1)[-1] or "index"
        tail = "".join(c if c.isalnum() or c in "-_" else "-" for c in tail)[:60]
        return f"{tail}-{digest}"


@dataclass(slots=True)
class CleanDocument:
    """Contenido textual limpio y normalizado de una página."""

    url: str
    title: str
    text: str
    description: str = ""
    section: str = ""
    headings: list[str] = field(default_factory=list)
    language: str = "es"
    scraped_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    word_count: int = 0
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "section": self.section,
            "headings": self.headings,
            "language": self.language,
            "scraped_at": self.scraped_at.isoformat(),
            "word_count": self.word_count,
            "content_hash": self.content_hash,
            "text": self.text,
        }


@dataclass(slots=True)
class Chunk:
    """Fragmento indexable de un documento."""

    id: str
    text: str
    url: str
    title: str
    section: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """Payload que se persiste junto al vector en Qdrant."""
        return {
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            **self.metadata,
        }


@dataclass(slots=True)
class RetrievedChunk:
    """Chunk recuperado del índice, con su puntuación."""

    id: str
    text: str
    url: str
    title: str
    score: float
    section: str = ""
    chunk_index: int = 0
    rerank_score: float | None = None

    @property
    def effective_score(self) -> float:
        """Puntuación vigente: la del reranker si existe, si no la vectorial."""
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass(slots=True)
class Citation:
    """Fuente citada en una respuesta.

    Se guardan las dos puntuaciones por separado a propósito: la del reranker
    es un logit sin acotar (valores típicos de -10 a +10) mientras que la
    vectorial es una similitud coseno en [0, 1]. Mezclarlas en un solo campo
    hace que la métrica agregada `avg_top_score` no signifique nada, porque
    promediaría magnitudes de escalas distintas según si el reranker actuó.
    """

    index: int
    url: str
    title: str
    #: Similitud coseno de la búsqueda vectorial, en [0, 1]. Comparable entre
    #: consultas y a lo largo del tiempo: es la que alimenta la analítica.
    score: float
    #: Logit del cross-encoder, si el reranking se aplicó. Solo ordena.
    rerank_score: float | None = None


@dataclass(slots=True)
class Message:
    """Mensaje individual de una conversación."""

    role: Role
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: int | None = None


@dataclass(slots=True)
class RAGAnswer:
    """Resultado completo de una consulta al sistema RAG."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    conversation_id: str = ""
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: int = 0
    retrieval_ms: int = 0
    rerank_ms: int = 0
    llm_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    grounded: bool = True
    reranked: bool = False
    history_used: int = 0
    #: Consulta realmente enviada al índice. Difiere de la pregunta cuando el
    #: seguimiento se reconstruyó con el historial; exponerla hace depurable
    #: una recuperación que de otro modo parece magia.
    search_query: str = ""
    rewritten: bool = False
    rewrite_ms: int = 0
    #: Preguntas de seguimiento propuestas, derivadas del contexto recuperado.
    suggestions: list[str] = field(default_factory=list)
    suggestions_ms: int = 0
    #: ID del mensaje persistido; lo usa la UI para enviar feedback sobre él.
    message_id: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
