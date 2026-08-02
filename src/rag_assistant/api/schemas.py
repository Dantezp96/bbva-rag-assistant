"""Esquemas de entrada/salida de la API (contrato HTTP)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Pregunta del usuario")
    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="ID de la conversación. Si se omite, el sistema crea una nueva.",
    )
    user_id: str | None = Field(default=None, max_length=120)
    history_window: int | None = Field(
        default=None,
        ge=0,
        le=50,
        description="Sobrescribe HISTORY_WINDOW_SIZE solo para esta petición.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "¿Qué tipos de cuenta de ahorro ofrece BBVA?",
                "conversation_id": "demo-001",
            }
        }
    }


class SourceOut(BaseModel):
    index: int
    url: str
    title: str
    score: float = Field(..., description="Similitud coseno de la búsqueda vectorial, en [0, 1]")
    rerank_score: float | None = Field(
        default=None,
        description="Logit del cross-encoder si se aplicó reranking. Solo ordena; no acotado.",
    )


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    message_id: int | None = None
    sources: list[SourceOut] = []
    grounded: bool = True
    reranked: bool = False
    history_used: int = 0
    search_query: str = Field(
        default="",
        description="Consulta enviada al índice. Difiere del mensaje si se reconstruyó "
        "con el historial (seguimientos sin sujeto).",
    )
    rewritten: bool = False
    latency_ms: int = 0
    rewrite_ms: int = 0
    retrieval_ms: int = 0
    rerank_ms: int = 0
    llm_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class FeedbackRequest(BaseModel):
    message_id: int
    value: int = Field(..., description="1 = útil, -1 = no útil")


class ConversationSummary(BaseModel):
    id: str
    title: str = ""
    user_id: str | None = None
    channel: str = "api"
    created_at: str
    updated_at: str
    message_count: int = 0
    total_tokens: int = 0


class IngestRequest(BaseModel):
    seeds: list[str] | None = Field(default=None, description="URLs semilla del crawl")
    max_pages: int | None = Field(default=None, gt=0, le=1000)
    recreate: bool = Field(default=False, description="Recrea la colección desde cero")
    skip_scrape: bool = Field(
        default=False, description="Reindexa el corpus ya guardado sin volver a rastrear"
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    type: str = "RAGAssistantError"
