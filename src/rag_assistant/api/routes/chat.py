"""Endpoints conversacionales."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from rag_assistant.api.dependencies import get_engine
from rag_assistant.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationSummary,
    FeedbackRequest,
    SourceOut,
)
from rag_assistant.core.exceptions import ConversationNotFoundError, RAGAssistantError
from rag_assistant.rag import RAGEngine

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="Enviar un mensaje al asistente")
def chat(request: ChatRequest, engine: RAGEngine = Depends(get_engine)) -> ChatResponse:
    """Responde una pregunta manteniendo el contexto de la conversación.

    El sistema recuerda los últimos `HISTORY_WINDOW_SIZE` mensajes de la
    conversación identificada por `conversation_id`.
    """
    answer = engine.ask(
        request.message,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        channel="api",
        history_window=request.history_window,
    )
    return ChatResponse(
        answer=answer.answer,
        conversation_id=answer.conversation_id,
        message_id=answer.message_id,
        sources=[
            SourceOut(index=c.index, url=c.url, title=c.title, score=c.score)
            for c in answer.citations
        ],
        grounded=answer.grounded,
        reranked=answer.reranked,
        history_used=answer.history_used,
        latency_ms=answer.latency_ms,
        retrieval_ms=answer.retrieval_ms,
        rerank_ms=answer.rerank_ms,
        llm_ms=answer.llm_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        model=answer.model,
    )


@router.get(
    "/conversations",
    response_model=list[ConversationSummary],
    summary="Listar conversaciones",
)
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    engine: RAGEngine = Depends(get_engine),
) -> list[ConversationSummary]:
    return [
        ConversationSummary(**row)
        for row in engine.repository.list_conversations(limit=limit, offset=offset)
    ]


@router.get("/conversations/{conversation_id}", summary="Historial de una conversación")
def get_conversation(conversation_id: str, engine: RAGEngine = Depends(get_engine)) -> dict:
    try:
        return engine.repository.get_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Eliminar una conversación",
)
def delete_conversation(conversation_id: str, engine: RAGEngine = Depends(get_engine)) -> None:
    try:
        engine.repository.delete_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT, summary="Valorar una respuesta")
def feedback(request: FeedbackRequest, engine: RAGEngine = Depends(get_engine)) -> None:
    """Registra si una respuesta fue útil. Alimenta la métrica de satisfacción."""
    try:
        engine.repository.set_feedback(request.message_id, request.value)
    except ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RAGAssistantError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
