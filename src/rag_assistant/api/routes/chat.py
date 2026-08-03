"""Endpoints conversacionales."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from rag_assistant.analytics import AnalyticsService
from rag_assistant.api.dependencies import get_analytics, get_engine
from rag_assistant.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationSummary,
    FeedbackRequest,
    SourceOut,
)
from rag_assistant.config import get_settings
from rag_assistant.core.exceptions import ConversationNotFoundError, RAGAssistantError
from rag_assistant.core.logging import get_logger
from rag_assistant.rag import RAGEngine

logger = get_logger(__name__)
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
            SourceOut(
                index=c.index,
                url=c.url,
                title=c.title,
                score=c.score,
                rerank_score=c.rerank_score,
            )
            for c in answer.citations
        ],
        grounded=answer.grounded,
        reranked=answer.reranked,
        history_used=answer.history_used,
        search_query=answer.search_query,
        rewritten=answer.rewritten,
        suggestions=answer.suggestions,
        latency_ms=answer.latency_ms,
        rewrite_ms=answer.rewrite_ms,
        suggestions_ms=answer.suggestions_ms,
        retrieval_ms=answer.retrieval_ms,
        rerank_ms=answer.rerank_ms,
        llm_ms=answer.llm_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        model=answer.model,
    )


@router.post("/stream", summary="Enviar un mensaje y recibir la respuesta por partes")
def chat_stream(request: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    """Igual que `POST /chat`, pero emitiendo eventos SSE conforme se producen.

    Tipos de evento, en orden:
      `start`    id de la conversación (útil cuando el cliente no lo fijó)
      `sources`  las fuentes, en cuanto se recuperan y antes de generar
      `token`    fragmentos de texto según llegan del modelo
      `error`    fallo controlado; el turno queda igualmente registrado
      `done`     telemetría completa, id del mensaje y sugerencias

    Las fuentes se emiten antes que el texto a propósito: el usuario ve de
    dónde saldrá la respuesta mientras el modelo todavía la escribe.
    """

    def eventos():
        try:
            for tipo, datos in engine.ask_stream(
                request.message,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                channel="web",
                history_window=request.history_window,
            ):
                yield f"event: {tipo}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 - el flujo ya está abierto
            logger.exception("api.stream_failed", error=str(exc))
            fallo = {"message": "Se interrumpió la generación de la respuesta."}
            yield f"event: error\ndata: {json.dumps(fallo, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        eventos(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Evita que un proxy intermedio acumule la respuesta y anule el
            # streaming: sin esto, el usuario vuelve a esperar en silencio.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/starters", summary="Preguntas sugeridas para empezar")
def starters(service: AnalyticsService = Depends(get_analytics)) -> dict:
    """Preguntas con las que arrancar una conversación nueva.

    Si el histórico ya tiene señal, se ofrecen las preguntas reales más
    repetidas que el sistema supo responder; si no, la lista configurada en
    `STARTER_QUESTIONS`. En ambos casos son preguntas que el corpus cubre.
    """
    settings = get_settings()
    populares = service.popular_questions(limit=settings.suggestions_count + 1)
    if populares:
        return {"questions": populares, "source": "historico"}
    return {"questions": settings.starter_questions, "source": "configuracion"}


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
