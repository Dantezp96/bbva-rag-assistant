"""Endpoints de analítica del histórico de conversaciones."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from rag_assistant.analytics import AnalyticsService
from rag_assistant.api.dependencies import get_analytics, get_engine
from rag_assistant.core.exceptions import ConversationNotFoundError
from rag_assistant.rag import RAGEngine

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("", summary="Informe de métricas e impacto")
def report(
    days: int | None = Query(
        default=None, ge=1, le=365, description="Acota el periodo. Omitir = todo el histórico."
    ),
    limit_examples: int = Query(default=10, ge=1, le=50),
    service: AnalyticsService = Depends(get_analytics),
) -> dict:
    """Recorre el histórico completo y devuelve las métricas de impacto."""
    return service.report(days=days, limit_examples=limit_examples).to_dict()


@router.get("/live", summary="Contadores del proceso en vivo")
def live_metrics(engine: RAGEngine = Depends(get_engine)) -> dict:
    """Métricas acumuladas en memoria desde que arrancó el proceso."""
    observer = engine.publisher.get("metrics")
    return observer.snapshot() if observer else {"queries": 0}


@router.get("/conversations/{conversation_id}", summary="Traza detallada de una conversación")
def conversation_detail(
    conversation_id: str, service: AnalyticsService = Depends(get_analytics)
) -> dict:
    try:
        return service.conversation_detail(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
