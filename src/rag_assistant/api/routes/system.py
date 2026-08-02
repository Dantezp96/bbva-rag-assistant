"""Endpoints de salud, configuración e ingesta."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from rag_assistant import __version__
from rag_assistant.api.dependencies import get_engine
from rag_assistant.api.schemas import HealthResponse, IngestRequest
from rag_assistant.config import get_settings
from rag_assistant.core.logging import get_logger
from rag_assistant.rag import RAGEngine

logger = get_logger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Estado del sistema")
def health(engine: RAGEngine = Depends(get_engine)) -> HealthResponse:
    """Comprueba base vectorial y LLM, e informa cuántos chunks hay indexados."""
    details = engine.health()
    store_ok = bool(details["vector_store"].get("reachable"))
    has_data = details.get("indexed_chunks", 0) > 0

    if store_ok and has_data:
        status = "ok"
    elif store_ok:
        status = "degraded"  # servicio arriba, pero sin corpus indexado
    else:
        status = "unhealthy"

    return HealthResponse(status=status, version=__version__, details=details)


@router.get("/config", summary="Configuración efectiva (sin secretos)")
def config() -> dict:
    """Devuelve la configuración en uso; útil para verificar el .env cargado."""
    return get_settings().redacted()


@router.post("/ingest", summary="Lanzar scraping + indexación en segundo plano")
def ingest(request: IngestRequest, background: BackgroundTasks) -> dict:
    """Dispara la ingesta sin bloquear la petición.

    Un crawl completo tarda varios minutos: mantener la conexión HTTP abierta
    provocaría timeouts en cualquier proxy intermedio. Para uso operativo es
    preferible la CLI (`docker compose run --rm ingest`), que muestra progreso.
    """
    from rag_assistant.pipelines import run_ingestion

    background.add_task(
        run_ingestion,
        seeds=request.seeds,
        max_pages=request.max_pages,
        recreate=request.recreate,
        skip_scrape=request.skip_scrape,
    )
    logger.info("api.ingest_scheduled", seeds=request.seeds, recreate=request.recreate)
    return {
        "status": "scheduled",
        "message": "Ingesta lanzada en segundo plano. Sigue el progreso en los logs del servicio.",
    }
