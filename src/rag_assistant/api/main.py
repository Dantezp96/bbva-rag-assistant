"""Aplicación FastAPI.

Expone el sistema RAG por HTTP. La UI de Streamlit y la CLI consumen esta
misma API, de modo que existe una sola implementación de la lógica.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_assistant import __version__
from rag_assistant.api.dependencies import get_engine
from rag_assistant.api.routes import analytics, chat, system
from rag_assistant.config import get_settings
from rag_assistant.conversation import init_database
from rag_assistant.core.exceptions import (
    CollectionNotFoundError,
    ConfigurationError,
    ConversationError,
    ConversationNotFoundError,
    LLMAuthenticationError,
    LLMRateLimitError,
    RAGAssistantError,
    VectorStoreError,
)
from rag_assistant.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: Traducción de cada excepción de dominio al código HTTP que le corresponde.
_STATUS_BY_EXCEPTION: list[tuple[type[RAGAssistantError], int]] = [
    (ConversationNotFoundError, 404),
    (CollectionNotFoundError, 409),
    (LLMAuthenticationError, 502),
    (LLMRateLimitError, 429),
    (VectorStoreError, 503),
    # Una base de datos caída no es culpa del cliente: 503, no 400. Comprobado
    # parando el contenedor de PostgreSQL (`scripts/e2e_resiliencia.py`).
    (ConversationError, 503),
    (ConfigurationError, 500),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    settings.ensure_directories()
    logger.info(
        "api.starting",
        version=__version__,
        llm=f"{settings.llm_provider}:{settings.llm_model}",
        embeddings=settings.embedding_model,
        history_window=settings.history_window_size,
    )

    try:
        init_database()
    except Exception as exc:  # noqa: BLE001 - se arranca degradado y se ve en /health
        logger.error("api.database_init_failed", error=str(exc))

    # Precarga de modelos: evita que la primera pregunta del usuario pague
    # los ~10 s de carga de los ONNX.
    try:
        get_engine().warmup()
    except Exception as exc:  # noqa: BLE001
        logger.warning("api.warmup_failed", error=str(exc))

    logger.info("api.ready")
    yield
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Asistente RAG — BBVA Colombia",
        description=(
            "Sistema RAG sobre el contenido público de https://www.bbva.com.co/.\n\n"
            "Scraping con Playwright · Embeddings locales (fastembed/ONNX) · "
            "Qdrant · Reranking con cross-encoder · Historial conversacional persistente."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RAGAssistantError)
    async def domain_error_handler(_: Request, exc: RAGAssistantError) -> JSONResponse:
        """Convierte errores de dominio en respuestas explicativas, no en 500 opacos."""
        status_code = next(
            (code for kind, code in _STATUS_BY_EXCEPTION if isinstance(exc, kind)), 400
        )
        logger.warning("api.domain_error", type=type(exc).__name__, error=str(exc))
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.message,
                "detail": exc.detail,
                "type": type(exc).__name__,
            },
        )

    @app.get("/", tags=["system"], summary="Información del servicio")
    def root() -> dict:
        return {
            "service": "Asistente RAG — BBVA Colombia",
            "version": __version__,
            "docs": "/docs",
            "endpoints": {
                "chat": "POST /chat",
                "conversations": "GET /chat/conversations",
                "analytics": "GET /analytics",
                "health": "GET /health",
            },
        }

    app.include_router(system.router)
    app.include_router(chat.router)
    app.include_router(analytics.router)
    return app


app = create_app()
