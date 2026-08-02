"""Motor y sesiones de base de datos.

Soporta PostgreSQL (por defecto en Docker) y SQLite (ejecución local sin
infraestructura). El esquema se crea con `create_all`: para un proyecto de
este alcance, añadir Alembic aportaría ceremonia sin valor —queda anotado
como mejora futura en el README.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag_assistant.config import get_settings
from rag_assistant.conversation.entities import Base
from rag_assistant.core.exceptions import ConversationError
from rag_assistant.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Motor único por proceso (mantiene un pool de conexiones reutilizable)."""
    settings = get_settings()
    url = settings.database_url

    kwargs: dict[str, object] = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        # SQLite: el fichero debe existir y se comparte entre hilos de FastAPI.
        db_path = url.split("///")[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_size=5, max_overflow=10, pool_recycle=1800)

    engine = create_engine(url, **kwargs)
    logger.info("db.engine_created", dialect=engine.dialect.name)
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def init_database() -> None:
    """Crea las tablas si no existen. Idempotente."""
    try:
        Base.metadata.create_all(bind=get_engine())
        logger.info("db.schema_ready")
    except Exception as exc:
        raise ConversationError(
            "No se pudo inicializar la base de datos", detail=str(exc)
        ) from exc


@contextmanager
def session_scope() -> Iterator[Session]:
    """Sesión transaccional: commit al salir bien, rollback si algo falla."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
