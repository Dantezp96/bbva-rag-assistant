"""Logging estructurado con `structlog`.

Se configura una sola vez por proceso. En desarrollo imprime en consola con
colores; en producción emite JSON, listo para ser ingerido por un agregador
(Loki, CloudWatch, Datadog…) sin post-procesado.
"""

from __future__ import annotations

import logging
import sys

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configura structlog + logging estándar. Idempotente."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    # Librerías ruidosas: solo advertencias hacia arriba.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "openai", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Devuelve un logger vinculado al módulo indicado."""
    return structlog.get_logger(name)
