"""Eventos del motor RAG y sus suscriptores.

PATRÓN DE DISEÑO: **Observer**.
Cada consulta atendida interesa a varias partes: hay que registrarla en el log,
acumular contadores en memoria para el endpoint de métricas y, en el futuro,
quizá exportarla a Prometheus o a un data warehouse. Meter todo eso dentro del
motor RAG lo convertiría en un cajón de sastre acoplado a cada destino.

Con Observer, el motor solo *publica* `QueryEvent` y los suscriptores reaccionan
de forma independiente. Añadir un exportador nuevo es registrar un observador,
sin tocar el motor.

Regla clave: **un observador que falla nunca rompe la respuesta al usuario**.
Los errores de los suscriptores se capturan y se registran.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rag_assistant.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class QueryEvent:
    """Instantánea de una consulta atendida por el motor RAG."""

    conversation_id: str
    question: str
    answer_preview: str
    latency_ms: int
    retrieval_ms: int
    rerank_ms: int
    llm_ms: int
    retrieved_count: int
    top_score: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    grounded: bool
    reranked: bool
    history_used: int
    success: bool = True
    error: str | None = None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class QueryObserver(ABC):
    """Suscriptor de eventos de consulta."""

    name: str = "observer"

    @abstractmethod
    def on_query(self, event: QueryEvent) -> None: ...


class LoggingObserver(QueryObserver):
    """Emite una línea estructurada por consulta (trazabilidad operativa)."""

    name = "logging"

    def on_query(self, event: QueryEvent) -> None:
        logger.info(
            "rag.query",
            conversation_id=event.conversation_id,
            question=event.question[:120],
            latency_ms=event.latency_ms,
            retrieval_ms=event.retrieval_ms,
            rerank_ms=event.rerank_ms,
            llm_ms=event.llm_ms,
            retrieved=event.retrieved_count,
            top_score=round(event.top_score, 4),
            tokens=event.total_tokens,
            grounded=event.grounded,
            reranked=event.reranked,
            success=event.success,
            error=event.error,
        )


class InMemoryMetricsObserver(QueryObserver):
    """Contadores agregados en memoria del proceso.

    Complementa —no sustituye— la analítica sobre PostgreSQL: da un pulso
    inmediato del proceso vivo (útil en `/health` y en desarrollo) sin tocar
    la base de datos.
    """

    name = "metrics"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.queries = 0
        self.failures = 0
        self.ungrounded = 0
        self.total_latency_ms = 0
        self.total_tokens = 0
        self.latencies: list[int] = []

    def on_query(self, event: QueryEvent) -> None:
        with self._lock:
            self.queries += 1
            self.total_latency_ms += event.latency_ms
            self.total_tokens += event.total_tokens
            if not event.success:
                self.failures += 1
            if not event.grounded:
                self.ungrounded += 1
            self.latencies.append(event.latency_ms)
            # Ventana acotada: esto es un proceso de larga vida, no un log.
            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            if not self.queries:
                return {"queries": 0}
            ordered = sorted(self.latencies)
            p95_index = max(0, int(len(ordered) * 0.95) - 1)
            return {
                "queries": self.queries,
                "failures": self.failures,
                "ungrounded": self.ungrounded,
                "avg_latency_ms": round(self.total_latency_ms / self.queries, 1),
                "p95_latency_ms": ordered[p95_index],
                "total_tokens": self.total_tokens,
            }


class QueryEventPublisher:
    """Sujeto observable: registra suscriptores y les notifica cada evento."""

    def __init__(self) -> None:
        self._observers: list[QueryObserver] = []

    def subscribe(self, observer: QueryObserver) -> None:
        self._observers.append(observer)

    def unsubscribe(self, observer: QueryObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def publish(self, event: QueryEvent) -> None:
        for observer in self._observers:
            try:
                observer.on_query(event)
            except Exception as exc:  # noqa: BLE001 - la telemetría no rompe el servicio
                logger.warning("observer.failed", observer=observer.name, error=str(exc))

    def get(self, name: str) -> QueryObserver | None:
        return next((o for o in self._observers if o.name == name), None)
