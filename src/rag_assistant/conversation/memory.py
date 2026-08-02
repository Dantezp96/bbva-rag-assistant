"""Memoria conversacional: ventana deslizante de los últimos N mensajes.

Requisito de la prueba: *"Mantener el historial de conversación de acuerdo a un
ID, teniendo en cuenta los N mensajes anteriores (N configurable)"*.

`N` se controla con `HISTORY_WINDOW_SIZE` y cuenta **mensajes**, no turnos: con
N=6 el modelo ve los 3 últimos intercambios completos (pregunta + respuesta).

Se eligió ventana deslizante en lugar de resumir el historial con el LLM
porque es determinista, no cuesta tokens extra ni introduce latencia, y no
puede alucinar sobre lo que el usuario dijo antes. El resumen progresivo queda
anotado como mejora futura para sesiones muy largas.
"""

from __future__ import annotations

from rag_assistant.config import Settings, get_settings
from rag_assistant.conversation.repository import ConversationRepository
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import Message

logger = get_logger(__name__)


class ConversationMemory:
    """Recupera el contexto conversacional acotado a los últimos N mensajes."""

    def __init__(
        self, repository: ConversationRepository, settings: Settings | None = None
    ) -> None:
        self._repository = repository
        self._settings = settings or get_settings()

    @property
    def window_size(self) -> int:
        return self._settings.history_window_size

    def load(self, conversation_id: str, *, window: int | None = None) -> list[Message]:
        """Últimos N mensajes de la conversación, en orden cronológico."""
        size = self.window_size if window is None else window
        if size <= 0:
            return []
        messages = self._repository.get_recent_messages(conversation_id, size)

        # Si la ventana empieza en una respuesta huérfana, se descarta: dejar
        # una respuesta sin su pregunta confunde al modelo más que ayudarle.
        while messages and messages[0].role.value == "assistant":
            messages.pop(0)

        logger.debug(
            "memory.loaded", conversation_id=conversation_id, window=size, loaded=len(messages)
        )
        return messages
