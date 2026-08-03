"""Contrato del modelo de lenguaje.

PATRÓN DE DISEÑO: **Strategy**.
El LLM es la pieza más volátil y más cara del sistema. Detrás de esta interfaz
conviven hoy OpenAI (API de pago) y Ollama (modelos abiertos, coste cero), y
se cambia entre ellos con una variable de entorno. Eso permite desarrollar y
demostrar el sistema sin gastar créditos, y desplegar con el proveedor que
convenga sin tocar el motor RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(slots=True)
class LLMResponse:
    """Respuesta normalizada, independiente del proveedor."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """Estrategia de generación de texto."""

    name: str = "base"

    @property
    @abstractmethod
    def model(self) -> str:
        """Identificador del modelo en uso."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Genera una respuesta a partir de mensajes en formato chat.

        `max_tokens` y `temperature` permiten que una llamada auxiliar y barata
        (p. ej. reescribir una consulta) no herede el presupuesto pensado para
        redactar la respuesta final.
        """

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Genera la respuesta por fragmentos, según llegan del proveedor.

        Por defecto no hay streaming real: se produce la respuesta completa y se
        entrega de una vez. Así, un proveedor que no lo soporte sigue siendo
        utilizable por la interfaz de streaming sin casos especiales aguas
        arriba —simplemente el usuario ve el texto de golpe—.
        """
        yield self.complete(messages, max_tokens=max_tokens, temperature=temperature).content

    @property
    def supports_streaming(self) -> bool:
        """`True` si `stream` emite fragmentos reales y no la respuesta entera."""
        return False

    @abstractmethod
    def health(self) -> bool:
        """`True` si el proveedor es alcanzable y está autenticado."""
