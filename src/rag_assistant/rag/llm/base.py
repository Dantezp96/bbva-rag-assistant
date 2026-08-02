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
    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Genera una respuesta a partir de mensajes en formato chat."""

    @abstractmethod
    def health(self) -> bool:
        """`True` si el proveedor es alcanzable y está autenticado."""
