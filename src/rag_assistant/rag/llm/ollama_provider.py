"""Proveedor Ollama: modelos abiertos, ejecución local, coste cero.

La prueba valora positivamente las herramientas sin costo. Este proveedor
permite ejecutar todo el sistema —scraping, indexación y generación— sin
ninguna API de pago:

    docker compose --profile ollama up -d
    docker compose exec ollama ollama pull llama3.2:3b
    # y en .env:  LLM_PROVIDER=ollama

Se habla con la API HTTP nativa de Ollama vía httpx para no añadir otra
dependencia al proyecto.
"""

from __future__ import annotations

import httpx

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import LLMError
from rag_assistant.core.logging import get_logger
from rag_assistant.rag.llm.base import LLMProvider, LLMResponse

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Adaptador sobre la API `/api/chat` de Ollama."""

    name = "ollama"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._base_url = self._settings.ollama_base_url.rstrip("/")

    @property
    def model(self) -> str:
        return self._settings.ollama_model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self._settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": (
                    self._settings.llm_temperature if temperature is None else temperature
                ),
                "num_predict": max_tokens or self._settings.llm_max_tokens,
            },
        }
        try:
            with httpx.Client(timeout=self._settings.llm_timeout_seconds) as client:
                response = client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = str(exc)
            if exc.response.status_code == 404:
                detail = (
                    f"El modelo '{self._settings.ollama_model}' no está descargado. "
                    f"Ejecuta: docker compose exec ollama ollama pull {self._settings.ollama_model}"
                )
            raise LLMError("Ollama devolvió un error", detail=detail) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"No se pudo contactar con Ollama en {self._base_url}",
                detail=f"{exc}. ¿Levantaste el perfil: docker compose --profile ollama up -d?",
            ) from exc

        return LLMResponse(
            content=(data.get("message", {}).get("content") or "").strip(),
            model=data.get("model", self._settings.ollama_model),
            # Ollama reporta tokens con otros nombres; los normalizamos.
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            finish_reason=data.get("done_reason", "stop"),
        )

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                return client.get(f"{self._base_url}/api/tags").status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("llm.ollama_unreachable", error=str(exc))
            return False
