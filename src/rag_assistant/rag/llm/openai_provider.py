"""Proveedor OpenAI (Chat Completions).

Modelo por defecto: `gpt-4o-mini` — la relación calidad/precio adecuada para
una tarea de RAG en español, donde el modelo no tiene que "saber" nada: solo
redactar a partir del contexto recuperado.
"""

from __future__ import annotations

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
)
from rag_assistant.core.logging import get_logger
from rag_assistant.rag.llm.base import LLMProvider, LLMResponse

logger = get_logger(__name__)

#: Endpoint por defecto. Se pasa SIEMPRE de forma explícita al SDK: si se le
#: entrega `base_url=None`, el SDK lee `OPENAI_BASE_URL` del entorno por su
#: cuenta, y una variable declarada pero vacía en el `.env` (caso habitual)
#: llega como cadena vacía, no como ausente. El resultado es un
#: `httpx.UnsupportedProtocol` envuelto en un `APIConnectionError` genérico
#: —"Connection error"— que apunta a la red cuando el problema es la URL.
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(LLMProvider):
    """Adaptador sobre el SDK oficial de OpenAI."""

    name = "openai"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise ConfigurationError(
                "Falta OPENAI_API_KEY",
                detail=(
                    "Defínela en el .env, o cambia a LLM_PROVIDER=ollama "
                    "para usar un modelo local sin coste."
                ),
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url or _DEFAULT_BASE_URL,
                timeout=self._settings.llm_timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )
        return self._client

    @property
    def base_url(self) -> str:
        return self._settings.openai_base_url or _DEFAULT_BASE_URL

    @property
    def model(self) -> str:
        return self._settings.llm_model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=(
                    self._settings.llm_temperature if temperature is None else temperature
                ),
                max_tokens=max_tokens or self._settings.llm_max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthenticationError(
                "OpenAI rechazó las credenciales",
                detail="Revisa OPENAI_API_KEY (¿caducada o revocada?).",
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(
                "OpenAI aplicó rate limiting",
                detail="Reintenta en unos segundos o reduce la concurrencia.",
            ) from exc
        except openai.APIConnectionError as exc:
            # El SDK envuelve cualquier fallo de transporte en un genérico
            # "Connection error"; la causa original es lo único accionable.
            raise LLMError(
                "No se pudo conectar con OpenAI",
                detail=f"{exc.__cause__ or exc} (base_url={self.base_url})",
            ) from exc
        except openai.APIError as exc:
            raise LLMError("Error en la API de OpenAI", detail=str(exc)) from exc
        except Exception as exc:  # red, timeout
            raise LLMError("No se pudo contactar con OpenAI", detail=str(exc)) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=(choice.message.content or "").strip(),
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    def health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm.openai_unreachable", error=str(exc))
            return False
