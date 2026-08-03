"""Proveedor OpenAI (Chat Completions).

Modelo por defecto: `gpt-4o-mini` — la relación calidad/precio adecuada para
una tarea de RAG en español, donde el modelo no tiene que "saber" nada: solo
redactar a partir del contexto recuperado.
"""

from __future__ import annotations

from collections.abc import Iterator

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    RAGAssistantError,
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
        #: Nombre del parámetro de tope de tokens que acepta el modelo. Se
        #: descubre en la primera llamada y se recuerda. Ver `_crear`.
        self._token_param: str | None = None

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

    @property
    def supports_streaming(self) -> bool:
        return True

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Emite el texto según llega, y guarda el consumo del último fragmento.

        La API devuelve el uso de tokens en un chunk final con `choices` vacío,
        y solo si se pide `include_usage`. Se retiene en `last_usage` para que
        la capa superior pueda persistir la telemetría igual que en el modo no
        estimado — de lo contrario, streaming significaría perder las métricas.
        """
        self.last_usage = (0, 0)
        self.last_model = self._settings.llm_model
        try:
            flujo = self._crear(
                messages,
                temperatura=(
                    self._settings.llm_temperature if temperature is None else temperature
                ),
                tope=max_tokens or self._settings.llm_max_tokens,
                stream=True,
            )
            for chunk in flujo:
                if getattr(chunk, "model", None):
                    self.last_model = chunk.model
                if chunk.usage:
                    self.last_usage = (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                if chunk.choices and (texto := chunk.choices[0].delta.content):
                    yield texto
        except Exception as exc:
            raise self._traducir_error(exc) from exc

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        try:
            response = self._crear(
                messages,
                temperatura=(
                    self._settings.llm_temperature if temperature is None else temperature
                ),
                tope=max_tokens or self._settings.llm_max_tokens,
            )
        except Exception as exc:
            raise self._traducir_error(exc) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=(choice.message.content or "").strip(),
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
        )

    def _traducir_error(self, exc: Exception) -> Exception:
        """Convierte los errores del SDK en excepciones de dominio accionables."""
        import openai

        if isinstance(exc, RAGAssistantError):
            return exc
        if isinstance(exc, openai.AuthenticationError):
            return LLMAuthenticationError(
                "OpenAI rechazó las credenciales",
                detail="Revisa OPENAI_API_KEY (¿caducada o revocada?).",
            )
        if isinstance(exc, openai.RateLimitError):
            return LLMRateLimitError(
                "OpenAI aplicó rate limiting",
                detail="Reintenta en unos segundos o reduce la concurrencia.",
            )
        if isinstance(exc, openai.APIConnectionError):
            # El SDK envuelve cualquier fallo de transporte en un genérico
            # "Connection error"; la causa original es lo único accionable.
            return LLMError(
                "No se pudo conectar con OpenAI",
                detail=f"{exc.__cause__ or exc} (base_url={self.base_url})",
            )
        if isinstance(exc, openai.APIError):
            return LLMError("Error en la API de OpenAI", detail=str(exc))
        return LLMError("No se pudo contactar con OpenAI", detail=str(exc))

    def _crear(
        self, messages: list[dict[str, str]], *, temperatura: float, tope: int,
        stream: bool = False,
    ):
        """Invoca la API resolviendo el nombre del parámetro de tope de tokens.

        Los modelos GPT-5 rechazan `max_tokens` y exigen `max_completion_tokens`:

            'max_tokens' is not supported with this model.
            Use 'max_completion_tokens' instead.

        Con el nombre fijo en el código, cambiar `LLM_MODEL` a un modelo nuevo
        rompía el sistema con un 400 — justo lo contrario de lo que promete
        tener el LLM tras una interfaz intercambiable.

        Se resuelve probando y recordando, no mirando el nombre del modelo: una
        lista de prefijos («gpt-5», «o1»…) haría falso el mismo día que salga
        una familia nueva. El intento fallido se rechaza en validación, así que
        no consume tokens, y solo ocurre una vez por proceso.
        """
        import openai

        candidatos = [self._token_param] if self._token_param else [
            "max_tokens", "max_completion_tokens"
        ]
        extra = (
            {"stream": True, "stream_options": {"include_usage": True}} if stream else {}
        )
        for i, parametro in enumerate(candidatos):
            try:
                respuesta = self.client.chat.completions.create(
                    model=self._settings.llm_model,
                    messages=messages,
                    temperature=temperatura,
                    **{parametro: tope},
                    **extra,
                )
            except openai.BadRequestError as exc:
                ultimo = i + 1 >= len(candidatos)
                if ultimo or "max_completion_tokens" not in str(exc):
                    raise
                logger.info(
                    "llm.token_param_ajustado",
                    modelo=self._settings.llm_model,
                    de=parametro,
                    a=candidatos[i + 1],
                )
                continue
            self._token_param = parametro
            return respuesta
        raise AssertionError("inalcanzable")  # pragma: no cover

    def health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm.openai_unreachable", error=str(exc))
            return False
