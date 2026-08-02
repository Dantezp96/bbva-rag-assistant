"""Fábrica de proveedores de LLM (Factory Method)."""

from __future__ import annotations

from functools import lru_cache

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.exceptions import ConfigurationError
from rag_assistant.rag.llm.base import LLMProvider
from rag_assistant.rag.llm.ollama_provider import OllamaProvider
from rag_assistant.rag.llm.openai_provider import OpenAIProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def create_llm_provider(
    settings: Settings | None = None, *, provider: str | None = None
) -> LLMProvider:
    settings = settings or get_settings()
    resolved = (provider or settings.llm_provider).lower()
    try:
        cls = _REGISTRY[resolved]
    except KeyError as exc:
        raise ConfigurationError(
            f"Proveedor de LLM desconocido: '{resolved}'",
            detail=f"Opciones válidas: {', '.join(sorted(_REGISTRY))}",
        ) from exc
    return cls(settings)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    return create_llm_provider()
