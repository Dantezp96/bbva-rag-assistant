"""Proveedores de LLM: construcción, fábrica y errores accionables."""

from __future__ import annotations

import pytest

from rag_assistant.config import Settings
from rag_assistant.core.exceptions import ConfigurationError
from rag_assistant.rag.llm import (
    OllamaProvider,
    OpenAIProvider,
    create_llm_provider,
)
from rag_assistant.rag.llm.openai_provider import _DEFAULT_BASE_URL


def test_sin_api_key_el_error_sugiere_la_alternativa_gratuita():
    with pytest.raises(ConfigurationError, match="ollama"):
        OpenAIProvider(Settings(llm_provider="openai", openai_api_key=None))


def test_base_url_vacia_no_se_propaga_al_sdk():
    """Regresión: `OPENAI_BASE_URL=` (declarada y vacía) rompía las llamadas.

    Si al SDK se le pasa `base_url=None`, él mismo lee `OPENAI_BASE_URL` del
    entorno; una variable declarada pero vacía llega como cadena vacía —no como
    ausente— y httpx la rechaza por no tener esquema. El fallo se manifestaba
    como un `APIConnectionError` genérico ("Connection error") que apuntaba a
    la red, no a la configuración. Por eso el endpoint se pasa SIEMPRE explícito.
    """
    provider = OpenAIProvider(Settings(openai_api_key="sk-test", openai_base_url=""))
    assert provider.base_url == _DEFAULT_BASE_URL
    assert provider.client.base_url.scheme in ("http", "https")


def test_base_url_personalizada_se_respeta():
    provider = OpenAIProvider(
        Settings(openai_api_key="sk-test", openai_base_url="https://gateway.example.com/v1")
    )
    assert provider.base_url == "https://gateway.example.com/v1"


def test_la_fabrica_devuelve_el_proveedor_configurado():
    openai_provider = create_llm_provider(
        Settings(llm_provider="openai", openai_api_key="sk-test")
    )
    assert isinstance(openai_provider, OpenAIProvider)
    assert isinstance(create_llm_provider(Settings(llm_provider="ollama")), OllamaProvider)


def test_la_fabrica_rechaza_un_proveedor_desconocido():
    with pytest.raises(ConfigurationError, match="Opciones válidas"):
        create_llm_provider(Settings(), provider="gemini")


def test_ollama_inalcanzable_no_lanza_en_health():
    provider = OllamaProvider(Settings(ollama_base_url="http://127.0.0.1:1"))
    assert provider.health() is False
