"""Proveedores de LLM: construcción, fábrica y errores accionables."""

from __future__ import annotations

import httpx
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


def test_se_adapta_al_modelo_que_exige_max_completion_tokens():
    """Regresión: los modelos GPT-5 rechazan `max_tokens`.

    Con el nombre del parámetro fijo en el código, cambiar LLM_MODEL a un
    modelo nuevo devolvía un 400 — justo lo contrario de lo que promete tener
    el LLM tras una interfaz intercambiable.
    """
    import openai

    provider = OpenAIProvider(Settings(openai_api_key="sk-test", llm_model="gpt-5.4-nano"))
    llamadas: list[dict] = []

    class RespuestaFalsa:
        model = "gpt-5.4-nano"
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
        choices = [
            type("C", (), {
                "message": type("M", (), {"content": "hola"})(),
                "finish_reason": "stop",
            })()
        ]

    def create(**kwargs):
        llamadas.append(kwargs)
        if "max_tokens" in kwargs:
            raise openai.BadRequestError(
                message="Unsupported parameter: 'max_tokens' is not supported with this "
                        "model. Use 'max_completion_tokens' instead.",
                response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                body=None,
            )
        return RespuestaFalsa()

    provider._client = type(
        "Cliente", (), {"chat": type("Chat", (), {"completions": type(
            "Comp", (), {"create": staticmethod(create)})()})()}
    )()

    r = provider.complete([{"role": "user", "content": "hola"}])
    assert r.content == "hola"
    # Primer intento con max_tokens, segundo con el nombre nuevo.
    assert "max_tokens" in llamadas[0]
    assert "max_completion_tokens" in llamadas[1]

    # El nombre correcto se recuerda: la siguiente llamada no repite el fallo.
    llamadas.clear()
    provider.complete([{"role": "user", "content": "otra"}])
    assert len(llamadas) == 1
    assert "max_completion_tokens" in llamadas[0]


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
