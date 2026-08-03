"""Streaming SSE: orden de eventos, telemetría preservada y degradación."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from tests.conftest import FakeVectorStore
from tests.test_rag import _chunks

from rag_assistant.conversation import InMemoryConversationRepository
from rag_assistant.rag import RAGEngine


@pytest.fixture
def engine_stream(settings, fake_embedder, fake_llm):
    """Motor con un LLM que emite por partes y reporta consumo al final."""

    class LLMConStream(type(fake_llm)):
        def __init__(self):
            super().__init__()
            self.last_usage = (0, 0)
            self.last_model = "fake-model"

        @property
        def supports_streaming(self):
            return True

        def stream(self, messages, *, max_tokens=None, temperature=None):
            self.calls.append(messages)
            yield from ["Las cuentas ", "de ahorro ", "son varias [1]."]
            self.last_usage = (300, 25)

    object.__setattr__(settings, "suggestions_enabled", False)
    object.__setattr__(settings, "query_rewrite_enabled", False)
    return RAGEngine(
        settings,
        repository=InMemoryConversationRepository(),
        embedder=fake_embedder,
        store=FakeVectorStore(results=_chunks(3)),
        llm=LLMConStream(),
    )


def test_el_orden_de_eventos_pone_las_fuentes_antes_del_texto(engine_stream):
    """La interfaz debe poder pintar de dónde saldrá la respuesta cuanto antes.

    Se comprueba el orden RELATIVO, no la posición: entre medias hay eventos
    `stage` cuyo número depende de qué etapas opcionales estén activas.
    """
    tipos = [t for t, _ in engine_stream.ask_stream("¿Qué cuentas hay?")]
    assert tipos[0] == "start"
    assert tipos[-1] == "done"
    assert tipos.index("sources") < tipos.index("token")
    # Se anuncia cada etapa antes de ejecutarla, para no dejar la espera muda.
    assert tipos.index("stage") < tipos.index("sources")
    assert "retrieval" in [d["stage"] for t, d in engine_stream.ask_stream("otra") if t == "stage"]


def test_el_texto_llega_por_partes_y_se_recompone(engine_stream):
    eventos = list(engine_stream.ask_stream("¿Qué cuentas hay?"))
    partes = [d["text"] for t, d in eventos if t == "token"]
    assert len(partes) == 3          # se emitió por fragmentos, no de una vez
    assert "".join(partes) == "Las cuentas de ahorro son varias [1]."


def test_el_streaming_no_pierde_la_telemetria(engine_stream):
    """Servir por partes no puede costar las métricas: alimentan la analítica."""
    done = next(d for t, d in engine_stream.ask_stream("¿Qué cuentas hay?") if t == "done")
    assert done["prompt_tokens"] == 300
    assert done["completion_tokens"] == 25
    # Con dobles todo tarda menos de 1 ms, así que solo se comprueba que los
    # campos existan y sean coherentes; la latencia real se mide contra el
    # sistema en marcha, no aquí.
    assert done["retrieval_ms"] >= 0
    assert done["llm_ms"] >= 0
    assert done["latency_ms"] >= 0
    assert done["message_id"] is not None      # el turno quedó persistido
    assert done["grounded"] is True            # citó [1]


def test_el_turno_servido_por_partes_queda_en_el_historial(engine_stream):
    list(engine_stream.ask_stream("¿Qué cuentas hay?", conversation_id="s1"))
    detalle = engine_stream.repository.get_conversation("s1")
    assert len(detalle["messages"]) == 2
    assert detalle["messages"][1]["content"] == "Las cuentas de ahorro son varias [1]."


def test_sin_contexto_no_se_abre_el_flujo_del_llm(settings, fake_embedder, fake_llm):
    """El cortocircuito debe seguir funcionando también en streaming."""
    engine = RAGEngine(
        settings,
        repository=InMemoryConversationRepository(),
        embedder=fake_embedder,
        store=FakeVectorStore(results=[]),
        llm=fake_llm,
    )
    eventos = list(engine.ask_stream("algo fuera del corpus"))
    tipos = [t for t, _ in eventos]
    assert "sources" not in tipos
    assert fake_llm.calls == []
    done = next(d for t, d in eventos if t == "done")
    assert done["grounded"] is False


def test_un_proveedor_sin_streaming_sigue_sirviendo(settings, fake_embedder, fake_llm):
    """`stream` por defecto entrega la respuesta entera: nadie aguas arriba se entera."""
    engine = RAGEngine(
        settings,
        repository=InMemoryConversationRepository(),
        embedder=fake_embedder,
        store=FakeVectorStore(results=_chunks(2)),
        llm=fake_llm,
    )
    partes = [d["text"] for t, d in engine.ask_stream("¿Qué cuentas hay?") if t == "token"]
    assert partes == ["Respuesta de prueba basada en [1]."]


# ------------------------------------------------------------------ HTTP ---
@pytest.fixture
def client_stream(engine_stream, monkeypatch):
    from rag_assistant.api import dependencies
    from rag_assistant.api.main import create_app

    original = dependencies.get_engine
    monkeypatch.setattr("rag_assistant.api.main.get_engine", lambda: engine_stream)
    monkeypatch.setattr("rag_assistant.api.main.init_database", lambda: None)
    app = create_app()
    app.dependency_overrides[original] = lambda: engine_stream
    with TestClient(app) as c:
        yield c


def test_el_endpoint_devuelve_sse_bien_formado(client_stream):
    r = client_stream.post("/chat/stream", json={"message": "¿Qué cuentas hay?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    # Sin buffering intermedio no habría streaming real tras un proxy.
    assert r.headers.get("x-accel-buffering") == "no"

    eventos = [b for b in r.text.split("\n\n") if b.strip()]
    tipos = [b.split("\n")[0].removeprefix("event: ") for b in eventos]
    assert tipos[0] == "start" and tipos[-1] == "done"
    assert tipos.index("sources") < tipos.index("token")

    done = json.loads(eventos[-1].split("data: ", 1)[1])
    assert done["model"] == "fake-model"
    assert done["message_id"] is not None
