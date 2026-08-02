"""API HTTP: contrato, manejo de errores y flujo conversacional."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_assistant.conversation import InMemoryConversationRepository
from rag_assistant.rag import RAGEngine

from tests.conftest import FakeVectorStore
from tests.test_rag import _chunks


@pytest.fixture
def client(settings, fake_embedder, fake_llm, monkeypatch):
    """Cliente con el motor RAG cableado a dobles: sin Qdrant, sin OpenAI, sin BD."""
    from rag_assistant.api import dependencies
    from rag_assistant.api.main import create_app

    engine = RAGEngine(
        settings,
        repository=InMemoryConversationRepository(),
        embedder=fake_embedder,
        store=FakeVectorStore(results=_chunks(3)),
        llm=fake_llm,
    )

    # Los routers hicieron `from ...dependencies import get_engine` en tiempo de
    # importación, así que la clave de `dependency_overrides` debe ser ESE objeto
    # original. Se captura antes de tocar nada.
    original_get_engine = dependencies.get_engine

    # El lifespan llama a get_engine()/init_database() directamente (no vía
    # Depends): se sustituyen para que el arranque no toque Postgres ni cargue
    # los modelos ONNX reales.
    monkeypatch.setattr("rag_assistant.api.main.get_engine", lambda: engine)
    monkeypatch.setattr("rag_assistant.api.main.init_database", lambda: None)

    app = create_app()
    app.dependency_overrides[original_get_engine] = lambda: engine
    with TestClient(app) as test_client:
        test_client.engine = engine
        yield test_client


def test_root_describe_el_servicio(client):
    body = client.get("/").json()
    assert body["service"].startswith("Asistente RAG")
    assert "endpoints" in body


def test_health_reporta_estado_y_dependencias(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded", "unhealthy"}
    assert "vector_store" in body["details"]
    assert "llm" in body["details"]


def test_config_no_expone_secretos(client):
    body = client.get("/config").json()
    assert body["openai_api_key"] == "***redacted***"


def test_chat_devuelve_respuesta_y_fuentes(client):
    response = client.post("/chat", json={"message": "¿Qué cuentas de ahorro hay?"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"]
    assert body["conversation_id"]
    assert body["message_id"] is not None
    assert len(body["sources"]) == 3
    assert body["sources"][0]["url"].startswith("https://")


def test_chat_mantiene_el_hilo_por_conversation_id(client):
    primera = client.post(
        "/chat", json={"message": "¿Qué es un CDT?", "conversation_id": "hilo-1"}
    ).json()
    segunda = client.post(
        "/chat", json={"message": "¿y el plazo?", "conversation_id": "hilo-1"}
    ).json()

    assert primera["history_used"] == 0
    assert segunda["history_used"] == 2
    assert segunda["conversation_id"] == "hilo-1"


def test_mensaje_vacio_es_rechazado_por_el_contrato(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_listar_y_consultar_conversaciones(client):
    client.post("/chat", json={"message": "hola", "conversation_id": "abc"})

    listado = client.get("/chat/conversations").json()
    assert any(c["id"] == "abc" for c in listado)

    detalle = client.get("/chat/conversations/abc").json()
    assert len(detalle["messages"]) == 2


def test_conversacion_inexistente_devuelve_404(client):
    assert client.get("/chat/conversations/no-existe").status_code == 404


def test_feedback_se_registra(client):
    message_id = client.post("/chat", json={"message": "hola"}).json()["message_id"]
    assert client.post("/chat/feedback", json={"message_id": message_id, "value": 1}).status_code == 204


def test_feedback_invalido_devuelve_400(client):
    message_id = client.post("/chat", json={"message": "hola"}).json()["message_id"]
    assert client.post("/chat/feedback", json={"message_id": message_id, "value": 7}).status_code == 400


def test_analitica_en_vivo_cuenta_las_consultas(client):
    client.post("/chat", json={"message": "una pregunta"})
    snapshot = client.get("/analytics/live").json()
    assert snapshot["queries"] >= 1
    assert "avg_latency_ms" in snapshot
