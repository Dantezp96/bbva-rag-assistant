"""Persistencia del historial y métricas de impacto (sobre SQLite temporal)."""

from __future__ import annotations

import pytest

from rag_assistant.analytics import AnalyticsService
from rag_assistant.conversation import (
    ConversationMemory,
    InMemoryConversationRepository,
    SqlConversationRepository,
    init_database,
)
from rag_assistant.core.exceptions import ConversationError, ConversationNotFoundError
from rag_assistant.core.models import Citation, RAGAnswer, RetrievedChunk


@pytest.fixture
def sql_repo(settings, monkeypatch):
    """Repositorio SQL apuntando a la SQLite temporal del test."""
    from rag_assistant.config import get_settings
    from rag_assistant.conversation import db

    monkeypatch.setattr("rag_assistant.config.settings.get_settings", lambda: settings)
    monkeypatch.setattr("rag_assistant.conversation.db.get_settings", lambda: settings)
    db.get_engine.cache_clear()
    db.get_session_factory.cache_clear()
    init_database()
    yield SqlConversationRepository()
    db.get_engine.cache_clear()
    db.get_session_factory.cache_clear()
    get_settings.cache_clear()


def _answer(conversation_id: str, texto: str = "Respuesta [1]", **kwargs) -> RAGAnswer:
    defaults = {
        "citations": [
            Citation(index=1, url="https://www.bbva.com.co/ahorro.html", title="Ahorro", score=0.87)
        ],
        "retrieved": [
            RetrievedChunk(
                id="c1", text="t", url="https://www.bbva.com.co/ahorro.html",
                title="Ahorro", score=0.87,
            )
        ],
        "latency_ms": 1200, "retrieval_ms": 90, "rerank_ms": 210, "llm_ms": 880,
        "prompt_tokens": 800, "completion_tokens": 120, "model": "gpt-4o-mini",
    }
    defaults.update(kwargs)
    return RAGAnswer(answer=texto, conversation_id=conversation_id, **defaults)


# ------------------------------------------------------------ repositorio ---
def test_el_id_se_genera_si_no_se_indica(sql_repo):
    assert sql_repo.ensure_conversation().startswith("conv-")


def test_el_mismo_id_reutiliza_la_conversacion(sql_repo):
    primero = sql_repo.ensure_conversation("mi-sesion")
    segundo = sql_repo.ensure_conversation("mi-sesion")
    assert primero == segundo == "mi-sesion"
    assert len(sql_repo.list_conversations()) == 1


def test_los_mensajes_se_persisten_en_orden_cronologico(sql_repo):
    sql_repo.ensure_conversation("c1")
    sql_repo.add_user_message("c1", "primera")
    sql_repo.add_assistant_message("c1", _answer("c1"))
    sql_repo.add_user_message("c1", "segunda")

    contenidos = [m.content for m in sql_repo.get_recent_messages("c1", limit=10)]
    assert contenidos == ["primera", "Respuesta [1]", "segunda"]


def test_get_recent_messages_devuelve_los_ultimos_n(sql_repo):
    sql_repo.ensure_conversation("c1")
    for i in range(10):
        sql_repo.add_user_message("c1", f"mensaje {i}")
    recientes = sql_repo.get_recent_messages("c1", limit=3)
    assert [m.content for m in recientes] == ["mensaje 7", "mensaje 8", "mensaje 9"]


def test_el_titulo_es_la_primera_pregunta(sql_repo):
    sql_repo.ensure_conversation("c1")
    sql_repo.add_user_message("c1", "¿Cuáles son las tasas del CDT?")
    sql_repo.add_user_message("c1", "otra cosa")
    assert sql_repo.get_conversation("c1")["title"] == "¿Cuáles son las tasas del CDT?"


def test_conversacion_inexistente_lanza_error_tipado(sql_repo):
    with pytest.raises(ConversationNotFoundError):
        sql_repo.get_conversation("no-existe")


def test_feedback_solo_admite_1_o_menos_1(sql_repo):
    sql_repo.ensure_conversation("c1")
    message_id = sql_repo.add_assistant_message("c1", _answer("c1"))
    sql_repo.set_feedback(message_id, 1)
    with pytest.raises(ConversationError, match="1 \\(útil\\) o -1"):
        sql_repo.set_feedback(message_id, 5)


def test_borrar_conversacion_arrastra_sus_mensajes(sql_repo):
    sql_repo.ensure_conversation("c1")
    sql_repo.add_user_message("c1", "hola")
    sql_repo.delete_conversation("c1")
    assert sql_repo.list_conversations() == []


# ---------------------------------------------------------------- memoria ---
def test_la_ventana_recorta_a_n_mensajes(settings):
    repo = InMemoryConversationRepository()
    repo.ensure_conversation("c1")
    for i in range(12):
        repo.add_user_message("c1", f"m{i}")

    memoria = ConversationMemory(repo, settings)
    assert len(memoria.load("c1")) == settings.history_window_size


def test_la_ventana_no_empieza_por_una_respuesta_huerfana(settings):
    """Una respuesta sin su pregunta confunde al modelo: se descarta."""
    repo = InMemoryConversationRepository()
    repo.ensure_conversation("c1")
    repo.add_user_message("c1", "pregunta 1")
    repo.add_assistant_message("c1", _answer("c1", "respuesta 1"))
    repo.add_user_message("c1", "pregunta 2")
    repo.add_assistant_message("c1", _answer("c1", "respuesta 2"))

    ventana = ConversationMemory(repo, settings).load("c1", window=3)
    assert ventana[0].role.value == "user"


def test_ventana_cero_desactiva_la_memoria(settings):
    repo = InMemoryConversationRepository()
    repo.ensure_conversation("c1")
    repo.add_user_message("c1", "hola")
    assert ConversationMemory(repo, settings).load("c1", window=0) == []


# -------------------------------------------------------------- analítica ---
def test_informe_vacio_no_falla(sql_repo):
    report = AnalyticsService().report()
    assert report.total_conversations == 0
    assert report.grounded_rate == 0.0


def test_el_informe_recorre_el_historico_completo(sql_repo):
    """Requisito: recorrer el histórico para extraer métricas de impacto."""
    for i in range(3):
        cid = f"conv-{i}"
        sql_repo.ensure_conversation(cid, user_id=f"user-{i % 2}")
        sql_repo.add_user_message(cid, "¿Qué tasas tiene el crédito de vivienda?")
        sql_repo.add_assistant_message(cid, _answer(cid))

    report = AnalyticsService().report()

    assert report.total_conversations == 3
    assert report.total_questions == 3
    assert report.total_answers == 3
    assert report.unique_users == 2
    assert report.grounded_rate == 1.0
    assert report.avg_latency_ms == 1200
    assert report.avg_retrieval_ms == 90
    assert report.avg_llm_ms == 880
    assert report.total_prompt_tokens == 2400
    assert report.estimated_cost_usd > 0        # gpt-4o-mini está tarifado
    assert report.models_used == {"gpt-4o-mini": 3}


def test_el_coste_reconoce_los_ids_con_fecha_que_devuelve_la_api(sql_repo):
    """Regresión: la API responde con la instantánea (`gpt-4o-mini-2024-07-18`).

    Con búsqueda exacta en la tabla de precios el coste salía siempre 0.
    """
    from rag_assistant.analytics.service import _pricing_for

    assert _pricing_for("gpt-4o-mini-2024-07-18") == _pricing_for("gpt-4o-mini")
    assert _pricing_for("modelo-inexistente") == (0.0, 0.0)
    # El prefijo más largo gana: 'gpt-4o-mini-...' no debe tarifarse como 'gpt-4o'.
    assert _pricing_for("gpt-4o-mini-2024-07-18") != _pricing_for("gpt-4o-2024-08-06")

    sql_repo.ensure_conversation("c1")
    sql_repo.add_user_message("c1", "pregunta")
    sql_repo.add_assistant_message("c1", _answer("c1", model="gpt-4o-mini-2024-07-18"))

    report = AnalyticsService().report()
    assert report.estimated_cost_usd > 0
    assert report.untariffed_models == []


def test_una_variante_desconocida_no_hereda_la_tarifa_de_su_familia():
    """Regresión: `gpt-4.1-nano` se estaba cobrando a precio de `gpt-4.1`.

    La búsqueda por prefijo más largo casaba `gpt-4.1-nano-2025-04-14` con la
    clave `gpt-4.1`, aplicándole 2,00 y 8,00 en vez de su tarifa real: el coste
    salía unas veinte veces inflado. Y el fallo era silencioso, porque al casar
    con una clave el modelo tampoco aparecía en `untariffed_models`.

    Solo se recorta el sufijo de fecha; una variante desconocida vale 0 y se
    reporta.
    """
    from rag_assistant.analytics.service import _PRICING_USD_PER_MTOK, _pricing_for

    familia = _PRICING_USD_PER_MTOK["gpt-4.1"]
    assert _pricing_for("gpt-4.1-nano-2025-04-14") != familia
    assert _pricing_for("gpt-4.1-nano") != familia

    # Una variante que de verdad no está en la tabla se reporta, no se inventa.
    assert _pricing_for("gpt-4.1-turbo-inventado") == (0.0, 0.0)
    assert _pricing_for("gpt-4o-imaginario-2030-01-01") == (0.0, 0.0)

    # Y el recorte del sufijo de fecha sigue funcionando para todos los alias.
    for alias in _PRICING_USD_PER_MTOK:
        assert _pricing_for(f"{alias}-2024-07-18") == _PRICING_USD_PER_MTOK[alias]


def test_el_informe_detecta_preguntas_sin_responder(sql_repo):
    """La métrica más accionable: qué contenido falta por indexar."""
    sql_repo.ensure_conversation("c1")
    sql_repo.add_user_message("c1", "¿Cuál es el horario de la oficina de Chapinero?")
    sql_repo.add_assistant_message(
        "c1", _answer("c1", "No encontré esa información", grounded=False)
    )

    report = AnalyticsService().report()
    assert report.grounded_rate == 0.0
    assert "¿Cuál es el horario de la oficina de Chapinero?" in report.unanswered_questions


def test_el_informe_agrega_temas_y_paginas_citadas(sql_repo):
    for i in range(4):
        cid = f"c{i}"
        sql_repo.ensure_conversation(cid)
        sql_repo.add_user_message(cid, "¿Qué requisitos tiene la hipoteca de vivienda?")
        sql_repo.add_assistant_message(cid, _answer(cid))

    report = AnalyticsService().report()
    terminos = {t["term"] for t in report.top_topics}
    assert "vivienda" in terminos and "requisitos" in terminos
    assert report.top_cited_pages[0]["citations"] == 4


def test_el_informe_calcula_satisfaccion(sql_repo):
    sql_repo.ensure_conversation("c1")
    positivo = sql_repo.add_assistant_message("c1", _answer("c1"))
    negativo = sql_repo.add_assistant_message("c1", _answer("c1"))
    sql_repo.set_feedback(positivo, 1)
    sql_repo.set_feedback(negativo, -1)

    report = AnalyticsService().report()
    assert report.feedback_positive == 1
    assert report.feedback_negative == 1
    assert report.satisfaction_rate == 0.5
