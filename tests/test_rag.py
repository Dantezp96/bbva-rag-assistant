"""Pipeline RAG: prompt builder, cadena de etapas y motor completo."""

from __future__ import annotations

import pytest
from tests.conftest import FakeVectorStore

from rag_assistant.conversation import InMemoryConversationRepository
from rag_assistant.core.models import Message, RetrievedChunk, Role
from rag_assistant.rag import RAGEngine
from rag_assistant.rag.prompt_builder import PromptBuilder
from rag_assistant.rag.stages import NO_CONTEXT_ANSWER


def _chunks(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=f"c{i}",
            text=f"Contenido del fragmento número {i} sobre cuentas de ahorro.",
            url=f"https://www.bbva.com.co/pagina-{i}.html",
            title=f"Página {i}",
            score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


# --------------------------------------------------------- prompt builder ---
def test_prompt_incluye_sistema_contexto_y_pregunta():
    messages, citations = (
        PromptBuilder()
        .with_context(_chunks(2))
        .with_question("¿Qué cuentas de ahorro hay?")
        .build()
    )
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "¿Qué cuentas de ahorro hay?" in messages[-1]["content"]
    assert "[1]" in messages[-1]["content"] and "[2]" in messages[-1]["content"]
    assert [c.index for c in citations] == [1, 2]


def test_la_cita_separa_similitud_vectorial_de_puntuacion_del_reranker():
    """No deben mezclarse: coseno está en [0,1] y el reranker devuelve logits."""
    chunk = _chunks(1)[0]
    chunk.rerank_score = 6.42
    _, citations = PromptBuilder().with_context([chunk]).with_question("q").build()
    assert citations[0].score == pytest.approx(0.9)
    assert citations[0].rerank_score == pytest.approx(6.42)


def test_prompt_respeta_el_presupuesto_de_contexto():
    largos = [
        RetrievedChunk(id="x", text="a" * 5000, url="https://x", title="T", score=0.9)
        for _ in range(5)
    ]
    builder = PromptBuilder(context_max_chars=1200).with_context(largos).with_question("q")
    messages, _ = builder.build()
    assert len(messages[-1]["content"]) < 3000


def test_prompt_sin_contexto_instruye_a_no_inventar():
    messages, citations = PromptBuilder().with_context([]).with_question("q").build()
    assert citations == []
    assert "No se recuperó ningún fragmento" in messages[-1]["content"]


def test_el_historial_se_intercala_entre_sistema_y_pregunta():
    historial = [
        Message(role=Role.USER, content="¿Qué es un CDT?"),
        Message(role=Role.ASSISTANT, content="Es un certificado de depósito a término."),
    ]
    messages, _ = (
        PromptBuilder()
        .with_history(historial)
        .with_context(_chunks(1))
        .with_question("¿Y cuál es el plazo mínimo?")
        .build()
    )
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert "¿Qué es un CDT?" in messages[1]["content"]


# ------------------------------------------------------------------ motor ---
def _engine(settings, embedder, store, llm) -> RAGEngine:
    return RAGEngine(
        settings,
        repository=InMemoryConversationRepository(),
        embedder=embedder,
        store=store,
        llm=llm,
    )


def test_ask_devuelve_respuesta_con_citas(settings, fake_embedder, fake_llm):
    store = FakeVectorStore(results=_chunks(3))
    answer = _engine(settings, fake_embedder, store, fake_llm).ask("¿Qué cuentas hay?")

    assert answer.answer == "Respuesta de prueba basada en [1]."
    assert answer.conversation_id
    assert answer.citations
    assert answer.model == "fake-model"
    assert answer.total_tokens == 160
    assert answer.latency_ms >= 0


def test_sin_contexto_no_se_llama_al_llm(settings, fake_embedder, fake_llm):
    """Cortocircuito de la cadena: ahorra coste y evita que el modelo invente."""
    store = FakeVectorStore(results=[])
    answer = _engine(settings, fake_embedder, store, fake_llm).ask("pregunta sin cobertura")

    assert answer.answer == NO_CONTEXT_ANSWER
    assert answer.grounded is False
    assert fake_llm.calls == []


def test_la_memoria_recuerda_los_mensajes_previos(settings, fake_embedder, fake_llm):
    """Requisito: mantener el historial por ID con los N mensajes anteriores."""
    engine = _engine(settings, fake_embedder, FakeVectorStore(results=_chunks(2)), fake_llm)

    primera = engine.ask("¿Qué es una cuenta de ahorro?", conversation_id="conv-1")
    segunda = engine.ask("¿Y cuánto cuesta?", conversation_id="conv-1")

    assert primera.conversation_id == segunda.conversation_id == "conv-1"
    assert primera.history_used == 0       # conversación nueva
    assert segunda.history_used == 2       # pregunta + respuesta anteriores

    ultimo_prompt = fake_llm.calls[-1]
    assert any("¿Qué es una cuenta de ahorro?" in m["content"] for m in ultimo_prompt)


def test_conversaciones_distintas_no_comparten_memoria(settings, fake_embedder, fake_llm):
    engine = _engine(settings, fake_embedder, FakeVectorStore(results=_chunks(1)), fake_llm)
    engine.ask("pregunta A", conversation_id="conv-A")
    respuesta = engine.ask("pregunta B", conversation_id="conv-B")
    assert respuesta.history_used == 0


def test_la_ventana_de_memoria_es_configurable(settings, fake_embedder, fake_llm):
    """HISTORY_WINDOW_SIZE limita cuántos mensajes se recuerdan (aquí, 4)."""
    engine = _engine(settings, fake_embedder, FakeVectorStore(results=_chunks(1)), fake_llm)
    for i in range(6):
        engine.ask(f"pregunta {i}", conversation_id="conv-larga")
    ultima = engine.ask("pregunta final", conversation_id="conv-larga")
    assert ultima.history_used == settings.history_window_size == 4


def test_el_historial_se_puede_sobrescribir_por_peticion(settings, fake_embedder, fake_llm):
    engine = _engine(settings, fake_embedder, FakeVectorStore(results=_chunks(1)), fake_llm)
    engine.ask("uno", conversation_id="c")
    engine.ask("dos", conversation_id="c")
    sin_memoria = engine.ask("tres", conversation_id="c", history_window=0)
    assert sin_memoria.history_used == 0


def test_un_fallo_del_llm_no_rompe_el_servicio(settings, fake_embedder, fake_llm):
    """Se devuelve un mensaje accionable y se registra el turno fallido."""
    from rag_assistant.core.exceptions import LLMRateLimitError

    class LLMQueFalla(type(fake_llm)):
        def complete(self, messages):
            raise LLMRateLimitError("Rate limit", detail="Reintenta en unos segundos.")

    engine = _engine(settings, fake_embedder, FakeVectorStore(results=_chunks(1)), LLMQueFalla())
    answer = engine.ask("¿Qué cuentas hay?")

    assert "Rate limit" in answer.answer
    assert answer.grounded is False


# ------------------------------------------------- reescritura de consulta ---
def _engine_con_reescritura(settings, embedder, store, llm):
    object.__setattr__(settings, "query_rewrite_enabled", True)
    return _engine(settings, embedder, store, llm)


def test_la_primera_pregunta_no_se_reescribe(settings, fake_embedder, fake_llm):
    """Sin historial no hay nada que resolver: no se gasta una llamada al LLM."""
    engine = _engine_con_reescritura(
        settings, fake_embedder, FakeVectorStore(results=_chunks(2)), fake_llm
    )
    r = engine.ask("¿Qué créditos de vivienda ofrecen?", conversation_id="c1")
    assert r.rewritten is False
    assert r.search_query == "¿Qué créditos de vivienda ofrecen?"
    assert len(fake_llm.calls) == 1  # solo la generación


def test_un_seguimiento_sin_sujeto_se_reescribe_antes_de_buscar(
    settings, fake_embedder, fake_llm
):
    """Regresión del fallo medido: '¿el plazo máximo de ese producto?' recuperaba CDT.

    Se comprueba que la consulta que llega al índice ya lleva el sujeto, y que
    la respuesta se sigue redactando sobre la pregunta original del usuario.
    """
    fake_llm._content = "¿Cuál es el plazo máximo del crédito de vivienda?"
    store = FakeVectorStore(results=_chunks(2))
    engine = _engine_con_reescritura(settings, fake_embedder, store, fake_llm)

    engine.ask("¿Qué créditos de vivienda ofrecen?", conversation_id="c1")
    r = engine.ask("¿Cuál es el plazo máximo de ese producto?", conversation_id="c1")

    assert r.rewritten is True
    assert "vivienda" in r.search_query.lower()
    # La pregunta original, no la reescrita, es la que ve el usuario en el prompt.
    ultimo_prompt = fake_llm.calls[-1][-1]["content"]
    assert "¿Cuál es el plazo máximo de ese producto?" in ultimo_prompt


def test_si_la_reescritura_falla_se_sigue_con_la_pregunta_original(
    settings, fake_embedder, fake_llm
):
    """Reescribir es una mejora, nunca un punto de caída."""

    class LLMQueFallaAlReescribir(type(fake_llm)):
        def complete(self, messages, *, max_tokens=None, temperature=None):
            if max_tokens == 80:  # la llamada de reescritura
                raise RuntimeError("proveedor caído")
            return super().complete(messages)

    llm = LLMQueFallaAlReescribir()
    engine = _engine_con_reescritura(
        settings, fake_embedder, FakeVectorStore(results=_chunks(2)), llm
    )
    engine.ask("pregunta uno", conversation_id="c1")
    r = engine.ask("¿y el plazo?", conversation_id="c1")

    assert r.rewritten is False
    assert r.search_query == "¿y el plazo?"
    assert r.answer  # la respuesta se produjo igualmente


def test_no_se_permite_que_la_reescritura_cambie_el_tema(settings, fake_embedder, fake_llm):
    """Regresión de un fallo observado en vivo.

    Con historial sobre créditos de vehículo, el modelo devolvía la pregunta
    ANTERIOR ante una pregunta nueva y autocontenida sobre cuentas de ahorro.
    Eso no es reescribir: es cambiarle la pregunta al usuario, y corrompe justo
    las preguntas normales, que son la mayoría.
    """
    fake_llm._content = "¿Cuáles son los plazos del crédito de vehículo?"
    engine = _engine_con_reescritura(
        settings, fake_embedder, FakeVectorStore(results=_chunks(2)), fake_llm
    )
    engine.ask("¿Qué plazos tiene el crédito de vehículo?", conversation_id="c1")
    r = engine.ask("¿Qué tipos de cuenta de ahorro hay?", conversation_id="c1")

    assert r.rewritten is False
    assert r.search_query == "¿Qué tipos de cuenta de ahorro hay?"


def test_la_reescritura_se_acepta_si_conserva_el_tema():
    """El complemento del test anterior: añadir el sujeto SÍ es válido."""
    from rag_assistant.rag.stages import QueryRewriteStage

    valida = QueryRewriteStage._es_reescritura_valida
    assert valida("¿Cuál es el plazo máximo de ese producto?",
                  "¿Cuál es el plazo máximo del crédito de vivienda?")
    assert valida("¿Y esa cobra cuota de manejo?",
                  "¿La Cuenta Digital BBVA cobra cuota de manejo?")
    # Sin palabras temáticas propias no hay nada que preservar: se acepta.
    assert valida("¿Y eso?", "¿Cuál es la tasa del crédito de vivienda?")
    # Cambio de tema: se rechaza.
    assert not valida("¿Qué tipos de cuenta de ahorro hay?",
                      "¿Cuáles son los plazos del crédito de vehículo?")


def test_se_descarta_una_reescritura_degenerada(settings, fake_embedder, fake_llm):
    """Si el modelo devuelve basura o un discurso, gana la pregunta original."""
    fake_llm._content = "x"  # demasiado corta para ser una consulta
    engine = _engine_con_reescritura(
        settings, fake_embedder, FakeVectorStore(results=_chunks(2)), fake_llm
    )
    engine.ask("pregunta uno", conversation_id="c1")
    r = engine.ask("¿y el plazo máximo?", conversation_id="c1")
    assert r.search_query == "¿y el plazo máximo?"


# ------------------------------------------ sugerencias de seguimiento ------
def _engine_con_sugerencias(settings, embedder, store, llm, count=3):
    object.__setattr__(settings, "suggestions_enabled", True)
    object.__setattr__(settings, "suggestions_count", count)
    return _engine(settings, embedder, store, llm)


def test_las_sugerencias_se_limpian_y_deduplican():
    """El modelo numera y añade preámbulo pese a prohibírselo: hay que sanear."""
    from rag_assistant.rag.stages import _limpiar_sugerencias

    bruto = (
        "Aquí tienes algunas preguntas:\n"
        "1. ¿Qué requisitos piden?\n"
        "- ¿Cuál es la tasa de interés?\n"
        '"¿Qué requisitos piden?"\n'
        "• ¿Se puede pagar anticipadamente?\n"
        "esto no es una pregunta\n"
        "¿ok?\n"
    )
    salida = _limpiar_sugerencias(bruto, limite=5, pregunta_actual="otra cosa")
    assert salida == [
        "¿Qué requisitos piden?",
        "¿Cuál es la tasa de interés?",
        "¿Se puede pagar anticipadamente?",
    ]


def test_no_se_repite_la_pregunta_que_el_usuario_acaba_de_hacer():
    from rag_assistant.rag.stages import _limpiar_sugerencias

    salida = _limpiar_sugerencias(
        "¿Qué plazos tiene?\n¿Cuál es la tasa?",
        limite=3,
        pregunta_actual="¿Qué plazos tiene?",
    )
    assert salida == ["¿Cuál es la tasa?"]


def test_se_proponen_sugerencias_tras_una_respuesta_fundamentada(
    settings, fake_embedder, fake_llm
):
    fake_llm._content = "¿Qué requisitos piden?\n¿Cuál es la tasa de interés?"
    engine = _engine_con_sugerencias(
        settings, fake_embedder, FakeVectorStore(results=_chunks(3)), fake_llm
    )
    r = engine.ask("¿Qué créditos de vivienda ofrecen?")
    assert r.suggestions == ["¿Qué requisitos piden?", "¿Cuál es la tasa de interés?"]


def test_sin_contexto_no_se_proponen_sugerencias(settings, fake_embedder, fake_llm):
    """Sugerir seguimiento a un 'no encontré información' sería inventar."""
    engine = _engine_con_sugerencias(
        settings, fake_embedder, FakeVectorStore(results=[]), fake_llm
    )
    r = engine.ask("algo que no está en el corpus")
    assert r.suggestions == []
    assert fake_llm.calls == []  # ni siquiera se llamó al LLM


def test_un_fallo_generando_sugerencias_no_afecta_a_la_respuesta(
    settings, fake_embedder, fake_llm
):
    class LLMQueFallaAlSugerir(type(fake_llm)):
        def complete(self, messages, *, max_tokens=None, temperature=None):
            if max_tokens == 120:  # la llamada de sugerencias
                raise RuntimeError("proveedor caído")
            return super().complete(messages)

    engine = _engine_con_sugerencias(
        settings, fake_embedder, FakeVectorStore(results=_chunks(2)), LLMQueFallaAlSugerir()
    )
    r = engine.ask("¿Qué cuentas hay?")
    assert r.suggestions == []
    assert r.answer  # la respuesta se entregó igualmente
    assert r.grounded


def test_se_respeta_el_numero_configurado_de_sugerencias(settings, fake_embedder, fake_llm):
    fake_llm._content = (
        "¿Qué requisitos piden?\n¿Cuál es la tasa?\n¿Qué plazos hay?\n"
        "¿Se puede prepagar?\n¿Cobran seguro?"
    )
    engine = _engine_con_sugerencias(
        settings, fake_embedder, FakeVectorStore(results=_chunks(2)), fake_llm, count=2
    )
    assert len(engine.ask("¿Qué cuentas hay?").suggestions) == 2


def test_pregunta_vacia_no_consume_recursos(settings, fake_embedder, fake_llm):
    engine = _engine(settings, fake_embedder, FakeVectorStore(results=_chunks(1)), fake_llm)
    answer = engine.ask("   ")
    assert fake_llm.calls == []
    assert "Escribe una pregunta" in answer.answer
