"""Pipeline RAG: prompt builder, cadena de etapas y motor completo."""

from __future__ import annotations

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


def test_pregunta_vacia_no_consume_recursos(settings, fake_embedder, fake_llm):
    engine = _engine(settings, fake_embedder, FakeVectorStore(results=_chunks(1)), fake_llm)
    answer = engine.ask("   ")
    assert fake_llm.calls == []
    assert "Escribe una pregunta" in answer.answer
