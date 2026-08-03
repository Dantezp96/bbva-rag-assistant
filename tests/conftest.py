"""Fixtures compartidas.

Los tests corren sin infraestructura: SQLite en fichero temporal y dobles para
LLM, embeddings y base vectorial. Esto es posible precisamente por las
abstracciones del diseño (Strategy, Adapter, Repository).
"""

from __future__ import annotations

import pytest

from rag_assistant.config import Settings
from rag_assistant.core.models import Chunk, CleanDocument, RetrievedChunk
from rag_assistant.indexing.embeddings.base import EmbeddingProvider
from rag_assistant.indexing.vector_store.base import VectorStore
from rag_assistant.rag.llm.base import LLMProvider, LLMResponse


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Configuración aislada por test, con SQLite en disco temporal."""
    return Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'test.db'}",
        raw_data_dir=tmp_path / "raw",
        clean_data_dir=tmp_path / "clean",
        model_cache_dir=tmp_path / "models",
        openai_api_key="sk-test",
        history_window_size=4,
        retrieval_top_k=5,
        reranker_top_n=3,
        # Las etapas opcionales se apagan por defecto y cada test enciende la
        # que va a ejercitar. Si no, añadir una etapa nueva a la cadena rompe
        # tests ajenos por efecto colateral (conteo de llamadas, tokens), que
        # es justo lo que pasó al introducir las sugerencias.
        reranker_enabled=False,
        query_rewrite_enabled=False,
        suggestions_enabled=False,
        chunk_size=200,
        chunk_overlap=40,
        chunk_min_size=0,
    )


class FakeEmbedder(EmbeddingProvider):
    """Embeddings deterministas: el vector depende del contenido, no del azar."""

    name = "fake"

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    def _vector(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) or 1
        return [((seed * (i + 1)) % 97) / 97 for i in range(self._dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeVectorStore(VectorStore):
    """Índice en memoria que devuelve los chunks en el orden en que se insertaron."""

    name = "fake"

    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._results = results
        self.recreated = False

    def ensure_collection(self, dimension: int, *, recreate: bool = False) -> None:
        self.dimension = dimension
        self.recreated = recreate

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)
        return len(chunks)

    def search(self, vector, *, top_k, score_threshold=None, filters=None):
        if self._results is not None:
            return self._results[:top_k]
        return [
            RetrievedChunk(
                id=c.id, text=c.text, url=c.url, title=c.title, score=0.9 - i * 0.05
            )
            for i, c in enumerate(self.chunks[:top_k])
        ]

    def count(self) -> int:
        return len(self.chunks)

    def health(self) -> bool:
        return True

    def info(self) -> dict:
        return {"provider": self.name, "reachable": True, "points": self.count()}


class FakeLLM(LLMProvider):
    """LLM que devuelve una respuesta fija y guarda los mensajes recibidos."""

    name = "fake"

    def __init__(self, content: str = "Respuesta de prueba basada en [1].") -> None:
        self._content = content
        self.calls: list[list[dict[str, str]]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            content=self._content, model=self.model, prompt_tokens=120, completion_tokens=40
        )

    def health(self) -> bool:
        return True


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_document() -> CleanDocument:
    return CleanDocument(
        url="https://www.bbva.com.co/personas/productos/cuentas/ahorro.html",
        title="Cuentas de ahorro",
        text=(
            "Cuentas de ahorro BBVA Colombia\n\n"
            "## Cuenta de Ahorro Digital\n"
            "Ábrela 100% en línea desde la app. Sin cuota de manejo el primer año.\n"
            "- Sin monto mínimo de apertura\n"
            "- Retiros gratis en cajeros BBVA\n\n"
            "## Cuenta Nómina\n"
            "Diseñada para el pago de tu salario. Exenta de cuota de manejo "
            "mientras recibas la nómina en la cuenta.\n"
            "- Tarjeta débito sin costo\n"
            "- Transferencias ilimitadas por la app\n"
        ),
        description="Conoce las cuentas de ahorro de BBVA Colombia.",
        section="personas/productos",
    )
