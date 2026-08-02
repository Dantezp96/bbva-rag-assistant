"""Chunking e indexación."""

from __future__ import annotations

import pytest

from rag_assistant.core.exceptions import IndexingError
from rag_assistant.indexing import IndexingPipeline
from rag_assistant.indexing.chunking import FixedChunker, RecursiveChunker, create_chunker

TEXTO = (
    "## Crédito de vivienda\n"
    "Financiamos hasta el 70% del valor del inmueble para vivienda No VIS. "
    "Los plazos van desde 5 hasta 30 años según el perfil del solicitante.\n\n"
    "## Requisitos\n"
    "Ser mayor de edad, acreditar ingresos y no tener reportes negativos "
    "en centrales de riesgo al momento de la solicitud.\n\n"
    "## Tasas\n"
    "La tasa de interés se define según el plazo y el monto financiado."
)


@pytest.mark.parametrize("cls", [RecursiveChunker, FixedChunker])
def test_ningun_chunk_excede_el_tamano_configurado(cls):
    chunks = cls(chunk_size=120, chunk_overlap=20).split(TEXTO)
    assert chunks
    assert all(len(c) <= 120 for c in chunks)


def test_el_chunking_no_pierde_contenido():
    chunks = RecursiveChunker(chunk_size=150, chunk_overlap=30).split(TEXTO)
    unido = " ".join(chunks)
    assert "Financiamos hasta el 70%" in unido
    assert "centrales de riesgo" in unido


def test_overlap_invalido_se_rechaza_al_construir():
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_document_anota_procedencia(sample_document):
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=40).chunk_document(sample_document)
    assert chunks
    assert all(c.url == sample_document.url for c in chunks)
    assert all(c.title == sample_document.title for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.total_chunks == len(chunks) for c in chunks)


def test_los_ids_son_deterministas_para_ingesta_idempotente(sample_document):
    """Reindexar el mismo contenido debe actualizar el punto, no duplicarlo."""
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
    primeros = chunker.chunk_document(sample_document)
    segundos = chunker.chunk_document(sample_document)
    assert [c.id for c in primeros] == [c.id for c in segundos]


def test_factory_devuelve_la_estrategia_configurada(settings):
    assert isinstance(create_chunker(settings, strategy="fixed"), FixedChunker)
    assert isinstance(create_chunker(settings, strategy="recursive"), RecursiveChunker)


def test_factory_rechaza_estrategia_desconocida(settings):
    from rag_assistant.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="desconocida"):
        create_chunker(settings, strategy="inventada")


# --------------------------------------------------------------- pipeline ---
def test_pipeline_indexa_documentos(settings, fake_embedder, fake_store, sample_document):
    pipeline = IndexingPipeline(settings, embedder=fake_embedder, store=fake_store)
    report = pipeline.run([sample_document], recreate=True)

    assert report.documents == 1
    assert report.chunks > 0
    assert report.vectors_written == report.chunks
    assert report.indexed_total == fake_store.count()
    assert fake_store.recreated is True
    assert fake_store.dimension == fake_embedder.dimension


def test_pipeline_falla_con_mensaje_util_si_no_hay_corpus(settings, fake_embedder, fake_store):
    with pytest.raises(IndexingError, match="No hay documentos"):
        IndexingPipeline(settings, embedder=fake_embedder, store=fake_store).run([])
