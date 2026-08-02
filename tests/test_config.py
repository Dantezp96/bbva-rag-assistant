"""Configuración: Singleton, parseo de listas CSV y validaciones cruzadas."""

from __future__ import annotations

import pytest

from rag_assistant.config import Settings, get_settings


def test_get_settings_es_singleton():
    """Todas las capas deben ver exactamente la misma instancia."""
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()


def test_listas_se_declaran_como_csv_en_el_env():
    """En el .env las listas se escriben `a,b,c`, no como JSON."""
    settings = Settings(
        scraper_allowed_domains="www.bbva.com.co, bbva.com.co",
        scraper_url_deny_patterns="/buscador,\\.pdf$",
    )
    assert settings.scraper_allowed_domains == ["www.bbva.com.co", "bbva.com.co"]
    assert len(settings.deny_regexes) == 2
    assert settings.deny_regexes[0].search("https://x/buscador/algo")


def test_overlap_mayor_que_chunk_size_es_error():
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=200)


def test_reranker_no_puede_devolver_mas_de_lo_que_recibe():
    """top_n se ajusta a top_k en vez de fallar: es un ajuste, no un error."""
    settings = Settings(retrieval_top_k=5, reranker_top_n=50, reranker_enabled=True)
    assert settings.reranker_top_n == 5


def test_variable_vacia_equivale_a_no_configurada():
    assert Settings(openai_api_key="   ").openai_api_key is None


def test_redacted_oculta_los_secretos():
    dump = Settings(openai_api_key="sk-secreto-real").redacted()
    assert dump["openai_api_key"] == "***redacted***"
    assert "sk-secreto-real" not in str(dump)
