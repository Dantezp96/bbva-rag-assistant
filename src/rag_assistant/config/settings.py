"""Configuración centralizada y externalizada del sistema.

PATRÓN DE DISEÑO: **Singleton** (vía `functools.lru_cache`).
`get_settings()` construye el objeto `Settings` una única vez por proceso y
devuelve siempre la misma instancia. Motivo: parsear el entorno y validar es
trabajo repetido e innecesario, y —más importante— garantiza que **todos** los
componentes (scraper, indexador, RAG, API, CLI) observen exactamente la misma
configuración durante la vida del proceso, evitando estados divergentes.

Se usa `lru_cache` en lugar de una metaclase Singleton clásica porque es
idiomático en Python, thread-safe y permite limpiar la caché en tests
(`get_settings.cache_clear()`), algo que un Singleton rígido dificulta.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProvider = Literal["openai", "ollama"]
EmbeddingProvider = Literal["fastembed"]
VectorStoreProvider = Literal["qdrant"]
FetcherKind = Literal["browser", "http"]
ChunkStrategy = Literal["recursive", "fixed"]


class Settings(BaseSettings):
    """Todos los parámetros operativos del sistema, leídos de entorno/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- app ---
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    # ---------------------------------------------------------------- llm ---
    llm_provider: LLMProvider = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=800, gt=0)
    llm_timeout_seconds: int = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"

    # --------------------------------------------------------- embeddings ---
    embedding_provider: EmbeddingProvider = "fastembed"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    model_cache_dir: Path = Path("./models_cache")

    # ------------------------------------------------------- vector store ---
    vector_store_provider: VectorStoreProvider = "qdrant"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "bbva_web"
    qdrant_distance: Literal["cosine", "dot", "euclid"] = "cosine"

    # ------------------------------------------------------------ scraper ---
    scraper_base_url: str = "https://www.bbva.com.co/"
    # `NoDecode` desactiva el parseo JSON automático de pydantic-settings para
    # estos campos: en el `.env` se declaran como CSV legible (`a,b,c`), no como
    # `["a","b","c"]`. El validador `_split_csv` hace la conversión.
    scraper_allowed_domains: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["www.bbva.com.co"]
    )
    scraper_fetcher: FetcherKind = "browser"
    #: Canal de navegador para el fetcher `browser`. "chrome" = Google Chrome
    #: estable, el único que supera la protección anti-bot de bbva.com.co en
    #: modo headless. Vacío = usar el Chromium empaquetado por Playwright.
    scraper_browser_channel: str | None = "chrome"
    scraper_max_pages: int = Field(default=120, gt=0)
    scraper_max_depth: int = Field(default=3, ge=0)
    scraper_concurrency: int = Field(default=4, gt=0)
    scraper_request_delay: float = Field(default=0.5, ge=0.0)
    scraper_timeout_seconds: int = Field(default=45, gt=0)
    #: Espera adicional (ms) a que la red se calme tras DOMContentLoaded, para
    #: recoger el contenido que inyecta JavaScript. 0 lo desactiva.
    scraper_settle_ms: int = Field(default=2000, ge=0)
    scraper_respect_robots: bool = True
    scraper_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    scraper_url_deny_patterns: Annotated[list[str], NoDecode] = Field(default_factory=list)
    raw_data_dir: Path = Path("./data/raw")
    clean_data_dir: Path = Path("./data/clean")

    # ----------------------------------------------------------- chunking ---
    chunk_strategy: ChunkStrategy = "recursive"
    chunk_size: int = Field(default=900, gt=50)
    chunk_overlap: int = Field(default=150, ge=0)
    chunk_min_size: int = Field(default=120, ge=0)

    # -------------------------------------------- recuperación + reranker ---
    retrieval_top_k: int = Field(default=20, gt=0)
    retrieval_score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    reranker_enabled: bool = True
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = Field(default=5, gt=0)
    context_max_chars: int = Field(default=8000, gt=0)

    # ------------------------------------------------------------ memoria ---
    history_window_size: int = Field(default=6, ge=0)
    history_max_chars: int = Field(default=4000, gt=0)

    # ------------------------------------------------------------- datos ----
    database_url: str = "sqlite+pysqlite:///./data/conversations.db"

    # -------------------------------------------------------------- api ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"
    ui_port: int = 8501
    api_cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------- validadores ----
    @field_validator(
        "scraper_allowed_domains",
        "scraper_url_deny_patterns",
        "api_cors_origins",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Permite declarar listas como CSV en el `.env` (`a,b,c`)."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "openai_api_key", "openai_base_url", "qdrant_api_key", "scraper_browser_channel",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        """Una variable declarada pero vacía en `.env` equivale a "no configurada"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_coherence(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE")
        if self.reranker_enabled and self.reranker_top_n > self.retrieval_top_k:
            # El reranker no puede devolver más documentos de los que recibe.
            object.__setattr__(self, "reranker_top_n", self.retrieval_top_k)
        return self

    # ---------------------------------------------------------- utilidades --
    @property
    def deny_regexes(self) -> list[re.Pattern[str]]:
        return [re.compile(pattern) for pattern in self.scraper_url_deny_patterns]

    def ensure_directories(self) -> None:
        """Crea los directorios de datos si no existen (idempotente)."""
        for directory in (self.raw_data_dir, self.clean_data_dir, self.model_cache_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, object]:
        """Volcado de configuración seguro para logs: oculta los secretos."""
        data = self.model_dump(mode="json")
        for secret_key in ("openai_api_key", "qdrant_api_key", "database_url"):
            if data.get(secret_key):
                data[secret_key] = "***redacted***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Punto de acceso único a la configuración (Singleton)."""
    return Settings()
