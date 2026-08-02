"""Jerarquía de excepciones del dominio.

Una jerarquía propia permite que las capas superiores (API, CLI, UI) distingan
un fallo *esperado y explicable* (p. ej. el sitio bloqueó al scraper) de un bug
real, y traduzcan cada caso a un código HTTP y a un mensaje accionable para el
usuario en vez de un 500 genérico.
"""

from __future__ import annotations


class RAGAssistantError(Exception):
    """Raíz de todos los errores controlados del sistema."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        return f"{self.message} ({self.detail})" if self.detail else self.message


class ConfigurationError(RAGAssistantError):
    """Configuración ausente, inválida o incoherente."""


# --------------------------------------------------------------- scraping ---
class ScrapingError(RAGAssistantError):
    """Error genérico durante el scraping."""


class FetchError(ScrapingError):
    """No se pudo descargar una URL (red, timeout, status HTTP no exitoso)."""

    def __init__(self, url: str, *, status_code: int | None = None, detail: str | None = None):
        super().__init__(f"Fallo al descargar {url}", detail=detail)
        self.url = url
        self.status_code = status_code


class BlockedByTargetError(FetchError):
    """El sitio objetivo bloqueó la petición (403/429, protección anti-bot)."""


class RobotsDisallowedError(ScrapingError):
    """`robots.txt` prohíbe explícitamente rastrear la URL."""


class ExtractionError(ScrapingError):
    """El HTML se descargó pero no se pudo extraer contenido útil."""


# -------------------------------------------------------------- indexación --
class IndexingError(RAGAssistantError):
    """Error durante el chunking, la vectorización o la escritura en el índice."""


class EmbeddingError(IndexingError):
    """El modelo de embeddings falló o no pudo cargarse."""


class VectorStoreError(IndexingError):
    """Error de comunicación con la base de datos vectorial."""


class CollectionNotFoundError(VectorStoreError):
    """La colección aún no existe: hay que ejecutar la ingesta."""


# --------------------------------------------------------------------- rag --
class RetrievalError(RAGAssistantError):
    """Error al recuperar documentos relevantes."""


class LLMError(RAGAssistantError):
    """Error al invocar el modelo de lenguaje."""


class LLMRateLimitError(LLMError):
    """El proveedor del LLM aplicó rate-limiting."""


class LLMAuthenticationError(LLMError):
    """Credenciales del proveedor del LLM ausentes o inválidas."""


# ----------------------------------------------------------- conversación ---
class ConversationError(RAGAssistantError):
    """Error en la persistencia o recuperación del historial."""


class ConversationNotFoundError(ConversationError):
    """No existe una conversación con el ID indicado."""
