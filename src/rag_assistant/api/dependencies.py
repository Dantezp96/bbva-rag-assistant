"""Dependencias compartidas de la API.

El `RAGEngine` se construye una sola vez por proceso: carga modelos ONNX y
mantiene pools de conexiones, así que crearlo por petición sería inaceptable.
"""

from __future__ import annotations

from functools import lru_cache

from rag_assistant.analytics import AnalyticsService
from rag_assistant.rag import RAGEngine


@lru_cache(maxsize=1)
def get_engine() -> RAGEngine:
    return RAGEngine()


@lru_cache(maxsize=1)
def get_analytics() -> AnalyticsService:
    return AnalyticsService()
