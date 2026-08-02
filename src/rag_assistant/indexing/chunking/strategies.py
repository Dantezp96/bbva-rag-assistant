"""Implementaciones concretas de troceado."""

from __future__ import annotations

import re

from rag_assistant.indexing.chunking.base import ChunkingStrategy

#: Separadores ordenados de mayor a menor semántica: se intenta cortar primero
#: entre secciones, luego entre párrafos, después entre frases y, en último
#: extremo, entre palabras.
_SEPARATORS = ["\n## ", "\n\n", "\n", ". ", " "]


class RecursiveChunker(ChunkingStrategy):
    """Troceado recursivo por separadores, respetando la estructura del texto.

    Es la estrategia por defecto: en páginas de producto de un banco, cortar
    por encabezado y párrafo mantiene juntas las condiciones de un producto
    (tasas, requisitos, beneficios), que es justo lo que el usuario pregunta.
    """

    name = "recursive"

    def split(self, text: str) -> list[str]:
        chunks = self._split_recursive(text, _SEPARATORS)
        return self._merge_with_overlap(chunks)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            # Sin separadores utilizables: corte duro por tamaño.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        separator, rest = separators[0], separators[1:]
        parts = text.split(separator)
        out: list[str] = []
        for part in parts:
            piece = part if separator in ("\n\n", "\n", " ") else separator + part
            if len(piece) > self.chunk_size:
                out.extend(self._split_recursive(piece, rest))
            elif piece.strip():
                out.append(piece)
        return out

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Reagrupa fragmentos pequeños hasta `chunk_size`, con solapamiento."""
        merged: list[str] = []
        buffer = ""
        for piece in pieces:
            candidate = f"{buffer}\n{piece}".strip() if buffer else piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue
            if buffer:
                merged.append(buffer)
                # El solapamiento evita perder la frase que queda a caballo
                # entre dos chunks.
                tail = buffer[-self.chunk_overlap :] if self.chunk_overlap else ""
                buffer = f"{tail}\n{piece}".strip() if tail else piece
            else:
                buffer = piece
            while len(buffer) > self.chunk_size:
                merged.append(buffer[: self.chunk_size])
                buffer = buffer[self.chunk_size - self.chunk_overlap :]
        if buffer.strip():
            merged.append(buffer)
        return merged


class FixedChunker(ChunkingStrategy):
    """Ventana deslizante de tamaño fijo, alineada a frontera de palabra.

    Simple y predecible. Útil como línea base para comparar contra la
    estrategia recursiva.
    """

    name = "fixed"

    def split(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
        return chunks
