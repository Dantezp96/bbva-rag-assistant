"""Construcción del prompt enviado al LLM.

PATRÓN DE DISEÑO: **Builder**.
El prompt final se compone de piezas heterogéneas —instrucciones de sistema,
historial de los últimos N mensajes, contexto recuperado numerado, y la
pregunta actual— cada una con su propio presupuesto de caracteres y su propio
formato. Un Builder con interfaz fluida deja explícito el orden de montaje,
permite omitir partes (una conversación nueva no tiene historial) y concentra
en un solo sitio las reglas de recorte.

La alternativa (f-strings dispersos por el motor) hace imposible razonar sobre
qué entra realmente en la ventana de contexto.
"""

from __future__ import annotations

from rag_assistant.core.models import Citation, Message, RetrievedChunk

SYSTEM_PROMPT = """\
Eres el asistente virtual de BBVA Colombia. Respondes preguntas de usuarios \
internos usando EXCLUSIVAMENTE la información del CONTEXTO extraído del sitio \
web institucional.

REGLAS:
1. Responde únicamente con información presente en el CONTEXTO. No inventes \
cifras, tasas, plazos, requisitos ni nombres de producto.
2. Si el CONTEXTO no contiene la respuesta, dilo con claridad: "No encontré esa \
información en el sitio de BBVA Colombia" y sugiere qué podría consultar el \
usuario. No completes con conocimiento general.
3. Cita las fuentes con marcadores [1], [2]… referidos a los fragmentos del \
CONTEXTO en los que te apoyas. Cita solo lo que realmente uses.
4. Usa el HISTORIAL para resolver referencias ("¿y ese cuánto cuesta?", "el \
anterior"), no como fuente de datos.
5. Responde en español neutro de Colombia, de forma directa y estructurada. \
Usa viñetas cuando enumeres condiciones o requisitos.
6. Para importes, tasas o condiciones, reproduce el dato exactamente como \
aparece en el CONTEXTO.
7. No solicites ni manejes datos personales, credenciales ni información de \
cuentas.
"""

_NO_CONTEXT_NOTE = (
    "No se recuperó ningún fragmento relevante del sitio para esta pregunta. "
    "Indícaselo al usuario siguiendo la regla 2."
)


class PromptBuilder:
    """Construye, paso a paso, la lista de mensajes para el LLM."""

    def __init__(self, *, context_max_chars: int = 8000, history_max_chars: int = 4000) -> None:
        self._context_max_chars = context_max_chars
        self._history_max_chars = history_max_chars
        self._system = SYSTEM_PROMPT
        self._history: list[Message] = []
        self._chunks: list[RetrievedChunk] = []
        self._question = ""
        self._citations: list[Citation] = []

    # ------------------------------------------------------------- fluidez --
    def with_system_prompt(self, prompt: str) -> PromptBuilder:
        self._system = prompt
        return self

    def with_history(self, messages: list[Message]) -> PromptBuilder:
        self._history = messages
        return self

    def with_context(self, chunks: list[RetrievedChunk]) -> PromptBuilder:
        self._chunks = chunks
        return self

    def with_question(self, question: str) -> PromptBuilder:
        self._question = question.strip()
        return self

    # -------------------------------------------------------------- montaje -
    def _render_context(self) -> str:
        """Numera los fragmentos y recorta al presupuesto de caracteres."""
        if not self._chunks:
            self._citations = []
            return _NO_CONTEXT_NOTE

        blocks: list[str] = []
        citations: list[Citation] = []
        used = 0
        for position, chunk in enumerate(self._chunks, start=1):
            header = f"[{position}] {chunk.title} — {chunk.url}"
            body = chunk.text.strip()
            block = f"{header}\n{body}"

            remaining = self._context_max_chars - used
            if remaining <= len(header) + 100:
                break  # no cabe ni un fragmento mínimamente útil
            if len(block) > remaining:
                block = block[:remaining].rsplit(" ", 1)[0] + " […]"

            blocks.append(block)
            used += len(block)
            citations.append(
                Citation(
                    index=position,
                    url=chunk.url,
                    title=chunk.title,
                    score=round(chunk.effective_score, 4),
                )
            )

        self._citations = citations
        return "\n\n---\n\n".join(blocks)

    def _render_history(self) -> list[dict[str, str]]:
        """Historial más reciente primero en prioridad, respetando el presupuesto."""
        rendered: list[dict[str, str]] = []
        used = 0
        for message in reversed(self._history):
            content = message.content.strip()
            if used + len(content) > self._history_max_chars:
                break
            rendered.append({"role": message.role.value, "content": content})
            used += len(content)
        rendered.reverse()
        return rendered

    def build(self) -> tuple[list[dict[str, str]], list[Citation]]:
        """Devuelve `(mensajes, citas)` listos para el proveedor de LLM."""
        context = self._render_context()

        messages: list[dict[str, str]] = [{"role": "system", "content": self._system}]
        messages.extend(self._render_history())
        messages.append(
            {
                "role": "user",
                "content": (
                    f"CONTEXTO (fragmentos del sitio de BBVA Colombia):\n"
                    f"{context}\n\n"
                    f"PREGUNTA DEL USUARIO:\n{self._question}\n\n"
                    f"Responde siguiendo las reglas, citando con [n] las fuentes usadas."
                ),
            }
        )
        return messages, self._citations
