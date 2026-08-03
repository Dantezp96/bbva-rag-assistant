"""Etapas del pipeline RAG.

PATRÓN DE DISEÑO: **Chain of Responsibility**.
Responder una pregunta es una secuencia de pasos —recuperar, reordenar,
construir el prompt, generar— donde cada paso enriquece un contexto compartido
y decide si pasa el testigo al siguiente. Modelarlo como una cadena aporta:

* **Composición**: activar o desactivar el reranker es añadir o quitar un
  eslabón, no meter un `if` en un método de 200 líneas.
* **Cortocircuito**: si la recuperación no encuentra nada relevante, la etapa
  puede terminar la cadena y devolver una respuesta honesta sin gastar una
  llamada al LLM.
* **Observabilidad**: cada eslabón mide su propio tiempo, lo que da la
  telemetría por etapa que consume el módulo de analítica.
"""

from __future__ import annotations

import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import Citation, Message, RetrievedChunk
from rag_assistant.indexing.embeddings import EmbeddingProvider
from rag_assistant.indexing.vector_store import VectorStore
from rag_assistant.rag.llm import LLMProvider
from rag_assistant.rag.prompt_builder import PromptBuilder
from rag_assistant.rag.reranker import CrossEncoderReranker

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "No encontré información sobre eso en el contenido indexado del sitio de "
    "BBVA Colombia. Puedes reformular la pregunta con otros términos, o "
    "consultar directamente en https://www.bbva.com.co/."
)


@dataclass
class RAGContext:
    """Estado que atraviesa la cadena, enriquecido en cada etapa."""

    question: str
    conversation_id: str = ""
    history: list[Message] = field(default_factory=list)

    #: Consulta que se usa para BUSCAR, que no siempre es la que escribió el
    #: usuario: en un seguimiento como "¿y cuál es el plazo?" hay que
    #: reconstruirla con el historial antes de tocar el índice. La respuesta,
    #: en cambio, se redacta siempre sobre `question`.
    search_query: str = ""
    rewritten: bool = False
    rewrite_ms: int = 0

    candidates: list[RetrievedChunk] = field(default_factory=list)
    documents: list[RetrievedChunk] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    answer: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    retrieval_ms: int = 0
    rerank_ms: int = 0
    llm_ms: int = 0

    reranked: bool = False
    grounded: bool = True
    stop: bool = False  # cortocircuita el resto de la cadena

    suggestions: list[str] = field(default_factory=list)
    suggestions_ms: int = 0


class Stage(ABC):
    """Eslabón de la cadena.

    La cadena efectiva es:
        reescritura -> recuperación -> reranking -> prompt -> generación
    Los eslabones primero y tercero son opcionales (`QUERY_REWRITE_ENABLED`,
    `RERANKER_ENABLED`), y añadir el de reescritura no obligó a tocar ninguno
    de los demás.
    """

    name: str = "stage"

    def __init__(self) -> None:
        self._next: Stage | None = None

    def set_next(self, stage: Stage) -> Stage:
        """Encadena y devuelve el siguiente eslabón (permite encadenado fluido)."""
        self._next = stage
        return stage

    @property
    def next_stage(self) -> Stage | None:
        """Siguiente eslabón, o `None` si es el último.

        Lo usa el modo streaming para recorrer la cadena paso a paso y poder
        anunciar qué etapa está en curso, en vez de delegar en `handle` y no
        poder emitir nada hasta el final.
        """
        return self._next

    def handle(self, context: RAGContext) -> RAGContext:
        context = self.process(context)
        if context.stop or self._next is None:
            return context
        return self._next.handle(context)

    @abstractmethod
    def process(self, context: RAGContext) -> RAGContext:
        """Trabajo propio de la etapa."""


_REWRITE_SYSTEM = """\
Reescribe la ÚLTIMA pregunta del usuario como una consulta de búsqueda \
autocontenida, resolviendo pronombres y referencias implícitas con el \
historial ("ese producto", "y el plazo", "cuánto cuesta").

REGLAS:
- Devuelve SOLO la consulta reescrita, sin comillas ni explicación.
- CONSERVA TODAS las palabras clave del usuario. Solo puedes AÑADIR el sujeto \
que falta, nunca sustituir el tema por otro.
- Si la pregunta ya se entiende por sí sola, devuélvela EXACTAMENTE IGUAL.
- No respondas la pregunta ni añadas información nueva.

Ejemplo correcto (falta el sujeto, se añade):
  Historial: usuario: "¿Qué créditos de vivienda ofrecen?"
  Pregunta:  "¿Cuál es el plazo máximo de ese producto?"
  Salida:    ¿Cuál es el plazo máximo del crédito de vivienda?

Ejemplo correcto (ya es autocontenida, no se toca):
  Historial: usuario: "¿Qué plazos tiene el crédito de vehículo?"
  Pregunta:  "¿Qué tipos de cuenta de ahorro hay?"
  Salida:    ¿Qué tipos de cuenta de ahorro hay?

NUNCA sustituyas la pregunta por el tema anterior. Si el usuario cambia de \
tema, respétalo.
"""

#: Palabras sin carga temática: no sirven para comprobar que la reescritura
#: conserva el asunto de la pregunta.
_PALABRAS_VACIAS_REESCRITURA = {
    "cual", "cuales", "cuanto", "cuanta", "cuantos", "cuantas", "como", "donde",
    "cuando", "para", "por", "con", "los", "las", "del", "una", "uno", "que",
    "puedo", "puede", "tiene", "hay", "son", "ser", "esta", "este", "esos",
    "esas", "eso", "esa", "ese", "sobre", "tienen", "ofrecen", "ofrece", "mas",
    "pero", "todo", "toda", "desde", "hasta", "entre", "porque", "sin",
}


def _palabras_tematicas(texto: str) -> set[str]:
    """Palabras con carga temática de una pregunta, sin tildes ni signos."""
    normalizado = unicodedata.normalize("NFKD", texto.lower())
    normalizado = "".join(c for c in normalizado if not unicodedata.combining(c))
    return {
        palabra
        for palabra in re.findall(r"[a-z]{4,}", normalizado)
        if palabra not in _PALABRAS_VACIAS_REESCRITURA
    }


class QueryRewriteStage(Stage):
    """Reconstruye la consulta de búsqueda usando el historial.

    **El problema que resuelve, medido.** Inyectar el historial en el prompt
    hace que el LLM *redacte* bien, pero la búsqueda vectorial seguía usando la
    pregunta literal. Con "¿Cuál es el plazo máximo de ese producto?" —siete
    palabras, ninguna del dominio— tras una conversación sobre vivienda, el
    índice devolvía fragmentos sobre CDT y el sistema respondía "36 meses":
    fiel al contexto recuperado, pero contestando a otro producto.

    Es un fallo silencioso y por eso peligroso: la respuesta viene con fuentes
    y suena segura. Esta etapa lo corta de raíz reconstruyendo el sujeto antes
    de tocar el índice.

    Diseño:
    - Solo actúa si hay historial; una primera pregunta no se toca.
    - Llamada barata y acotada (pocos tokens, temperatura 0).
    - Ante cualquier fallo se sigue con la pregunta original: reescribir es una
      mejora, nunca un punto de caída.
    - La respuesta se redacta siempre sobre la pregunta ORIGINAL; lo reescrito
      solo alimenta la búsqueda.
    """

    name = "rewrite"

    def __init__(self, llm: LLMProvider, *, max_history: int = 4) -> None:
        super().__init__()
        self._llm = llm
        self._max_history = max_history

    def process(self, context: RAGContext) -> RAGContext:
        context.search_query = context.question
        if not context.history:
            return context

        historial = "\n".join(
            f"{m.role.value}: {m.content[:300]}" for m in context.history[-self._max_history :]
        )
        started = time.perf_counter()
        try:
            respuesta = self._llm.complete(
                [
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Historial:\n{historial}\n\nPregunta: {context.question}",
                    },
                ],
                max_tokens=80,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - degradación deliberada
            logger.warning("rag.rewrite_failed", error=str(exc))
            return context

        nueva = respuesta.content.strip().strip('"').strip()
        context.rewrite_ms = int((time.perf_counter() - started) * 1000)
        context.prompt_tokens += respuesta.prompt_tokens
        context.completion_tokens += respuesta.completion_tokens

        if not self._es_reescritura_valida(context.question, nueva):
            return context

        context.search_query = nueva
        context.rewritten = nueva.lower() != context.question.lower()
        if context.rewritten:
            logger.info(
                "rag.query_rewritten",
                original=context.question[:80],
                reescrita=nueva[:80],
                ms=context.rewrite_ms,
            )
        return context

    @staticmethod
    def _es_reescritura_valida(original: str, nueva: str) -> bool:
        """Comprueba que la reescritura AÑADE contexto en vez de sustituir el tema.

        Sin esta comprobación el modelo, al ver el historial, a veces devuelve
        directamente la pregunta ANTERIOR. Medido en vivo:

            original  = "¿Qué tipos de cuenta de ahorro hay?"
            reescrita = "¿Cuáles son los plazos del crédito de vehículo?"

        Eso no es reescribir, es cambiarle la pregunta al usuario, y es peor que
        no reescribir nada: corrompe justamente las preguntas autocontenidas,
        que son la mayoría. La invariante es simple y verificable: una
        reescritura legítima conserva al menos una palabra temática del
        original. Si el usuario no aportó ninguna ("¿y eso?"), no hay nada que
        preservar y se acepta —que es justo cuando reescribir hace más falta—.
        """
        if not nueva or not (3 < len(nueva) <= max(200, len(original) * 4)):
            return False

        tematicas_original = _palabras_tematicas(original)
        if not tematicas_original:
            return True
        if tematicas_original & _palabras_tematicas(nueva):
            return True

        logger.warning(
            "rag.rewrite_rechazada",
            original=original[:80],
            reescrita=nueva[:80],
            motivo="no conserva ninguna palabra temática del usuario",
        )
        return False


class RetrievalStage(Stage):
    """Vectoriza la pregunta y recupera los candidatos del índice."""

    name = "retrieval"

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        top_k: int,
        score_threshold: float,
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold

    def process(self, context: RAGContext) -> RAGContext:
        started = time.perf_counter()
        # Se busca con `search_query` (reescrita si hubo seguimiento); si la
        # etapa de reescritura no corrió, es la pregunta original.
        consulta = context.search_query or context.question
        vector = self._embedder.embed_query(consulta)
        candidates = self._store.search(
            vector, top_k=self._top_k, score_threshold=self._score_threshold
        )
        context.candidates = candidates
        context.retrieval_ms = int((time.perf_counter() - started) * 1000)

        if not candidates:
            # Sin evidencia no se llama al LLM: se ahorra coste y se evita que
            # el modelo rellene el hueco con conocimiento general.
            context.answer = NO_CONTEXT_ANSWER
            context.grounded = False
            context.stop = True
            logger.info("rag.no_context", question=context.question[:80])

        logger.debug("rag.retrieved", n=len(candidates), ms=context.retrieval_ms)
        return context


class RerankStage(Stage):
    """Reordena los candidatos con un cross-encoder (bonus)."""

    name = "rerank"

    def __init__(self, reranker: CrossEncoderReranker, *, top_n: int) -> None:
        super().__init__()
        self._reranker = reranker
        self._top_n = top_n

    def process(self, context: RAGContext) -> RAGContext:
        # El reranker también puntúa contra la consulta reescrita: comparar el
        # pasaje con "¿y el plazo?" no discrimina nada.
        documents, elapsed_ms, applied = self._reranker.rerank(
            context.search_query or context.question, context.candidates, top_n=self._top_n
        )
        context.documents = documents
        context.rerank_ms = elapsed_ms
        context.reranked = applied
        return context


class PromptStage(Stage):
    """Ensambla el prompt final con el Builder."""

    name = "prompt"

    def __init__(self, *, context_max_chars: int, history_max_chars: int) -> None:
        super().__init__()
        self._context_max_chars = context_max_chars
        self._history_max_chars = history_max_chars

    def process(self, context: RAGContext) -> RAGContext:
        builder = (
            PromptBuilder(
                context_max_chars=self._context_max_chars,
                history_max_chars=self._history_max_chars,
            )
            .with_history(context.history)
            .with_context(context.documents)
            .with_question(context.question)
        )
        context.messages, context.citations = builder.build()
        return context


class GenerationStage(Stage):
    """Invoca al LLM y recoge la respuesta y el consumo de tokens."""

    name = "generation"

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__()
        self._llm = llm

    def process(self, context: RAGContext) -> RAGContext:
        started = time.perf_counter()
        response = self._llm.complete(context.messages)
        context.llm_ms = int((time.perf_counter() - started) * 1000)
        context.answer = response.content
        context.model = response.model
        context.prompt_tokens = response.prompt_tokens
        context.completion_tokens = response.completion_tokens

        # Heurística de "grounding": si el modelo no citó ninguna fuente y
        # además declaró no haber encontrado la información, la respuesta no
        # está respaldada por el corpus. Se registra para la analítica.
        lowered = context.answer.lower()
        cited = any(f"[{c.index}]" in context.answer for c in context.citations)
        context.grounded = cited or "no encontré" not in lowered

        logger.info(
            "rag.generated",
            model=context.model,
            ms=context.llm_ms,
            tokens=response.total_tokens,
            cited=cited,
        )
        return context


_SUGGEST_SYSTEM = """\
Propón preguntas de seguimiento que el usuario podría querer hacer a continuación.

REGLAS ESTRICTAS:
- SOLO preguntas cuya respuesta esté en el CONTEXTO proporcionado. Es preferible \
devolver menos preguntas que proponer una que el sistema no pueda responder.
- No repitas la pregunta que el usuario acaba de hacer ni lo ya respondido.
- Breves (máximo 12 palabras), en español, formuladas como las haría un usuario.
- Una por línea. Sin numeración, sin guiones, sin comillas, sin texto adicional.
- Si el contexto no da para ninguna pregunta útil, no devuelvas nada.
"""


class SuggestionStage(Stage):
    """Propone preguntas de seguimiento ancladas en el contexto recuperado.

    **Por qué ancladas y no libres.** Una sugerencia es una promesa implícita:
    el usuario pulsa el botón asumiendo que el sistema sabe la respuesta. Si se
    generan a partir de conocimiento general del modelo, la mitad llevan a
    "no encontré esa información" —peor que no ofrecer nada, porque el usuario
    ya invirtió un clic y una expectativa—. Por eso el prompt recibe los mismos
    fragmentos que se usaron para responder y se le prohíbe salirse de ellos.

    No se proponen sugerencias cuando la respuesta no quedó fundamentada: ahí
    no hay contexto del que derivarlas, y ofrecerlas sería inventar.

    Coste: una llamada corta al LLM (~100 tokens). Se mide por separado en
    `suggestions_ms` y se desactiva con `SUGGESTIONS_ENABLED=false`.
    """

    name = "suggestions"

    def __init__(self, llm: LLMProvider, *, count: int = 3, context_chars: int = 2500) -> None:
        super().__init__()
        self._llm = llm
        self._count = count
        self._context_chars = context_chars

    def process(self, context: RAGContext) -> RAGContext:
        if self._count <= 0 or not context.documents or not context.grounded:
            return context

        contexto = "\n\n".join(
            f"[{i}] {c.title}\n{c.text}" for i, c in enumerate(context.documents, start=1)
        )[: self._context_chars]

        started = time.perf_counter()
        try:
            respuesta = self._llm.complete(
                [
                    {"role": "system", "content": _SUGGEST_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"CONTEXTO:\n{contexto}\n\n"
                            f"El usuario preguntó: {context.question}\n\n"
                            f"Propón como máximo {self._count} preguntas de seguimiento."
                        ),
                    },
                ],
                max_tokens=120,
                temperature=0.3,
            )
        except Exception as exc:  # noqa: BLE001 - las sugerencias son un extra
            logger.warning("rag.suggestions_failed", error=str(exc))
            return context

        context.suggestions_ms = int((time.perf_counter() - started) * 1000)
        context.prompt_tokens += respuesta.prompt_tokens
        context.completion_tokens += respuesta.completion_tokens
        context.suggestions = _limpiar_sugerencias(
            respuesta.content, limite=self._count, pregunta_actual=context.question
        )
        logger.debug("rag.suggestions", n=len(context.suggestions), ms=context.suggestions_ms)
        return context


def _limpiar_sugerencias(bruto: str, *, limite: int, pregunta_actual: str) -> list[str]:
    """Normaliza la salida del modelo a una lista de preguntas presentables.

    El modelo tiende a numerar, poner viñetas o añadir una frase introductoria
    pese a que el prompt lo prohíbe; conviene sanear en vez de confiar.
    """
    vistas: set[str] = {pregunta_actual.strip().lower()}
    limpias: list[str] = []
    for linea in bruto.splitlines():
        texto = linea.strip().strip('"').strip("'")
        texto = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", texto).strip()
        if not (8 <= len(texto) <= 120) or "?" not in texto:
            continue
        clave = texto.lower()
        if clave in vistas:
            continue
        vistas.add(clave)
        limpias.append(texto)
        if len(limpias) >= limite:
            break
    return limpias


def build_chain(stages: list[Stage]) -> Stage:
    """Encadena las etapas y devuelve la cabeza de la cadena."""
    if not stages:
        raise ValueError("La cadena RAG necesita al menos una etapa")
    for current, following in zip(stages, stages[1:], strict=False):
        current.set_next(following)
    return stages[0]
