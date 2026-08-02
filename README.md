# Asistente RAG — BBVA Colombia

Sistema RAG (Retrieval-Augmented Generation) que responde preguntas sobre el contenido
publicado en **https://www.bbva.com.co/**, extraído mediante web scraping, indexado en una
base de datos vectorial y consultable desde una interfaz conversacional con memoria.

Todo el stack se levanta con **un solo comando**.

```
┌─────────┐   scraping    ┌──────────┐   chunking    ┌────────┐
│ bbva.   │──────────────▶│  data/   │──────────────▶│ Qdrant │
│ com.co  │   Playwright  │ raw+clean│   embeddings  │        │
└─────────┘   + Chrome    └──────────┘   fastembed   └────┬───┘
                                                          │ top-20
┌──────────┐      ┌──────────┐      ┌──────────────┐      │
│ Streamlit│─────▶│ FastAPI  │─────▶│ reranker     │◀─────┘
│    CLI   │ HTTP │ RAGEngine│      │ cross-encoder│  top-5
└──────────┘      └────┬─────┘      └──────┬───────┘
                       │                   ▼
                  ┌────▼─────┐        ┌─────────┐
                  │PostgreSQL│        │   LLM   │
                  │ historial│        │ OpenAI/ │
                  │+analítica│        │ Ollama  │
                  └──────────┘        └─────────┘
```

---

## Tabla de contenidos

1. [Requisitos previos](#1-requisitos-previos)
2. [Puesta en marcha desde cero](#2-puesta-en-marcha-desde-cero)
3. [Cómo usar el sistema](#3-cómo-usar-el-sistema)
4. [Analítica del histórico de conversaciones](#4-analítica-del-histórico-de-conversaciones)
5. [Patrones de diseño](#5-patrones-de-diseño)
6. [Stack tecnológico y justificación](#6-stack-tecnológico-y-justificación)
7. [Arquitectura del proyecto](#7-arquitectura-del-proyecto)
8. [Configuración completa](#8-configuración-completa)
9. [Decisiones de diseño relevantes](#9-decisiones-de-diseño-relevantes)
10. [Limitaciones conocidas](#10-limitaciones-conocidas)
11. [Futuras mejoras](#11-futuras-mejoras)
12. [Tests](#12-tests)

---

## 1. Requisitos previos

| Requisito | Versión | Notas |
|---|---|---|
| **Docker** | 24+ | Con el plugin `docker compose` v2 |
| **RAM libre** | ~4 GB | Qdrant + PostgreSQL + API con modelos ONNX cargados |
| **Disco** | ~5 GB | Imágenes (la de la API incluye Google Chrome) + modelos + índice |
| **Conexión a internet** | — | Para el scraping y para descargar los modelos la primera vez |

**Variables de entorno.** Todas viven en `.env` (se crea copiando `.env.example`). La única
obligatoria es `OPENAI_API_KEY`, y aun esa es evitable: con `LLM_PROVIDER=ollama` el sistema
funciona sin ninguna API de pago (ver [§3.5](#35-ejecución-sin-apis-de-pago)).

> No hace falta instalar Python, Playwright ni ningún modelo en la máquina anfitriona.
> Todo ocurre dentro de los contenedores.

---

## 2. Puesta en marcha desde cero

### Paso 1 — Clonar el repositorio

```bash
git clone <URL-DEL-REPOSITORIO>
cd bbva-rag-assistant
```

### Paso 2 — Configurar el entorno

```bash
cp .env.example .env
```

Edita `.env` y pon tu clave de OpenAI:

```dotenv
OPENAI_API_KEY=sk-...
```

Todo lo demás trae valores por defecto que funcionan. `.env` está en `.gitignore`: las
credenciales nunca se versionan.

### Paso 3 — Levantar el stack

```bash
docker compose up -d --build
```

Esto construye las imágenes y arranca cuatro servicios:

| Servicio | Puerto | Qué es |
|---|---|---|
| `qdrant` | 6333 | Base de datos vectorial (self-hosted) |
| `postgres` | 5432 | Historial de conversaciones y analítica |
| `api` | 8000 | FastAPI + motor RAG |
| `ui` | 8501 | Interfaz web conversacional |

La primera construcción tarda varios minutos (instala Google Chrome en la imagen de la API).
Comprueba que todo esté sano:

```bash
docker compose ps
curl http://localhost:8000/health
```

En este punto el estado será `degraded`: los servicios están arriba pero el índice está vacío.

### Paso 4 — Ingestar el sitio

```bash
docker compose run --rm ingest
```

Este comando hace las dos mitades del trabajo:

1. **Scraping** — rastrea `https://www.bbva.com.co/` en anchura hasta `SCRAPER_MAX_PAGES`
   páginas, guardando el **HTML crudo** en `data/raw/` y el **texto limpio** en `data/clean/`.
2. **Indexación** — trocea, vectoriza con un modelo local y escribe los puntos en Qdrant.

Tarda entre 5 y 15 minutos según `SCRAPER_MAX_PAGES` (120 por defecto). La primera vez
descarga además los modelos de embeddings y reranking (~500 MB, quedan cacheados en un
volumen).

Para una prueba rápida:

```bash
docker compose run --rm ingest rag-assistant ingest --max-pages 25
```

### Paso 5 — Usar el sistema

```bash
curl http://localhost:8000/health     # debe devolver "ok" e indexed_chunks > 0
```

Abre **http://localhost:8501** y pregunta.

---

## 3. Cómo usar el sistema

### 3.1 Interfaz web — http://localhost:8501

La vista **Chat** mantiene la conversación con memoria. Cada respuesta muestra:

- las **fuentes citadas** con enlace a la página original y su puntuación de relevancia;
- la **telemetría**: latencia total y por etapa, tokens consumidos, cuántos mensajes de
  historial se usaron, y si la respuesta quedó fundamentada en el corpus;
- botones **👍 / 👎** cuya valoración alimenta la métrica de satisfacción.

En la barra lateral puedes fijar un `conversation_id` para **retomar una conversación
anterior**, o empezar una nueva.

La vista **Analítica** es el panel descrito en [§4](#4-analítica-del-histórico-de-conversaciones).

Preguntas de ejemplo que funcionan bien con el corpus indexado:

```
¿Qué tipos de cuenta de ahorro ofrece BBVA Colombia?
¿Hasta qué porcentaje financia BBVA un crédito de vivienda?
¿Qué plazos tiene el crédito de vehículo?
¿Qué es un CDT y qué modalidades hay?
```

Y para comprobar la memoria conversacional, encadena:

```
Tú  › ¿Qué créditos de vivienda ofrecen?
Bot › [respuesta con las modalidades]
Tú  › ¿Y cuál es el plazo máximo del primero?     ← "el primero" se resuelve con el historial
```

### 3.2 API REST — http://localhost:8000/docs

```bash
# Primera pregunta (el sistema crea la conversación)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué tipos de cuenta de ahorro ofrece BBVA?",
       "conversation_id": "demo-001"}'

# Segunda pregunta en el MISMO hilo: recuerda el contexto
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Y cuál de esas no cobra cuota de manejo?",
       "conversation_id": "demo-001"}'
```

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/chat` | Enviar un mensaje. Mantiene el hilo por `conversation_id`. |
| `GET` | `/chat/conversations` | Listar conversaciones |
| `GET` | `/chat/conversations/{id}` | Historial completo de una conversación |
| `DELETE` | `/chat/conversations/{id}` | Eliminar una conversación |
| `POST` | `/chat/feedback` | Valorar una respuesta (`1` / `-1`) |
| `GET` | `/analytics` | Informe de métricas de impacto (`?days=30`) |
| `GET` | `/analytics/live` | Contadores del proceso en vivo |
| `GET` | `/health` | Estado de las dependencias |
| `GET` | `/config` | Configuración efectiva (sin secretos) |
| `POST` | `/ingest` | Lanzar la ingesta en segundo plano |

Respuesta de `/chat`:

```json
{
  "answer": "BBVA Colombia ofrece varias cuentas de ahorro [1][2]...",
  "conversation_id": "demo-001",
  "message_id": 42,
  "sources": [
    {"index": 1, "url": "https://www.bbva.com.co/personas/productos/cuentas/ahorro.html",
     "title": "Cuentas de ahorro", "score": 0.8734}
  ],
  "grounded": true, "reranked": true, "history_used": 2,
  "latency_ms": 2140, "retrieval_ms": 38, "rerank_ms": 210, "llm_ms": 1880,
  "prompt_tokens": 1834, "completion_tokens": 156, "model": "gpt-4o-mini"
}
```

### 3.3 CLI

```bash
docker compose run --rm ingest rag-assistant chat          # chat interactivo
docker compose run --rm ingest rag-assistant chat -q "¿Qué es un CDT?"
docker compose run --rm ingest rag-assistant analytics     # informe de métricas
docker compose run --rm ingest rag-assistant conversations # listar histórico
docker compose run --rm ingest rag-assistant health        # estado del sistema
```

### 3.4 Atajos con Make

```bash
make up         # levantar todo
make ingest     # scraping + indexación
make chat       # chat por consola
make analytics  # informe de métricas
make reindex    # reindexar sin volver a rastrear el sitio
make down       # parar
make clean      # parar y BORRAR volúmenes (índice e historial)
```

### 3.5 Ejecución sin APIs de pago

El enunciado valora positivamente las herramientas sin costo. Embeddings, reranking y base
vectorial **ya son locales y gratuitos**. Para que también lo sea el LLM:

```bash
docker compose --profile ollama up -d
docker compose exec ollama ollama pull llama3.2:3b
```

y en `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

```bash
docker compose restart api
```

A partir de ahí el sistema completo funciona sin ninguna llamada de pago. La calidad de
redacción baja respecto a `gpt-4o-mini`, pero la recuperación —que es la parte que
determina si la respuesta es correcta— es idéntica.

---

## 4. Analítica del histórico de conversaciones

> Requisito: *"una funcionalidad que me permita recorrer el histórico de conversaciones para
> extraer métricas y valores de impacto"*.

Disponible en tres formas: pestaña **Analítica** de la UI, `GET /analytics` y
`rag-assistant analytics`.

Cada métrica está elegida por la **decisión que permite tomar**, no por ser fácil de calcular:

**1 · Volumen y adopción** — ¿se está usando?
`total_conversations`, `total_messages`, `unique_users`, `avg_messages_per_conversation`,
`multi_turn_conversation_rate`, serie diaria.
La *tasa multi-turno* mide qué fracción de conversaciones tuvo más de un intercambio: es la
medida directa de si la memoria conversacional aporta valor o el usuario pregunta y se va.

**2 · Calidad** — ¿sirve?
`grounded_rate`, `no_answer_rate`, `error_rate`, `avg_top_score`, `satisfaction_rate` y
**`unanswered_questions`**.
Esta última es la salida más accionable de todo el informe: la lista literal de preguntas
que el corpus no supo responder. Cada una es una página que falta por indexar o un contenido
que falta por publicar.

**3 · Rendimiento y coste** — ¿es sostenible?
`avg_latency_ms`, `p50/p95`, y el desglose por etapa (`retrieval` / `rerank` / `llm`), más
tokens y `estimated_cost_usd` por conversación.
El desglose por etapa es lo que dice **dónde** optimizar: si el 90 % de la latencia es el LLM,
tocar el reranker no sirve de nada.

**4 · Contenido** — ¿qué le importa a la gente?
`top_topics` (términos más consultados, sin palabras vacías) y `top_cited_pages` (qué páginas
del sitio están respondiendo de verdad).

Las agregaciones se resuelven en SQL, no en Python, para que el coste no crezca con el
histórico.

---

## 5. Patrones de diseño

Se implementan **siete**. Cada uno resuelve un problema concreto de este sistema; ninguno
está para rellenar el requisito.

### 5.1 Strategy

📍 `scraping/fetchers/`, `indexing/chunking/`, `indexing/embeddings/`, `rag/llm/`

Cuatro puntos del sistema son decisiones reversibles que conviene poder cambiar sin tocar
código: cómo se descarga una página, cómo se trocea el texto, qué modelo vectoriza y qué LLM
redacta. Cada uno vive tras una interfaz con varias implementaciones intercambiables por
variable de entorno.

| Interfaz | Implementaciones | Se elige con |
|---|---|---|
| `Fetcher` | `HttpFetcher`, `BrowserFetcher` | `SCRAPER_FETCHER` |
| `ChunkingStrategy` | `RecursiveChunker`, `FixedChunker` | `CHUNK_STRATEGY` |
| `EmbeddingProvider` | `FastEmbedProvider` | `EMBEDDING_PROVIDER` |
| `LLMProvider` | `OpenAIProvider`, `OllamaProvider` | `LLM_PROVIDER` |

**Por qué aquí:** es lo que permite pasar de OpenAI a un modelo local sin tocar el motor RAG,
y lo que hizo posible resolver el bloqueo anti-bot de BBVA cambiando de estrategia de
descarga en vez de reescribir el crawler.

### 5.2 Factory Method

📍 `*/factory.py` (cinco fábricas)

Cada familia de estrategias tiene su fábrica con un registro `nombre → clase`. El resto del
sistema pide *"un fetcher"*, *"un LLM"*, y recibe uno ya construido y configurado.

**Por qué aquí:** concentra en un punto la traducción de configuración a objeto, y da errores
útiles (`"Proveedor de LLM desconocido: 'gpt'. Opciones válidas: ollama, openai"`) en lugar
de un `KeyError` a mitad de la ingesta. `register_fetcher()` permite además añadir
implementaciones desde fuera sin tocar el paquete.

### 5.3 Adapter

📍 `indexing/vector_store/qdrant_store.py`

`VectorStore` define el vocabulario que necesita nuestro dominio (`upsert`, `search`,
`ensure_collection`); `QdrantVectorStore` lo traduce a la API de `qdrant_client`, incluyendo
detalles que no deben filtrarse hacia arriba (Qdrant exige IDs UUID, así que se deriva uno
estable del ID del chunk).

**Por qué aquí:** el motor RAG **nunca importa `qdrant_client`**. Migrar a pgvector o Milvus
es escribir otro adaptador, no reescribir el sistema. Y en los tests se sustituye por un
índice en memoria.

### 5.4 Repository

📍 `conversation/repository.py`

El motor necesita *"los últimos N mensajes de la conversación X"* y *"guarda este turno"*. No
necesita saber que existe SQLAlchemy, ni sesiones, ni transacciones.

Dos implementaciones: `SqlConversationRepository` (PostgreSQL/SQLite) e
`InMemoryConversationRepository`.

**Por qué aquí:** toda la lógica transaccional queda confinada a un módulo, y los tests del
motor RAG corren **sin base de datos**. Cambiar de SQLite a PostgreSQL fue cambiar una URL.

### 5.5 Chain of Responsibility

📍 `rag/stages.py`

Responder es una secuencia:
`QueryRewriteStage → RetrievalStage → RerankStage → PromptStage → GenerationStage`.
Cada eslabón enriquece un `RAGContext` compartido y decide si pasa el testigo.

**Por qué aquí, y no un método largo:**

- **Composición** — activar el reranker es añadir o quitar un eslabón (`RERANKER_ENABLED`),
  no meter un `if` dentro de un método de 200 líneas.
- **Cortocircuito** — si la recuperación no encuentra nada relevante, `RetrievalStage` corta
  la cadena y devuelve una respuesta honesta **sin gastar una llamada al LLM**. Eso ahorra
  dinero y, sobre todo, evita que el modelo rellene el hueco con conocimiento general —el
  fallo más grave posible en un asistente bancario.
- **Observabilidad** — cada etapa mide su propio tiempo, lo que produce el desglose
  `retrieval / rerank / llm` que consume la analítica.

Y la justificación que no es teórica: al probar el sistema apareció un fallo real de
recuperación en los seguimientos ([§9.8](#98-la-búsqueda-necesitaba-reescribir-la-consulta-no-solo-recordarla)).
Corregirlo consistió en **añadir un eslabón** al principio de la cadena. Recuperación,
reranking, prompt y generación no se tocaron.

### 5.6 Builder

📍 `rag/prompt_builder.py`

```python
messages, citations = (
    PromptBuilder(context_max_chars=8000, history_max_chars=4000)
    .with_history(history)
    .with_context(documents)
    .with_question(question)
    .build()
)
```

**Por qué aquí:** el prompt se compone de piezas heterogéneas con presupuestos de caracteres
independientes y reglas de recorte propias. Con f-strings dispersos por el motor es imposible
razonar sobre qué entra de verdad en la ventana de contexto. El Builder además devuelve las
citas ya numeradas y alineadas con los marcadores `[n]` del prompt, de modo que la respuesta
y sus fuentes no pueden desincronizarse.

### 5.7 Observer

📍 `analytics/observers.py`

El motor publica un `QueryEvent` por consulta; los suscriptores reaccionan
(`LoggingObserver`, `InMemoryMetricsObserver`).

**Por qué aquí:** cada consulta interesa a varios destinos y esa lista va a crecer
(Prometheus, un data warehouse). Meterlos dentro del motor lo acoplaría a todos ellos.
Además, **un observador que falla nunca rompe la respuesta al usuario**: los errores de los
suscriptores se capturan y se registran.

### 5.8 Singleton *(complementario)*

📍 `config/settings.py` — `get_settings()` con `lru_cache(maxsize=1)`.

Garantiza que scraper, indexador, motor, API y CLI observen exactamente la misma
configuración validada. Se usa `lru_cache` en lugar de una metaclase porque es idiomático,
thread-safe y permite `cache_clear()` en tests, cosa que un Singleton rígido dificulta.
El mismo mecanismo cachea el modelo de embeddings y el cliente de Qdrant, que son caros de
construir.

### 5.9 Facade *(complementario)*

📍 `rag/engine.py` — `RAGEngine.ask()` coordina memoria, embeddings, base vectorial,
reranker, prompt, LLM, persistencia y telemetría tras **un solo método**. API, CLI y UI usan
esa misma fachada, así que las tres comparten exactamente el mismo comportamiento.

---

## 6. Stack tecnológico y justificación

| Pieza | Elección | Por qué esta y no otra |
|---|---|---|
| **Lenguaje** | Python 3.12 | Requisito del enunciado. |
| **Scraping** | Playwright + Google Chrome | BBVA devuelve **403** a `curl`/`httpx`. Se necesita un navegador real (ver [§9.1](#91-el-sitio-bloquea-a-los-clientes-http-y-también-al-chromium-de-playwright)). |
| **Limpieza** | trafilatura + BeautifulSoup | Se ejecutan **los dos** y se elige el más informativo por página (ver [§9.2](#92-un-solo-extractor-de-texto-no-basta)). |
| **Embeddings** | fastembed (ONNX), `paraphrase-multilingual-MiniLM-L12-v2` | Local y **gratuito**. Frente a `sentence-transformers`: evita PyTorch y baja la imagen de ~3 GB a <1 GB. Es la librería oficial de Qdrant. Multilingüe, entrenado en 50+ idiomas: el corpus es español. |
| **Base vectorial** | Qdrant 1.12 (self-hosted) | Apache-2.0, **un solo contenedor**, sin cuenta ni cuota. Filtrado por payload nativo e índices sobre metadatos. Alternativas descartadas: Pinecone (SaaS de pago), Chroma (menos maduro en concurrencia), pgvector (habría ahorrado un servicio, pero peor rendimiento y ergonomía en búsqueda vectorial pura). |
| **Reranker** | `Xenova/ms-marco-MiniLM-L-6-v2` vía fastembed | Cross-encoder ligero, local y gratuito. Aporta la mejora de precisión del bonus sin añadir dependencias ni coste. |
| **LLM** | OpenAI `gpt-4o-mini` · alternativa Ollama | En RAG el modelo no necesita "saber", solo redactar fielmente a partir del contexto: un modelo pequeño y barato es la elección correcta. Ollama cubre el caso de coste cero. |
| **API** | FastAPI + Uvicorn | Validación con Pydantic (los mismos modelos que la configuración), OpenAPI automático, async nativo. |
| **UI** | Streamlit | El enunciado pide *"funcional y limpia, no bonita"*. Streamlit da un chat con estado y un panel de métricas en un fichero, sin frontend que mantener. |
| **Historial** | PostgreSQL 16 + SQLAlchemy 2.0 | El requisito de analítica es agregación sobre el histórico: eso es SQL. SQLAlchemy permite además correr sobre SQLite en local sin infraestructura. |
| **CLI** | Typer + Rich | Tipado, ayuda automática y salida legible. |
| **Config** | pydantic-settings | Valida al arrancar. Un `.env` incoherente falla al inicio con un mensaje claro, no a mitad de una ingesta de 15 minutos. |
| **Logging** | structlog | Consola en desarrollo, JSON en producción sin cambiar código. |
| **Tests** | pytest | 83 tests sin dependencias externas. |

---

## 7. Arquitectura del proyecto

```
bbva-rag-assistant/
├── docker-compose.yml           # 4 servicios + 2 perfiles opcionales
├── docker/
│   ├── Dockerfile.app           # API + CLI (multi-etapa, incluye Google Chrome)
│   └── Dockerfile.ui            # Streamlit (imagen ligera aparte)
├── .env.example                 # TODA la configuración, documentada
├── Makefile
├── data/
│   ├── raw/                     # HTML crudo + _manifest.jsonl
│   └── clean/                   # JSON limpio + corpus.jsonl
├── src/rag_assistant/
│   ├── config/settings.py       # Singleton de configuración
│   ├── core/                    # excepciones, logging, modelos de dominio
│   ├── scraping/
│   │   ├── fetchers/            # Strategy: http | browser  (+ Factory)
│   │   ├── crawler.py           # BFS con robots, dedupe, límites
│   │   ├── cleaner.py           # HTML → texto limpio
│   │   └── storage.py           # persistencia crudos + limpios
│   ├── indexing/
│   │   ├── chunking/            # Strategy: recursive | fixed  (+ Factory)
│   │   ├── embeddings/          # Strategy: fastembed  (+ Factory)
│   │   ├── vector_store/        # Adapter sobre Qdrant  (+ Factory)
│   │   └── pipeline.py
│   ├── rag/
│   │   ├── llm/                 # Strategy: openai | ollama  (+ Factory)
│   │   ├── reranker.py          # cross-encoder (bonus)
│   │   ├── prompt_builder.py    # Builder
│   │   ├── stages.py            # Chain of Responsibility
│   │   └── engine.py            # Facade
│   ├── conversation/            # Repository + memoria de N mensajes
│   ├── analytics/               # Observer + métricas de impacto
│   ├── api/                     # FastAPI
│   ├── ui/streamlit_app.py
│   ├── cli.py
│   └── pipelines.py             # orquestación scraping + indexación
└── tests/                       # 83 tests, sin dependencias externas
```

**Regla de dependencias:** `core` no depende de nada; `scraping`, `indexing` y `conversation`
dependen solo de `core` y `config`; `rag` los orquesta; `api`, `ui` y `cli` son solo capas de
entrega. Ningún módulo de dominio importa FastAPI, `qdrant_client` ni el SDK de OpenAI
directamente.

---

## 8. Configuración completa

Todo se controla desde `.env`. Los parámetros que más afectan al comportamiento:

| Variable | Por defecto | Qué hace |
|---|---|---|
| **`HISTORY_WINDOW_SIZE`** | `6` | **N mensajes previos que recuerda el sistema.** Cuenta mensajes, no turnos: 6 ≈ los 3 últimos intercambios. |
| `QUERY_REWRITE_ENABLED` | `true` | Reconstruye la consulta con el historial antes de buscar ([§9.8](#98-la-búsqueda-necesitaba-reescribir-la-consulta-no-solo-recordarla)). |
| `ONNX_THREADS` | `0` (auto) | Hilos por modelo ONNX. `0` = mitad de los núcleos, medido como óptimo ([§9.7](#97-usar-todos-los-núcleos-era-22-más-lento-que-usar-la-mitad)). |
| `LLM_PROVIDER` / `LLM_MODEL` | `openai` / `gpt-4o-mini` | Proveedor y modelo. |
| `LLM_TEMPERATURE` | `0.1` | Baja a propósito: se quiere fidelidad al contexto, no creatividad. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Tamaño y solapamiento de los fragmentos. |
| `CHUNK_STRATEGY` | `recursive` | `recursive` \| `fixed`. |
| `RETRIEVAL_TOP_K` | `20` | Candidatos que devuelve la búsqueda vectorial. |
| `RERANKER_ENABLED` | `true` | Activa el cross-encoder (bonus). |
| `RERANKER_TOP_N` | `5` | Documentos finales que llegan al LLM. |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.30` | Por debajo se descarta: mejor no responder que responder mal. |
| `CONTEXT_MAX_CHARS` | `8000` | Presupuesto de contexto en el prompt. |
| `SCRAPER_BASE_URL` | `https://www.bbva.com.co/` | Sitio objetivo. |
| `SCRAPER_FETCHER` | `browser` | `browser` (Playwright) \| `http` (httpx). |
| `SCRAPER_BROWSER_CHANNEL` | `chrome` | Google Chrome estable; vacío = Chromium de Playwright. |
| `SCRAPER_MAX_PAGES` / `_MAX_DEPTH` | `120` / `3` | Alcance del crawl. |
| `SCRAPER_REQUEST_DELAY` | `0.5` | Segundos entre peticiones (cortesía). |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de vectorización. |

Ver `.env.example` para la lista completa, agrupada y comentada.

### Cambiar de banco objetivo

El enunciado permite usar otro banco. Basta con cambiar tres variables:

```dotenv
SCRAPER_BASE_URL=https://www.bancodebogota.com/personas
SCRAPER_ALLOWED_DOMAINS=www.bancodebogota.com
SCRAPER_FETCHER=http          # este sitio no bloquea clientes HTTP: es mucho más rápido
```

```bash
docker compose run --rm ingest rag-assistant ingest --fresh
```

No hay una sola línea de código específica de BBVA en el crawler.

---

## 9. Decisiones de diseño relevantes

### 9.1 El sitio bloquea a los clientes HTTP, y también al Chromium de Playwright

`https://www.bbva.com.co/` está detrás de una protección anti-bot (Akamai) que devuelve **403**
con una página `Customdeny` a cualquier cliente HTTP plano — incluido `/robots.txt`.

Al montar el scraper con Playwright el bloqueo **seguía**. Se midieron cinco configuraciones
contra el sitio real:

| Configuración | Resultado |
|---|---|
| `curl` / `httpx` | 403 bloqueado |
| Chromium de Playwright, headless | **403 bloqueado** |
| Chromium de Playwright, headless + parches de stealth | **403 bloqueado** |
| Chromium de Playwright, con ventana | 200 OK |
| **Google Chrome estable, headless** | **200 OK** ← el que se usa |
| Google Chrome estable, con ventana | 200 OK |

La conclusión es contraintuitiva y determina la implementación: **el bloqueo no depende de los
parches de JavaScript sino del binario**. El Chromium que empaqueta Playwright tiene una huella
distinguible (códecs, `userAgentData`, fingerprint TLS/HTTP2). Google Chrome estable en modo
headless nuevo pasa el filtro **sin necesidad de Xvfb ni ventana gráfica**, lo que hace la
solución viable dentro de Docker.

Por eso `docker/Dockerfile.app` instala Google Chrome desde el repositorio oficial y el fetcher
lanza con `channel="chrome"`, degradando al Chromium empaquetado con un aviso explícito si
Chrome no estuviera disponible.

**Sobre `robots.txt`:** el sitio también lo sirve con 403, así que no hay reglas que parsear.
Ante esa ausencia el crawler **no** asume barra libre: mantiene el retardo entre peticiones,
el filtro de dominio, el tope de páginas y la lista de exclusión, y deja constancia en el log.
Solo se rastrea contenido público e institucional; nada tras autenticación.

### 9.2 Un solo extractor de texto no basta

Medido sobre páginas de producto reales de BBVA:

| Página | trafilatura | extractor por DOM |
|---|---|---|
| `prestamos/vivienda.html` | 371 chars | **6.876 chars** |
| `prestamos/online.html` | 398 chars | **10.430 chars** |
| `personas/aviso-legal.html` | **2.190 chars** | 1.271 chars |

Las páginas de producto son maquetación por componentes, no prosa: el algoritmo de densidad de
texto de trafilatura las descarta y se pierde justo lo que el usuario pregunta (plazos,
porcentajes de financiación, requisitos). En las páginas editoriales ocurre lo contrario.

Por eso `cleaner.py` **ejecuta ambos extractores y elige el más informativo por página**. Es
barato y evita perder contenido por casarse con una librería.

### 9.3 Guardar el HTML crudo no es redundante

`data/raw/` conserva el HTML íntegro además del texto limpio. Eso permite **reprocesar la
limpieza sin volver a rastrear el sitio**:

```bash
make reindex   # re-limpia desde el HTML guardado y reindexa, en segundos
```

Durante el desarrollo esto convirtió cada iteración sobre la calidad del extractor en
segundos en vez de en un crawl de 15 minutos, y evita molestar de nuevo al servidor de BBVA.

### 9.4 Sin evidencia, no se llama al LLM

Si la búsqueda vectorial no devuelve nada por encima de `RETRIEVAL_SCORE_THRESHOLD`, la cadena
se corta y se responde *"no encontré esa información"* **sin invocar al modelo**. Ahorra coste,
pero sobre todo elimina de raíz el modo de fallo más grave en un asistente bancario: que el
modelo complete el hueco con conocimiento general y suene igual de convincente que cuando
acierta.

En la misma línea, `LLM_TEMPERATURE=0.1` y un prompt que prohíbe explícitamente inventar
cifras, tasas y requisitos.

### 9.5 Ventana deslizante, no resumen del historial

`HISTORY_WINDOW_SIZE` recorta a los N mensajes más recientes. Se descartó resumir el historial
con el LLM: la ventana es determinista, no cuesta tokens ni latencia extra y **no puede
alucinar sobre lo que el usuario dijo antes**. El resumen progresivo queda como mejora futura
para sesiones muy largas.

Detalle fino: si la ventana empieza en una respuesta huérfana (sin su pregunta), se descarta
esa respuesta. Un turno cortado a la mitad confunde al modelo más de lo que le ayuda.

### 9.6 El reranker degrada con elegancia

Si el cross-encoder no puede cargarse (sin red en el primer arranque, disco lleno), se registra
el aviso y se continúa con el orden vectorial. Un reranker caído nunca debe dejar el chat sin
servicio: la respuesta será algo peor, no inexistente.

### 9.7 Usar todos los núcleos era 2,2× más lento que usar la mitad

Midiendo el sistema bajo carga apareció que `rag-api` consumía **862% de CPU de media y
1298% en pico** en una máquina de 12 núcleos, y que el reranking pasaba de 3,7 s con una
petición a 32 s con veinte —mientras el tiempo de LLM se mantenía plano en 1,9 s, porque
está en la red y no compite—.

La causa no era "falta de CPU". Aislando el cross-encoder en un microbenchmark
(`scripts/micro_threads.py`, 20 pasajes reales, mediana de 5 ejecuciones):

| Hilos ONNX | top_k=20 | top_k=12 | top_k=5 |
|---|---|---|---|
| `auto` (todos, default de ONNX) | 2.100 ms | 901 ms | 706 ms |
| 4 | 1.166 ms | 682 ms | 251 ms |
| **6 (mitad de los núcleos)** | **955 ms** | **522 ms** | 227 ms |
| 12 (explícito) | 1.329 ms | 829 ms | 458 ms |

El lote es pequeño, así que repartirlo entre 12 hilos cuesta más en sincronización de lo
que ahorra en cómputo: **el valor por defecto de ONNX era el peor de todos los probados**.
Y dejar núcleos libres es además lo que permite que dos peticiones concurrentes avancen en
paralelo en vez de pelearse.

Por eso `ONNX_THREADS=0` (automático) resuelve a la mitad de los núcleos disponibles.
Es un caso claro de por qué conviene medir antes de "optimizar": la intuición decía
*"dale todos los núcleos"* y la intuición estaba equivocada.

### 9.8 La búsqueda necesitaba reescribir la consulta, no solo recordarla

Inyectar el historial en el prompt hace que el modelo *redacte* bien los seguimientos, pero
la búsqueda vectorial seguía usando la pregunta literal. El fallo se reprodujo así:

```
Tú  › ¿Qué créditos de vivienda ofrecen?
Bot › [correcto: modalidades de vivienda]
Tú  › ¿Cuál es el plazo máximo de ese producto?
Bot › El plazo máximo para los CDT es de 36 meses.      ← producto equivocado
```

Siete palabras, ninguna del dominio: el índice devolvió fragmentos sobre CDT y el modelo
respondió con fidelidad… a un contexto que no era el pedido. Es el peor tipo de error,
porque llega con fuentes citadas y suena seguro.

La solución es `QueryRewriteStage`: reconstruye el sujeto con el historial **antes** de
tocar el índice. La respuesta se sigue redactando sobre la pregunta original; lo reescrito
solo alimenta la búsqueda y el reranking, y se expone en la respuesta (`search_query`) para
que la recuperación sea depurable. Si la reescritura falla o degenera, se continúa con la
pregunta original: es una mejora, nunca un punto de caída.

Merece la pena señalar qué costó añadirla: **un eslabón nuevo en la cadena**. Ni la
recuperación, ni el reranking, ni la generación se enteraron. Eso es lo que se compra con
el patrón Chain of Responsibility ([§5.5](#55-chain-of-responsibility)), y es la
justificación práctica de haberlo elegido.

### 9.9 IDs de chunk deterministas

El ID se deriva de `url + índice + hash del texto`. Reindexar el mismo contenido **actualiza**
el punto en vez de duplicarlo, así que la ingesta es idempotente y se puede repetir sin
ensuciar el índice.

---

## 10. Limitaciones conocidas

Enumeradas con honestidad, como pide el enunciado.

1. **Contenido tras JavaScript pesado o autenticación.** Se indexa el sitio público. Los
   simuladores (crédito, hipoteca) y todo lo que hay tras el login quedan fuera. Preguntas
   sobre tasas exactas de un caso concreto no tendrán respuesta.

2. **El scraping depende de una condición externa que puede cambiar.** BBVA puede endurecer su
   protección anti-bot en cualquier momento y dejar fuera también a Chrome headless. Si ocurre,
   el sistema lo detecta y lo reporta (`blocked` en el informe de crawl, con mensaje
   explicativo) en lugar de fallar en silencio, y el enunciado permite apuntar a otro banco
   cambiando tres variables ([§8](#cambiar-de-banco-objetivo)).

3. **Sin búsqueda híbrida.** La recuperación es densa pura. Una búsqueda léxica (BM25/sparse)
   combinada ayudaría con nombres propios de producto y números exactos, donde los embeddings
   son flojos. Qdrant lo soporta de forma nativa; no se implementó por alcance.

4. **Cobertura del crawl acotada.** `SCRAPER_MAX_PAGES=120` y profundidad 3 por defecto. El
   sitio completo es mayor. Es un límite de tiempo de ejecución, no técnico: subir el valor
   funciona, solo tarda más.

5. **El esquema se crea con `create_all`, sin migraciones.** Para un proyecto de este alcance
   Alembic añadía ceremonia sin valor. En un sistema con vida en producción sería necesario.

6. **Sin autenticación en la API.** El enunciado plantea usuarios internos y el despliegue es
   local. En un entorno real haría falta autenticación, rate limiting por usuario y CORS
   restringido.

7. **`estimated_cost_usd` es una estimación.** Usa una tabla de precios embebida en el código.
   Si se usa un modelo que no está en la tabla, el coste se reporta como 0. No sustituye a la
   facturación real del proveedor.

8. **La detección de "respuesta fundamentada" es heurística.** Se basa en si el modelo citó
   fuentes y en marcadores textuales. Es suficiente para una métrica agregada de tendencia,
   pero no es una verificación formal de que cada afirmación esté respaldada.

9. **Sin evaluación cuantitativa de la calidad del RAG.** No hay un set de preguntas
   etiquetado ni métricas tipo *recall@k* o *faithfulness*. La calidad se validó con una
   batería E2E contra el stack real, pero eso comprueba que el sistema funciona, no cuán
   bien responde. Es la primera cosa que añadiría (ver siguiente sección).

10. **Las preguntas meta sobre la propia conversación no se responden.** *"Resume lo que te
    acabo de preguntar"* devuelve *"no encontré información"*, porque la etapa de
    recuperación no halla contexto en el corpus y corta la cadena antes de llamar al LLM
    ([§9.4](#94-sin-evidencia-no-se-llama-al-llm)). Es el precio de esa decisión, y se
    asumió a conciencia: para un asistente bancario, garantizar que nunca se responde sin
    evidencia vale más que atender preguntas conversacionales. Se resuelve clasificando la
    intención antes de recuperar, y está anotado como mejora.

11. **La reescritura de consulta añade una llamada al LLM por turno de seguimiento.**
    Resolver la limitación descrita en [§9.8](#98-la-búsqueda-necesitaba-reescribir-la-consulta-no-solo-recordarla)
    cuesta ~700–1.300 ms y ~150 tokens extra cuando hay historial. Es un intercambio
    consciente: sin ella, los seguimientos recuperaban el producto equivocado. Se desactiva
    con `QUERY_REWRITE_ENABLED=false`. Una versión sin coste sería clasificar primero si la
    pregunta es autocontenida (con una heurística o un modelo diminuto) y reescribir solo
    cuando haga falta.

12. **El rendimiento bajo concurrencia sigue limitado por CPU.** Acotar los hilos de ONNX
    ([§9.7](#97-usar-todos-los-núcleos-era-22-más-lento-que-usar-la-mitad)) mejoró mucho las
    cosas, pero el reranking sigue siendo inferencia en CPU y es la etapa dominante. Para
    concurrencia alta de verdad haría falta GPU, un reranker más pequeño, o mover el
    reranking a un servicio aparte que se escale por su cuenta.

---

## 11. Futuras mejoras

Por orden de valor por esfuerzo:

1. **Set de evaluación y métricas de RAG.** 40–60 preguntas con respuesta esperada y fuente
   correcta, y medir *recall@k*, *MRR* y fidelidad (RAGAS o un LLM-juez) en CI. Sin esto,
   cualquier cambio de chunk size o de modelo es una corazonada. Es lo primero que haría.

2. **Búsqueda híbrida densa + sparse con fusión RRF.** Qdrant soporta vectores sparse de
   forma nativa. Es la mejora de calidad más directa para nombres de producto y cifras.

3. **Ingesta incremental programada.** Hoy la ingesta es completa y manual. Con el
   `content_hash` que ya se guarda, reindexar solo lo que cambió es inmediato; falta el
   planificador (cron o Celery beat) y detección de páginas eliminadas.

4. **Streaming de la respuesta (SSE).** La latencia percibida bajaría mucho: el usuario
   empieza a leer mientras el modelo sigue escribiendo.

5. **Resumen progresivo del historial** para conversaciones largas, combinado con la ventana
   deslizante actual: resumen de lo antiguo + últimos N mensajes literales.

6. **Clasificar la intención antes de reescribir.** La reescritura de consulta ya está
   implementada ([§9.8](#98-la-búsqueda-necesitaba-reescribir-la-consulta-no-solo-recordarla)),
   pero se ejecuta en todo turno con historial. Detectar antes si la pregunta ya es
   autocontenida ahorraría esa llamada, y en la misma pasada permitiría reconocer las
   preguntas meta sobre la propia conversación (limitación 10).

7. **Caché semántica de respuestas.** Preguntas equivalentes reusarían la respuesta,
   recortando coste y latencia. La analítica ya muestra qué temas se repiten.

8. **Observabilidad de verdad.** Exportador Prometheus (el patrón Observer ya deja el hueco
   hecho: es una clase nueva) y trazas OpenTelemetry por etapa.

9. **Autenticación, rate limiting y multi-tenant** para un despliegue real.

10. **Migraciones con Alembic** y backups del histórico.

---

## 12. Tests

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

No necesitan Docker: la imagen de la aplicación se construye solo con las dependencias de
ejecución (`pip install .`), sin las de desarrollo, para no cargar la imagen de producción con
pytest y sus transitivas.

**83 tests, sin dependencias externas** — sin Docker, sin Qdrant, sin OpenAI, sin PostgreSQL.
Se inyectan dobles de embeddings, base vectorial, LLM y repositorio. Que esto sea posible es
la comprobación práctica de que las abstracciones descritas en [§5](#5-patrones-de-diseño)
sirven para algo.

Qué cubren:

- configuración: Singleton, parseo de listas CSV, validaciones cruzadas, redacción de secretos;
- normalización de URLs (variantes equivalentes colapsan a la misma clave);
- limpieza de HTML: navegación, cookies, deduplicación por plantilla, páginas sin contenido;
- almacenamiento de crudos y limpios, y reprocesado sin re-crawl;
- chunking: ningún chunk excede el tamaño, no se pierde contenido, IDs deterministas;
- prompt: orden de mensajes, presupuesto de contexto, intercalado del historial;
- **memoria conversacional**: aislamiento por ID, ventana de N configurable, respuestas
  huérfanas, override por petición;
- **reescritura de consulta**: no se reescribe la primera pregunta, sí los seguimientos sin
  sujeto, se descartan reescrituras degeneradas, y un fallo del proveedor no impide
  responder;
- cortocircuito sin contexto (no se llama al LLM) y degradación ante fallo del LLM;
- contrato HTTP completo, incluidos los códigos de error;
- analítica: recorrido del histórico, preguntas sin responder, temas, páginas citadas,
  satisfacción y coste.

```bash
ruff check src tests    # linter, sin hallazgos
```

### Validación end-to-end contra el sistema real

Los tests unitarios usan dobles. Para comprobar el sistema completo —scraping real,
Qdrant, reranker, OpenAI y PostgreSQL— hay un script que ejercita la API en marcha:

```bash
docker compose up -d && docker compose run --rm ingest    # si aún no lo hiciste
pip install requests
python scripts/e2e_check.py
```

Verifica en 8 bloques: salud y corpus indexado, memoria conversacional multi-turno
(incluida una pregunta de seguimiento que *solo* se puede resolver con el historial),
aislamiento entre conversaciones, honestidad ante preguntas fuera del corpus, persistencia,
feedback, las cuatro familias de métricas y la no exposición de secretos.

Esta batería es la que destapó tres defectos reales que los tests unitarios no podían ver:
la conexión al LLM rota por comentarios en el `.env`, el coste estimado siempre a cero por
los identificadores de modelo con fecha, y el `avg_top_score` inválido por mezclar dos
escalas de puntuación. Los tres están corregidos y cubiertos por tests de regresión.

### Campaña de pruebas exigentes

Además del E2E funcional se sometió el sistema en marcha a cuatro baterías. Los scripts
están en `scripts/`. Resumen de lo medido:

**Rendimiento** (`bench_rendimiento.py`) — latencia secuencial p50/p95/p99, desglose por
etapa, concurrencia 1/2/5/10/20 y una conversación de 12 turnos. Reveló que el reranking
era el 63% de la latencia y que el consumo de CPU (862% de media, 1298% en pico) venía de
sobresuscripción de hilos, no de falta de máquina → corregido en
[§9.7](#97-usar-todos-los-núcleos-era-22-más-lento-que-usar-la-mitad).

**Calidad** — 12 preguntas con hechos verificados manualmente contra el sitio:
**12/12 datos correctos, 12/12 fuente correcta, 92% fundamentadas**. Cuatro preguntas
fuera del corpus: **4/4 rechazadas** sin inventar. Y tres cadenas conversacionales, que
destaparon el fallo de recuperación multi-turno → corregido en
[§9.8](#98-la-búsqueda-necesitaba-reescribir-la-consulta-no-solo-recordarla).

**Robustez y seguridad** — más de 40 comprobaciones, **todas correctas**: límites del
contrato (mensaje vacío, 2001 caracteres, JSON malformado, método incorrecto), entradas
extremas (emojis, símbolos, HTML/script, texto repetitivo), **4 inyecciones SQL** en el
`conversation_id` (las tablas sobrevivieron: SQLAlchemy parametriza), **5 inyecciones de
prompt** (ignorar instrucciones, cambio de rol, filtrar el system prompt, forzar
invención, falsa autoridad — todas rechazadas), idempotencia y ausencia de fuga de
secretos en `/config`, `/health` y en los mensajes de error.

**Resiliencia** (`e2e_resiliencia.py`, destructiva pero reversible) — se paran Qdrant y
PostgreSQL y se comprueba que el sistema **explica el problema y se recupera solo**.
Destapó que una base caída devolvía 400 y 500 opacos → corregido a 503 con mensaje limpio.
Arranque en frío: 22 s hasta `ok`, 5,8 s la primera consulta.

**UI** — Lighthouse: *Best Practices* 100, *Accessibility* 94, cero errores de consola. Los
fallos restantes son marcado generado por Streamlit y métricas de indexación pública
(`meta-description`, `robots.txt`) que no aplican a una herramienta interna.

Resultado del E2E funcional sobre la ejecución de referencia (116 páginas rastreadas,
113 documentos indexados, 801 fragmentos):

```
[OK] estado 'ok' con corpus indexado          [OK] turno 2 recuerda los mensajes previos
[OK] LLM alcanzable                           [OK] una conversación distinta no hereda memoria
[OK] turno 1 fundamentado en el corpus        [OK] no inventa: se declara sin información
[OK] similitud vectorial en [0,1]             [OK] los 3 turnos quedaron persistidos
[OK] reranker presente y separado             [OK] avg_top_score es coseno en [0,1]
...                                           RESULTADO: TODO CORRECTO
```

---

## Licencia

MIT. Proyecto desarrollado como prueba técnica. El contenido extraído de
`https://www.bbva.com.co/` pertenece a BBVA Colombia y se usa aquí únicamente con fines
demostrativos.
