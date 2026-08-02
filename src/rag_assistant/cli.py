"""Interfaz de línea de comandos.

Cubre el ciclo completo del sistema sin necesidad de la UI web:

    rag-assistant scrape        # rastrea el sitio y guarda crudos + limpios
    rag-assistant index         # vectoriza e indexa el corpus
    rag-assistant ingest        # scrape + index en un solo paso
    rag-assistant chat          # chat interactivo con memoria
    rag-assistant analytics     # métricas de impacto del histórico
    rag-assistant conversations # listar/inspeccionar el histórico
    rag-assistant health        # estado de las dependencias
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from rag_assistant import __version__
from rag_assistant.config import get_settings
from rag_assistant.core.exceptions import RAGAssistantError
from rag_assistant.core.logging import configure_logging

app = typer.Typer(
    name="rag-assistant",
    help="Sistema RAG sobre el sitio institucional de BBVA Colombia.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _bootstrap() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    settings.ensure_directories()


@app.callback()
def main() -> None:
    """Punto de entrada común: configura logging y directorios."""
    _bootstrap()


# ----------------------------------------------------------------- scrape ---
@app.command()
def scrape(
    seeds: Annotated[list[str] | None, typer.Option("--seed", "-s", help="URL semilla")] = None,
    max_pages: Annotated[int | None, typer.Option(help="Máximo de páginas")] = None,
    fresh: Annotated[bool, typer.Option(help="Borra los datos locales antes")] = False,
) -> None:
    """Rastrea el sitio y guarda los datos crudos y limpios en `data/`."""
    from rag_assistant.scraping import ScrapeStorage, SiteCrawler

    settings = get_settings()
    if max_pages:
        object.__setattr__(settings, "scraper_max_pages", max_pages)

    storage = ScrapeStorage(settings.raw_data_dir, settings.clean_data_dir)
    if fresh:
        storage.reset()

    console.print(
        Panel(
            f"[bold]Sitio:[/] {settings.scraper_base_url}\n"
            f"[bold]Estrategia:[/] {settings.scraper_fetcher}\n"
            f"[bold]Máx. páginas:[/] {settings.scraper_max_pages}  "
            f"[bold]Profundidad:[/] {settings.scraper_max_depth}",
            title="Scraping",
        )
    )

    crawler = SiteCrawler(settings, storage=storage)
    with console.status("Rastreando el sitio…"):
        report = asyncio.run(crawler.crawl(seeds))

    _print_dict("Resultado del scraping", report.as_dict())
    if report.documents_saved == 0:
        console.print("[red]No se guardó ningún documento.[/] Revisa la configuración.")
        raise typer.Exit(1)


# ------------------------------------------------------------------ index ---
@app.command()
def index(
    recreate: Annotated[bool, typer.Option(help="Recrea la colección desde cero")] = False,
    reclean: Annotated[bool, typer.Option(help="Re-limpia el HTML crudo antes de indexar")] = False,
) -> None:
    """Vectoriza el corpus limpio y lo indexa en la base vectorial."""
    from rag_assistant.indexing import IndexingPipeline
    from rag_assistant.pipelines import run_reclean_and_index
    from rag_assistant.scraping import ScrapeStorage

    settings = get_settings()
    try:
        if reclean:
            with console.status("Re-limpiando e indexando…"):
                result = run_reclean_and_index(recreate=recreate)
            report = result.index
        else:
            storage = ScrapeStorage(settings.raw_data_dir, settings.clean_data_dir)
            documents = storage.load_clean()
            console.print(f"Documentos en el corpus: [bold]{len(documents)}[/]")
            with console.status("Vectorizando e indexando…"):
                report = IndexingPipeline(settings).run(documents, recreate=recreate)
    except RAGAssistantError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    _print_dict("Resultado de la indexación", report.as_dict())


# ----------------------------------------------------------------- ingest ---
@app.command()
def ingest(
    seeds: Annotated[list[str] | None, typer.Option("--seed", "-s")] = None,
    max_pages: Annotated[int | None, typer.Option()] = None,
    recreate: Annotated[bool, typer.Option(help="Recrea la colección")] = True,
    fresh: Annotated[bool, typer.Option(help="Borra los datos locales antes")] = False,
    skip_scrape: Annotated[bool, typer.Option(help="Solo indexar lo ya descargado")] = False,
) -> None:
    """Ingesta completa: scraping + indexación en un solo comando."""
    from rag_assistant.pipelines import run_ingestion

    try:
        with console.status("Ingesta en curso (puede tardar varios minutos)…"):
            result = run_ingestion(
                seeds=seeds,
                max_pages=max_pages,
                recreate=recreate,
                skip_scrape=skip_scrape,
                fresh=fresh,
            )
    except RAGAssistantError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    if result.crawl:
        _print_dict("Scraping", result.crawl.as_dict())
    _print_dict("Indexación", result.index.as_dict())
    console.print("\n[green]Ingesta completada.[/] Ya puedes preguntar: `rag-assistant chat`")


# ------------------------------------------------------------------- chat ---
@app.command()
def chat(
    conversation_id: Annotated[
        str | None, typer.Option("--conversation", "-c", help="ID de conversación a continuar")
    ] = None,
    question: Annotated[
        str | None, typer.Option("--question", "-q", help="Pregunta única (modo no interactivo)")
    ] = None,
    show_sources: Annotated[bool, typer.Option(help="Mostrar las fuentes citadas")] = True,
) -> None:
    """Chat conversacional con memoria de los últimos N mensajes."""
    from rag_assistant.conversation import init_database
    from rag_assistant.rag import RAGEngine

    settings = get_settings()
    init_database()

    with console.status("Cargando modelos…"):
        engine = RAGEngine(settings)
        engine.warmup()

    def ask(text: str) -> str:
        with console.status("Pensando…"):
            answer = engine.ask(text, conversation_id=conversation_id, channel="cli")
        console.print(Panel(Markdown(answer.answer), title="Asistente", border_style="cyan"))
        if show_sources and answer.citations:
            table = Table("N.º", "Fuente", "Score", show_header=True, header_style="dim")
            for citation in answer.citations:
                table.add_row(f"[{citation.index}]", citation.url, f"{citation.score:.3f}")
            console.print(table)
        console.print(
            f"[dim]{answer.latency_ms} ms · recuperación {answer.retrieval_ms} ms · "
            f"rerank {answer.rerank_ms} ms · LLM {answer.llm_ms} ms · "
            f"{answer.total_tokens} tokens · historial {answer.history_used} msg[/]\n"
        )
        return answer.conversation_id

    if question:
        ask(question)
        return

    console.print(
        Panel(
            "Asistente RAG de BBVA Colombia.\n"
            f"Memoria: últimos [bold]{settings.history_window_size}[/] mensajes.\n"
            "Escribe [bold]salir[/] para terminar.",
            title=f"Chat interactivo · v{__version__}",
            border_style="green",
        )
    )
    while True:
        try:
            text = console.input("[bold green]Tú[/] › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Hasta luego.[/]")
            return
        if text.lower() in {"salir", "exit", "quit", "q"}:
            console.print("[dim]Hasta luego.[/]")
            return
        if not text:
            continue
        conversation_id = ask(text)


# -------------------------------------------------------------- analytics ---
@app.command()
def analytics(
    days: Annotated[int | None, typer.Option(help="Acota el periodo en días")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Salida JSON cruda")] = False,
) -> None:
    """Recorre el histórico de conversaciones y extrae métricas de impacto."""
    from rag_assistant.analytics import AnalyticsService
    from rag_assistant.conversation import init_database

    init_database()
    report = AnalyticsService().report(days=days)

    if as_json:
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
        return

    if report.total_conversations == 0:
        console.print("[yellow]Aún no hay conversaciones registradas.[/]")
        return

    periodo = f"últimos {days} días" if days else "todo el histórico"
    console.print(Panel(f"Periodo analizado: [bold]{periodo}[/]", title="Analítica"))

    _metrics_table(
        "1 · Volumen y adopción",
        {
            "Conversaciones": report.total_conversations,
            "Mensajes totales": report.total_messages,
            "Preguntas": report.total_questions,
            "Respuestas": report.total_answers,
            "Mensajes por conversación (media)": report.avg_messages_per_conversation,
            "Conversación más larga": report.max_messages_in_conversation,
            "Tasa multi-turno": _pct(report.multi_turn_conversation_rate),
        },
    )
    _metrics_table(
        "2 · Calidad de las respuestas",
        {
            "Respuestas fundamentadas": _pct(report.grounded_rate),
            "Sin información en el corpus": _pct(report.no_answer_rate),
            "Errores": _pct(report.error_rate),
            "Chunks recuperados (media)": report.avg_retrieved_chunks,
            "Score medio del mejor chunk": report.avg_top_score,
            "Respuestas con reranking": _pct(report.reranked_rate),
            "Satisfacción declarada": (
                _pct(report.satisfaction_rate)
                if report.satisfaction_rate is not None
                else "sin valoraciones"
            ),
        },
    )
    _metrics_table(
        "3 · Rendimiento y coste",
        {
            "Latencia media": f"{report.avg_latency_ms} ms",
            "Latencia p50 / p95": f"{report.p50_latency_ms} / {report.p95_latency_ms} ms",
            "  └ recuperación": f"{report.avg_retrieval_ms} ms",
            "  └ reranking": f"{report.avg_rerank_ms} ms",
            "  └ LLM": f"{report.avg_llm_ms} ms",
            "Tokens totales": report.total_prompt_tokens + report.total_completion_tokens,
            "Tokens por respuesta (media)": report.avg_tokens_per_answer,
            "Coste estimado": f"USD {report.estimated_cost_usd:.4f}",
            "Coste por conversación": f"USD {report.cost_per_conversation_usd:.5f}",
        },
    )

    if report.top_topics:
        table = Table(title="4 · Temas más consultados", header_style="bold")
        table.add_column("Término")
        table.add_column("Frecuencia", justify="right")
        for topic in report.top_topics:
            table.add_row(topic["term"], str(topic["count"]))
        console.print(table)

    if report.top_cited_pages:
        table = Table(title="Páginas más citadas", header_style="bold")
        table.add_column("URL", overflow="fold")
        table.add_column("Citas", justify="right")
        for page in report.top_cited_pages:
            table.add_row(page["url"], str(page["citations"]))
        console.print(table)

    if report.unanswered_questions:
        console.print(
            Panel(
                "\n".join(f"• {q}" for q in report.unanswered_questions),
                title="Preguntas sin responder (contenido a indexar)",
                border_style="yellow",
            )
        )


# ---------------------------------------------------------- conversations ---
@app.command()
def conversations(
    conversation_id: Annotated[
        str | None, typer.Option("--id", help="Ver el detalle de una conversación")
    ] = None,
    limit: Annotated[int, typer.Option(help="Número de conversaciones a listar")] = 20,
) -> None:
    """Lista el histórico de conversaciones o muestra una en detalle."""
    from rag_assistant.conversation import SqlConversationRepository, init_database

    init_database()
    repository = SqlConversationRepository()

    if conversation_id:
        try:
            detail = repository.get_conversation(conversation_id)
        except RAGAssistantError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        console.print(Panel(f"[bold]{detail['title'] or detail['id']}[/]", title=detail["id"]))
        for message in detail["messages"]:
            style = "green" if message["role"] == "user" else "cyan"
            who = "Tú" if message["role"] == "user" else "Asistente"
            console.print(f"[{style}]{who}:[/] {message['content']}\n")
        return

    rows = repository.list_conversations(limit=limit)
    if not rows:
        console.print("[yellow]No hay conversaciones registradas.[/]")
        return

    table = Table(title="Historial de conversaciones", header_style="bold")
    table.add_column("ID")
    table.add_column("Título", overflow="ellipsis", max_width=48)
    table.add_column("Msgs", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Actualizada")
    for row in rows:
        table.add_row(
            row["id"],
            row["title"] or "—",
            str(row["message_count"]),
            str(row["total_tokens"]),
            row["updated_at"][:19].replace("T", " "),
        )
    console.print(table)


# ----------------------------------------------------------------- health ---
@app.command()
def health() -> None:
    """Comprueba el estado de la base vectorial, el LLM y el corpus indexado."""
    from rag_assistant.rag import RAGEngine

    try:
        details = RAGEngine().health()
    except RAGAssistantError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(json.dumps(details, ensure_ascii=False, default=str))


@app.command()
def config() -> None:
    """Muestra la configuración efectiva (sin secretos)."""
    console.print_json(json.dumps(get_settings().redacted(), ensure_ascii=False, default=str))


# ------------------------------------------------------------- utilidades ---
def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _metrics_table(title: str, rows: dict[str, object]) -> None:
    table = Table(title=title, header_style="bold", show_header=False, title_justify="left")
    table.add_column("Métrica", style="dim")
    table.add_column("Valor", justify="right")
    for key, value in rows.items():
        table.add_row(key, str(value))
    console.print(table)


def _print_dict(title: str, payload: dict) -> None:
    table = Table(title=title, header_style="bold", show_header=False, title_justify="left")
    table.add_column("Campo", style="dim")
    table.add_column("Valor", overflow="fold")
    for key, value in payload.items():
        if key == "errors" and value:
            table.add_row(key, "\n".join(str(v) for v in value[:5]))
        elif key != "errors":
            table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":
    app()
