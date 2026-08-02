"""Interfaz web conversacional (Streamlit).

Deliberadamente minimalista, como pide el enunciado: lo importante es que sea
funcional y limpia. Consume la API HTTP en vez de importar el motor RAG, de
modo que UI y backend escalan por separado y existe una única implementación
de la lógica.

Dos vistas:
  * **Chat** — conversación con memoria, fuentes citadas, telemetría y feedback.
  * **Analítica** — el informe de impacto sobre el histórico de conversaciones.
"""

from __future__ import annotations

import os
import uuid

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120

st.set_page_config(
    page_title="Asistente RAG · BBVA Colombia",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ API -----
def api_get(path: str, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    if response.status_code >= 400:
        try:
            body = response.json()
            raise RuntimeError(body.get("detail") or body.get("error") or response.text)
        except ValueError:
            raise RuntimeError(response.text) from None
    return response.json() if response.content else None


# ---------------------------------------------------------------- estado ----
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"ui-{uuid.uuid4().hex[:10]}"
if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.title("💬 Asistente BBVA")
    st.caption("RAG sobre el contenido público de bbva.com.co")

    try:
        health = api_get("/health")
        status = health["status"]
        indexed = health["details"].get("indexed_chunks", 0)
        badge = {"ok": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(status, "⚪")
        st.markdown(f"**Estado:** {badge} `{status}`")
        st.markdown(f"**Fragmentos indexados:** `{indexed:,}`")
        with st.expander("Detalle del sistema"):
            st.json(health["details"])
        if indexed == 0:
            st.warning(
                "El índice está vacío. Ejecuta la ingesta:\n\n"
                "`docker compose run --rm ingest`"
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo contactar con la API en {API_URL}\n\n{exc}")
        st.stop()

    st.divider()
    st.text_input("ID de conversación", key="conversation_id")
    st.caption("Reutiliza un ID para retomar una conversación anterior.")

    if st.button("🆕 Nueva conversación", use_container_width=True):
        st.session_state.conversation_id = f"ui-{uuid.uuid4().hex[:10]}"
        st.session_state.messages = []
        st.rerun()

    st.divider()
    view = st.radio("Vista", ["Chat", "Analítica"], label_visibility="collapsed")


# ------------------------------------------------------------------ CHAT ----
def render_chat() -> None:
    st.header("Conversación")
    st.caption(
        "Pregunta sobre productos, cuentas, tarjetas, créditos o servicios "
        "publicados en el sitio de BBVA Colombia."
    )

    if not st.session_state.messages:
        st.info(
            "**Ejemplos de preguntas**\n\n"
            "- ¿Qué tipos de cuenta de ahorro ofrece BBVA Colombia?\n"
            "- ¿Cuáles son los requisitos para un crédito de vivienda?\n"
            "- ¿Qué tarjetas de crédito tienen disponibles?\n"
            "- ¿Qué es un CDT y qué modalidades hay?"
        )

    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("sources"):
                with st.expander(f"📚 Fuentes ({len(entry['sources'])})"):
                    for source in entry["sources"]:
                        st.markdown(
                            f"**[{source['index']}]** [{source['title'] or source['url']}]"
                            f"({source['url']}) · relevancia `{source['score']:.3f}`"
                        )
            if entry.get("meta"):
                st.caption(entry["meta"])

    prompt = st.chat_input("Escribe tu pregunta…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Consultando el contenido indexado…"):
        try:
            data = api_post(
                "/chat",
                {
                    "message": prompt,
                    "conversation_id": st.session_state.conversation_id,
                },
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error al consultar la API: {exc}")
            return

        st.markdown(data["answer"])
        if data.get("sources"):
            with st.expander(f"📚 Fuentes ({len(data['sources'])})"):
                for source in data["sources"]:
                    st.markdown(
                        f"**[{source['index']}]** [{source['title'] or source['url']}]"
                        f"({source['url']}) · relevancia `{source['score']:.3f}`"
                    )

        meta = (
            f"⏱️ {data['latency_ms']} ms "
            f"(recuperación {data['retrieval_ms']} · rerank {data['rerank_ms']} · "
            f"LLM {data['llm_ms']}) · "
            f"🎫 {data['prompt_tokens'] + data['completion_tokens']} tokens · "
            f"🧠 historial: {data['history_used']} msg · "
            f"{'✅ fundamentada' if data['grounded'] else '⚠️ sin respaldo en el corpus'}"
        )
        st.caption(meta)

        if data.get("message_id"):
            col_up, col_down, _ = st.columns([1, 1, 8])
            if col_up.button("👍", key=f"up-{data['message_id']}"):
                api_post("/chat/feedback", {"message_id": data["message_id"], "value": 1})
                st.toast("¡Gracias por tu valoración!")
            if col_down.button("👎", key=f"down-{data['message_id']}"):
                api_post("/chat/feedback", {"message_id": data["message_id"], "value": -1})
                st.toast("Anotado, lo tendremos en cuenta.")

    st.session_state.conversation_id = data["conversation_id"]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "sources": data.get("sources", []),
            "meta": meta,
        }
    )


# ------------------------------------------------------------- ANALÍTICA ----
def render_analytics() -> None:
    st.header("Analítica del histórico de conversaciones")
    st.caption("Métricas de uso, calidad, rendimiento y coste extraídas de todas las sesiones.")

    period = st.selectbox(
        "Periodo", ["Todo el histórico", "Últimos 7 días", "Últimos 30 días", "Últimos 90 días"]
    )
    days = {"Últimos 7 días": 7, "Últimos 30 días": 30, "Últimos 90 días": 90}.get(period)

    try:
        report = api_get("/analytics", **({"days": days} if days else {}))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo obtener la analítica: {exc}")
        return

    if report["total_conversations"] == 0:
        st.info("Todavía no hay conversaciones registradas. Usa el chat y vuelve aquí.")
        return

    st.subheader("1 · Volumen y adopción")
    cols = st.columns(5)
    cols[0].metric("Conversaciones", f"{report['total_conversations']:,}")
    cols[1].metric("Mensajes", f"{report['total_messages']:,}")
    cols[2].metric("Preguntas", f"{report['total_questions']:,}")
    cols[3].metric("Msgs/conversación", report["avg_messages_per_conversation"])
    cols[4].metric(
        "Tasa multi-turno",
        f"{report['multi_turn_conversation_rate'] * 100:.0f}%",
        help="Conversaciones con más de un intercambio: mide si la memoria aporta valor.",
    )

    if report["conversations_per_day"]:
        frame = pd.DataFrame(report["conversations_per_day"]).set_index("date")
        st.bar_chart(frame, height=200)

    st.subheader("2 · Calidad de las respuestas")
    cols = st.columns(5)
    cols[0].metric(
        "Fundamentadas",
        f"{report['grounded_rate'] * 100:.1f}%",
        help="Respuestas respaldadas por fragmentos del corpus.",
    )
    cols[1].metric(
        "Sin información",
        f"{report['no_answer_rate'] * 100:.1f}%",
        help="El corpus no cubría la pregunta. Cada punto es contenido por indexar.",
    )
    cols[2].metric("Errores", f"{report['error_rate'] * 100:.1f}%")
    cols[3].metric("Score medio", report["avg_top_score"])
    satisfaction = report.get("satisfaction_rate")
    cols[4].metric(
        "Satisfacción",
        f"{satisfaction * 100:.0f}%" if satisfaction is not None else "—",
        help=f"👍 {report['feedback_positive']} · 👎 {report['feedback_negative']}",
    )

    st.subheader("3 · Rendimiento y coste")
    cols = st.columns(5)
    cols[0].metric("Latencia media", f"{report['avg_latency_ms']:.0f} ms")
    cols[1].metric("p95", f"{report['p95_latency_ms']} ms")
    total_tokens = report["total_prompt_tokens"] + report["total_completion_tokens"]
    cols[2].metric("Tokens totales", f"{total_tokens:,}")
    cols[3].metric("Coste estimado", f"US$ {report['estimated_cost_usd']:.4f}")
    cols[4].metric("Coste/conversación", f"US$ {report['cost_per_conversation_usd']:.5f}")

    stages = pd.DataFrame(
        {
            "etapa": ["Recuperación", "Reranking", "LLM"],
            "ms": [report["avg_retrieval_ms"], report["avg_rerank_ms"], report["avg_llm_ms"]],
        }
    ).set_index("etapa")
    st.caption("Latencia media por etapa del pipeline — indica dónde optimizar.")
    st.bar_chart(stages, height=220)

    st.subheader("4 · Contenido")
    left, right = st.columns(2)
    with left:
        st.markdown("**Temas más consultados**")
        if report["top_topics"]:
            st.dataframe(
                pd.DataFrame(report["top_topics"])[["term", "count"]].rename(
                    columns={"term": "Término", "count": "Frecuencia"}
                ),
                hide_index=True,
                use_container_width=True,
            )
    with right:
        st.markdown("**Páginas más citadas**")
        if report["top_cited_pages"]:
            st.dataframe(
                pd.DataFrame(report["top_cited_pages"])[["url", "citations"]].rename(
                    columns={"url": "URL", "citations": "Citas"}
                ),
                hide_index=True,
                use_container_width=True,
            )

    if report["unanswered_questions"]:
        st.markdown("**Preguntas sin responder** — contenido pendiente de indexar")
        for question in report["unanswered_questions"]:
            st.markdown(f"- {question}")

    with st.expander("🗂️ Conversaciones registradas"):
        try:
            rows = api_get("/chat/conversations", limit=100)
            if rows:
                st.dataframe(
                    pd.DataFrame(rows)[
                        ["id", "title", "message_count", "total_tokens", "updated_at"]
                    ].rename(
                        columns={
                            "id": "ID",
                            "title": "Título",
                            "message_count": "Mensajes",
                            "total_tokens": "Tokens",
                            "updated_at": "Actualizada",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"No se pudo listar el histórico: {exc}")

    with st.expander("📄 Informe completo (JSON)"):
        st.json(report)


if view == "Chat":
    render_chat()
else:
    render_analytics()
