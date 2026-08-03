"use client";

import { useEffect, useState } from "react";

import { getAnalytics } from "@/lib/api";
import type { Analytics } from "@/lib/types";

const PERIODOS = [
  { label: "Todo", days: undefined },
  { label: "7 días", days: 7 },
  { label: "30 días", days: 30 },
  { label: "90 días", days: 90 },
];

function pct(v: number | null | undefined, d = 0) {
  return v == null ? "—" : `${(v * 100).toFixed(d)}%`;
}

function Kpi({
  label, value, hint, tone,
}: { label: string; value: string; hint?: string; tone?: "good" | "warn" }) {
  const color = tone === "good" ? "var(--good)" : tone === "warn" ? "var(--warn)" : "var(--fg)";
  return (
    <div className="surface flex flex-col gap-1.5 px-4 py-4">
      <span className="eyebrow">{label}</span>
      <span className="mono text-[24px] font-semibold leading-none" style={{ color }}>
        {value}
      </span>
      {hint && <span className="dim text-[11px] leading-snug">{hint}</span>}
    </div>
  );
}

function Seccion({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="flex items-baseline gap-2.5 text-[20px]">
        <span className="mono text-[12px]" style={{ color: "var(--accent)" }}>{n}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

/** Barras horizontales: comparar magnitudes se lee mejor que una tabla. */
function Barras({ datos, unidad = "" }: { datos: { k: string; v: number }[]; unidad?: string }) {
  const max = Math.max(...datos.map((d) => d.v), 1);
  return (
    <div className="flex flex-col gap-2">
      {datos.map((d) => (
        <div key={d.k} className="flex items-center gap-3">
          <span className="w-32 shrink-0 truncate text-[12.5px]">{d.k}</span>
          <span className="h-2 flex-1 overflow-hidden rounded-full" style={{ background: "var(--surface)" }}>
            <span
              className="block h-full rounded-full"
              style={{ width: `${(d.v / max) * 100}%`, background: "var(--accent)" }}
            />
          </span>
          <span className="mono w-16 shrink-0 text-right text-[11.5px]" style={{ color: "var(--fg-dim)" }}>
            {d.v.toLocaleString("es", { maximumFractionDigits: 0 })}{unidad}
          </span>
        </div>
      ))}
    </div>
  );
}

export function Dashboard() {
  const [datos, setDatos] = useState<Analytics | null>(null);
  const [dias, setDias] = useState<number | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDatos(null);
    setError(null);
    getAnalytics(dias)
      .then(setDatos)
      .catch(() => setError("No se pudo cargar la analítica."));
  }, [dias]);

  if (error)
    return <p className="mx-auto max-w-5xl px-6 py-14 text-[14px]" style={{ color: "var(--bad)" }}>{error}</p>;

  if (!datos)
    return <p className="dim mx-auto max-w-5xl px-6 py-14 text-[14px]">Cargando métricas…</p>;

  if (datos.total_conversations === 0)
    return (
      <div className="mx-auto max-w-5xl px-6 py-14">
        <p className="text-[15px]">Todavía no hay conversaciones registradas.</p>
        <p className="dim mt-1 text-[14px]">Usa el chat y vuelve aquí.</p>
      </div>
    );

  const tokens = datos.total_prompt_tokens + datos.total_completion_tokens;

  return (
    <div className="card mx-auto flex w-full max-w-5xl flex-col gap-9 px-5 py-8 sm:px-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow mb-2">Histórico de conversaciones</p>
          <h1 className="text-[32px]">Analítica de uso</h1>
        </div>
        <div className="flex gap-1">
          {PERIODOS.map((p) => (
            <button
              key={p.label}
              onClick={() => setDias(p.days)}
              className="rounded px-2.5 py-1.5 text-[12.5px] transition-colors"
              style={{
                background: dias === p.days ? "var(--surface)" : "transparent",
                color: dias === p.days ? "var(--fg)" : "var(--fg-dim)",
                fontWeight: dias === p.days ? 600 : 400,
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </header>

      <Seccion n="01" title="Volumen y adopción">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Kpi label="Conversaciones" value={datos.total_conversations.toLocaleString("es")} />
          <Kpi label="Mensajes" value={datos.total_messages.toLocaleString("es")} />
          <Kpi label="Msgs / conversación" value={String(datos.avg_messages_per_conversation)} />
          <Kpi
            label="Tasa multi-turno"
            value={pct(datos.multi_turn_conversation_rate)}
            hint="Mide si la memoria conversacional aporta valor real"
          />
        </div>
      </Seccion>

      <Seccion n="02" title="Calidad de las respuestas">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Kpi label="Fundamentadas" value={pct(datos.grounded_rate, 1)} tone="good"
               hint="Respaldadas por fragmentos del corpus" />
          <Kpi label="Sin información" value={pct(datos.no_answer_rate, 1)} tone="warn"
               hint="Cada punto es contenido por indexar" />
          <Kpi label="Score medio" value={datos.avg_top_score.toFixed(3)}
               hint="Similitud coseno del mejor fragmento" />
          <Kpi
            label="Satisfacción"
            value={pct(datos.satisfaction_rate)}
            hint={`↑ ${datos.feedback_positive} · ↓ ${datos.feedback_negative}`}
          />
        </div>
        {datos.unanswered_questions.length > 0 && (
          <div className="surface px-4 py-3.5">
            <p className="eyebrow mb-2">Preguntas sin responder · contenido pendiente</p>
            <ul className="flex flex-col gap-1.5">
              {datos.unanswered_questions.slice(0, 6).map((q, i) => (
                <li key={i} className="text-[13px]">{q}</li>
              ))}
            </ul>
          </div>
        )}
      </Seccion>

      <Seccion n="03" title="Rendimiento y coste">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Kpi label="Latencia p50" value={`${(datos.p50_latency_ms / 1000).toFixed(1)} s`} />
          <Kpi label="Latencia p95" value={`${(datos.p95_latency_ms / 1000).toFixed(1)} s`} />
          <Kpi label="Tokens" value={tokens.toLocaleString("es")} />
          <Kpi
            label="Coste / conversación"
            value={`US$ ${datos.cost_per_conversation_usd.toFixed(5)}`}
            hint={`US$ ${datos.estimated_cost_usd.toFixed(4)} en total`}
          />
        </div>
        <div className="surface px-4 py-4">
          <p className="eyebrow mb-3">Latencia media por etapa · dice dónde optimizar</p>
          <Barras
            unidad=" ms"
            datos={[
              { k: "Recuperación", v: datos.avg_retrieval_ms },
              { k: "Reranking", v: datos.avg_rerank_ms },
              { k: "Generación", v: datos.avg_llm_ms },
            ]}
          />
        </div>
        {datos.untariffed_models?.length > 0 && (
          <p className="mono text-[11px]" style={{ color: "var(--warn)" }}>
            Sin tarifa conocida, no contados en el coste: {datos.untariffed_models.join(", ")}
          </p>
        )}
      </Seccion>

      <Seccion n="04" title="Contenido">
        <div className="grid gap-3 md:grid-cols-2">
          <div className="surface px-4 py-4">
            <p className="eyebrow mb-3">Temas más consultados</p>
            <Barras datos={datos.top_topics.slice(0, 7).map((t) => ({ k: t.term, v: t.count }))} />
          </div>
          <div className="surface px-4 py-4">
            <p className="eyebrow mb-3">Páginas más citadas</p>
            <ul className="flex flex-col gap-2">
              {datos.top_cited_pages.slice(0, 7).map((p) => (
                <li key={p.url} className="flex items-center gap-3">
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mono flex-1 truncate text-[11.5px] hover:underline"
                    style={{ color: "var(--fg-dim)" }}
                  >
                    {p.url.replace(/^https?:\/\/[^/]+\//, "").replace(/\.html.*$/, "")}
                  </a>
                  <span className="mono shrink-0 text-[11.5px]" style={{ color: "var(--accent)" }}>
                    {p.citations}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Seccion>
    </div>
  );
}
