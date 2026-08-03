"use client";

import { useState } from "react";

import { sendFeedback } from "@/lib/api";
import type { Telemetry as T } from "@/lib/types";

/** Barra proporcional del reparto de tiempo entre etapas del pipeline. */
function Etapas({ t }: { t: T }) {
  const etapas = [
    { k: "Reescritura", ms: t.rewrite_ms, c: "var(--color-sky)" },
    { k: "Recuperación", ms: t.retrieval_ms, c: "var(--good)" },
    { k: "Reranking", ms: t.rerank_ms, c: "var(--accent)" },
    { k: "Generación", ms: t.llm_ms, c: "var(--color-navy)" },
    { k: "Sugerencias", ms: t.suggestions_ms, c: "var(--warn)" },
  ].filter((e) => e.ms > 0);
  const total = etapas.reduce((s, e) => s + e.ms, 0) || 1;

  return (
    <div className="mt-2 flex flex-col gap-1.5">
      <div className="flex h-1.5 overflow-hidden rounded-full">
        {etapas.map((e) => (
          <span
            key={e.k}
            style={{ width: `${(e.ms / total) * 100}%`, background: e.c }}
            title={`${e.k}: ${e.ms} ms`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5">
        {etapas.map((e) => (
          <span key={e.k} className="mono flex items-center gap-1 text-[10px]" style={{ color: "var(--fg-dim)" }}>
            <span className="inline-block h-1.5 w-1.5 rounded-sm" style={{ background: e.c }} aria-hidden />
            {e.k} {e.ms}ms
          </span>
        ))}
      </div>
    </div>
  );
}

export function Telemetry({ t }: { t: T }) {
  const [abierto, setAbierto] = useState(false);
  const [voto, setVoto] = useState<1 | -1 | null>(null);

  async function valorar(v: 1 | -1) {
    if (!t.message_id || voto) return;
    setVoto(v);
    try {
      await sendFeedback(t.message_id, v);
    } catch {
      setVoto(null);
    }
  }

  const tokens = t.prompt_tokens + t.completion_tokens;

  return (
    <div className="mt-3 flex flex-col gap-2 border-t pt-2.5" style={{ borderColor: "var(--border)" }}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <button
          onClick={() => setAbierto((v) => !v)}
          className="mono text-[10.5px] transition-colors"
          style={{ color: "var(--fg-dim)" }}
          aria-expanded={abierto}
        >
          {(t.latency_ms / 1000).toFixed(1)}s · {tokens.toLocaleString("es")} tokens
        </button>

        <span
          className="mono rounded-full px-2 py-0.5 text-[10px]"
          style={{
            color: t.grounded ? "var(--good)" : "var(--warn)",
            background: `color-mix(in srgb, ${t.grounded ? "var(--good)" : "var(--warn)"} 12%, transparent)`,
          }}
          title={
            t.grounded
              ? "La respuesta se apoya en fragmentos del sitio"
              : "El corpus no cubría esta pregunta"
          }
        >
          {t.grounded ? "fundamentada" : "sin respaldo"}
        </span>

        {t.history_used > 0 && (
          <span className="mono text-[10px]" style={{ color: "var(--fg-dim)" }}>
            memoria: {t.history_used} msg
          </span>
        )}

        {t.message_id && (
          <span className="ml-auto flex items-center gap-1">
            {voto ? (
              <span className="mono text-[10px]" style={{ color: "var(--fg-dim)" }}>
                gracias
              </span>
            ) : (
              <>
                <button
                  onClick={() => valorar(1)}
                  className="rounded px-1.5 py-0.5 text-[13px] leading-none transition-colors hover:brightness-125"
                  style={{ color: "var(--fg-dim)" }}
                  aria-label="Respuesta útil"
                >
                  ↑
                </button>
                <button
                  onClick={() => valorar(-1)}
                  className="rounded px-1.5 py-0.5 text-[13px] leading-none transition-colors hover:brightness-125"
                  style={{ color: "var(--fg-dim)" }}
                  aria-label="Respuesta no útil"
                >
                  ↓
                </button>
              </>
            )}
          </span>
        )}
      </div>

      {abierto && (
        <div className="rise">
          <Etapas t={t} />
          {t.rewritten && (
            <p className="mono mt-2 text-[10.5px]" style={{ color: "var(--fg-dim)" }}>
              Buscado como: <span style={{ color: "var(--accent)" }}>{t.search_query}</span>
            </p>
          )}
          <p className="mono mt-1 text-[10px]" style={{ color: "var(--fg-dim)" }}>
            {t.model} · {t.prompt_tokens} entrada + {t.completion_tokens} salida
            {t.reranked ? " · reranking aplicado" : ""}
          </p>
        </div>
      )}
    </div>
  );
}
