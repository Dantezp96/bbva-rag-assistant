"use client";

import { useState } from "react";

import type { Source } from "@/lib/types";

/** Convierte la URL en una miga de pan legible: /personas/productos/cuentas */
function ruta(url: string): string {
  try {
    const p = new URL(url).pathname.replace(/\.html$/, "").replace(/^\/|\/$/g, "");
    return p ? p.split("/").join(" › ") : "inicio";
  } catch {
    return url;
  }
}

export function Sources({ sources }: { sources: Source[] }) {
  const [abierto, setAbierto] = useState(false);
  if (!sources?.length) return null;

  return (
    <div className="mt-3">
      <button
        onClick={() => setAbierto((v) => !v)}
        className="mono flex items-center gap-1.5 text-[11px] transition-colors"
        style={{ color: "var(--fg-dim)" }}
        aria-expanded={abierto}
      >
        <span
          className="inline-block transition-transform"
          style={{ transform: abierto ? "rotate(90deg)" : "none" }}
          aria-hidden
        >
          ▸
        </span>
        {sources.length} {sources.length === 1 ? "FUENTE" : "FUENTES"} DEL SITIO
      </button>

      {abierto && (
        <ul className="rise mt-2 flex flex-col gap-1.5">
          {sources.map((s) => (
            <li key={s.index}>
              <a
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="surface flex items-start gap-2.5 px-3 py-2.5 transition-colors hover:brightness-[0.97]"
              >
                <span className="cite mt-0.5 shrink-0">[{s.index}]</span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-medium">
                    {s.title || ruta(s.url)}
                  </span>
                  <span className="dim mono block truncate text-[10.5px]">
                    {ruta(s.url)}
                  </span>
                </span>
                <span
                  className="mono mt-0.5 shrink-0 text-[10.5px]"
                  style={{ color: "var(--fg-dim)" }}
                  title={
                    s.rerank_score !== null
                      ? `Similitud vectorial ${s.score.toFixed(3)} · reranker ${s.rerank_score.toFixed(2)}`
                      : `Similitud vectorial ${s.score.toFixed(3)}`
                  }
                >
                  {s.score.toFixed(2)}
                </span>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
