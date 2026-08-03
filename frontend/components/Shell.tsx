"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";
import type { Health } from "@/lib/types";

const NAV = [
  { href: "/", label: "Conversación" },
  { href: "/analitica", label: "Analítica" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [fallo, setFallo] = useState(false);
  const [tema, setTema] = useState<"light" | "dark" | null>(null);

  useEffect(() => {
    const cargar = () =>
      getHealth()
        .then((h) => { setHealth(h); setFallo(false); })
        .catch(() => setFallo(true));
    cargar();
    const t = setInterval(cargar, 30000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const guardado = localStorage.getItem("tema") as "light" | "dark" | null;
    if (guardado) {
      setTema(guardado);
      document.documentElement.dataset.theme = guardado;
    }
  }, []);

  function alternarTema() {
    const actual =
      tema ??
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const nuevo = actual === "dark" ? "light" : "dark";
    setTema(nuevo);
    document.documentElement.dataset.theme = nuevo;
    localStorage.setItem("tema", nuevo);
  }

  const estado = fallo ? "unhealthy" : health?.status;
  const color =
    estado === "ok" ? "var(--good)" : estado === "degraded" ? "var(--warn)" : "var(--bad)";

  return (
    <div className="flex min-h-screen flex-col">
      <header
        className="sticky top-0 z-20 flex items-center gap-4 border-b px-4 py-3 sm:px-6"
        style={{ borderColor: "var(--border)", background: "var(--bg)" }}
      >
        <Link href="/" className="flex items-center gap-2.5 shrink-0">
          <span
            className="grid h-7 w-7 place-items-center rounded text-[13px] font-bold"
            style={{ background: "var(--color-navy)", color: "#fff" }}
            aria-hidden
          >
            B
          </span>
          <span className="text-[15px] font-semibold tracking-tight">
            Asistente <span className="dim font-normal">· BBVA Colombia</span>
          </span>
        </Link>

        <nav className="ml-2 flex gap-1">
          {NAV.map((n) => {
            const activo = pathname === n.href;
            return (
              <Link
                key={n.href}
                href={n.href}
                className="rounded px-3 py-1.5 text-sm transition-colors"
                style={{
                  background: activo ? "var(--surface-2)" : "transparent",
                  color: activo ? "var(--fg)" : "var(--fg-dim)",
                  fontWeight: activo ? 600 : 400,
                }}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <span
            className="mono hidden items-center gap-2 text-[11px] sm:flex"
            title={
              health
                ? `${health.details.llm.model} · ventana de ${health.details.history_window_size} mensajes`
                : "Consultando estado…"
            }
          >
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: color }}
              aria-hidden
            />
            <span className="dim">
              {fallo
                ? "API no disponible"
                : health
                  ? `${health.details.indexed_chunks.toLocaleString("es")} fragmentos`
                  : "…"}
            </span>
          </span>
          <button
            onClick={alternarTema}
            className="rounded border px-2 py-1 text-[11px] transition-colors"
            style={{ borderColor: "var(--border)", color: "var(--fg-dim)" }}
            aria-label="Cambiar entre tema claro y oscuro"
          >
            Tema
          </button>
        </div>
      </header>

      <main className="flex flex-1 flex-col">{children}</main>
    </div>
  );
}
