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
    <div className="flex min-h-screen flex-col gap-3 p-3 sm:gap-4 sm:p-4">
      {/* La cabecera es una tarjeta flotando sobre el fondo gris, como en
          bbva.com.co: nada va pegado al borde de la ventana. */}
      <header className="card sticky top-3 z-20 flex items-center gap-3 px-4 py-3 sm:top-4 sm:gap-5 sm:px-6">
        <Link href="/" className="flex shrink-0 items-center gap-2.5">
          <span
            className="grid h-8 w-8 place-items-center text-[13px] font-bold"
            style={{
              background: "var(--accent)",
              color: "var(--accent-fg)",
              borderRadius: "var(--radius-control)",
            }}
            aria-hidden
          >
            B
          </span>
          <span className="text-[15px] leading-tight">
            <span className="font-semibold">Asistente</span>
            <span className="dim hidden sm:inline"> · BBVA Colombia</span>
          </span>
        </Link>

        <nav className="flex gap-1">
          {NAV.map((n) => {
            const activo = pathname === n.href;
            return (
              <Link
                key={n.href}
                href={n.href}
                className="px-3 py-1.5 text-[14px] transition-colors"
                style={{
                  background: activo ? "var(--surface)" : "transparent",
                  color: activo ? "var(--accent)" : "var(--fg-dim)",
                  fontWeight: activo ? 600 : 400,
                  borderRadius: "var(--radius-control)",
                }}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2.5">
          <span
            className="mono hidden items-center gap-2 text-[11.5px] sm:flex"
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
            className="px-2.5 py-1.5 text-[12px] transition-colors"
            style={{
              background: "var(--surface)",
              color: "var(--fg-dim)",
              borderRadius: "var(--radius-control)",
            }}
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
