"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Answer } from "./Answer";
import { Sources } from "./Sources";
import { Telemetry } from "./Telemetry";
import { getStarters, streamChat } from "@/lib/api";
import { STAGE_LABELS, type Message, type Source, type Telemetry as T } from "@/lib/types";

const nuevaConversacion = () => `web-${Math.random().toString(36).slice(2, 12)}`;

export function Chat() {
  const [mensajes, setMensajes] = useState<Message[]>([]);
  // El id se crea al montar, no al renderizar: `Math.random()` daría un valor
  // en el prerender del servidor y otro en el cliente, y React aborta la
  // hidratación al no coincidir el texto.
  const [conversacion, setConversacion] = useState("");
  const [entrada, setEntrada] = useState("");
  const [etapa, setEtapa] = useState<string | null>(null);
  const [ocupado, setOcupado] = useState(false);
  const [starters, setStarters] = useState<{ questions: string[]; source: string } | null>(null);

  const abortar = useRef<(() => void) | null>(null);
  const finRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setConversacion(nuevaConversacion());
    getStarters().then(setStarters).catch(() => setStarters(null));
  }, []);

  // Solo se auto-desplaza si el usuario ya estaba al final: si ha subido a leer
  // una respuesta anterior, arrastrarle hacia abajo es hostil.
  useEffect(() => {
    const cerca =
      window.innerHeight + window.scrollY >= document.body.offsetHeight - 220;
    if (cerca) finRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [mensajes]);

  useEffect(() => () => abortar.current?.(), []);

  const enviar = useCallback(
    (texto: string) => {
      const pregunta = texto.trim();
      if (!pregunta || ocupado || !conversacion) return;

      const idAsistente = `a-${Date.now()}`;
      setMensajes((m) => [
        ...m,
        { id: `u-${Date.now()}`, role: "user", content: pregunta },
        { id: idAsistente, role: "assistant", content: "", streaming: true },
      ]);
      setEntrada("");
      setOcupado(true);
      setEtapa("rewrite");

      const actualizar = (cambio: Partial<Message>) =>
        setMensajes((m) => m.map((x) => (x.id === idAsistente ? { ...x, ...cambio } : x)));

      abortar.current = streamChat(
        { message: pregunta, conversation_id: conversacion },
        {
          onStart: (d) => setConversacion(d.conversation_id),
          onStage: (s) => setEtapa(s),
          onSources: (d) => {
            setEtapa("generando");
            actualizar({ sources: d.sources as Source[] });
          },
          onToken: (t) => {
            setEtapa(null);
            setMensajes((m) =>
              m.map((x) => (x.id === idAsistente ? { ...x, content: x.content + t } : x)),
            );
          },
          onError: (msg) => {
            actualizar({ content: msg, streaming: false, error: true });
            setEtapa(null);
            setOcupado(false);
          },
          onDone: (d) => {
            actualizar({
              streaming: false,
              telemetry: d as unknown as T,
              suggestions: (d.suggestions as string[]) ?? [],
            });
            setEtapa(null);
            setOcupado(false);
          },
        },
      );
    },
    [conversacion, ocupado],
  );

  function reiniciar() {
    abortar.current?.();
    setMensajes([]);
    setConversacion(nuevaConversacion());
    setOcupado(false);
    setEtapa(null);
    getStarters().then(setStarters).catch(() => {});
  }

  const vacio = mensajes.length === 0;
  const ultimo = mensajes[mensajes.length - 1];
  const sugerencias = !ocupado && ultimo?.role === "assistant" ? ultimo.suggestions ?? [] : [];

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 sm:px-6">
      {/* ------------------------------------------------ estado vacío --- */}
      {vacio && (
        <div className="flex flex-1 flex-col justify-center py-10">
          <p className="eyebrow mb-3">Asistente sobre bbva.com.co</p>
          <h1 className="mb-2 text-[27px] font-bold leading-tight tracking-tight sm:text-[32px]">
            ¿Qué necesitas saber?
          </h1>
          <p className="dim mb-7 max-w-[52ch] text-[15px]">
            Respondo con el contenido publicado en el sitio de BBVA Colombia y cito
            siempre la página de la que sale cada dato.
          </p>

          {starters && starters.questions.length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="eyebrow">
                {starters.source === "historico"
                  ? "Las más consultadas"
                  : "Para empezar"}
              </p>
              <div className="flex flex-col gap-1.5">
                {starters.questions.map((q) => (
                  <button
                    key={q}
                    onClick={() => enviar(q)}
                    className="surface group flex items-center gap-3 rounded-md px-4 py-3 text-left text-[14px] transition-all hover:brightness-[0.97]"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <span className="flex-1">{q}</span>
                    <span
                      className="mono shrink-0 text-[13px] opacity-0 transition-opacity group-hover:opacity-100"
                      style={{ color: "var(--accent)" }}
                      aria-hidden
                    >
                      →
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------ conversación --- */}
      {!vacio && (
        <div className="flex flex-1 flex-col gap-6 py-7">
          {mensajes.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="flex justify-end">
                <p
                  className="max-w-[85%] rounded-2xl rounded-br-sm px-4 py-2.5 text-[15px]"
                  style={{ background: "var(--surface-2)" }}
                >
                  {m.content}
                </p>
              </div>
            ) : (
              <div key={m.id} className="rise flex flex-col">
                {m.content ? (
                  <Answer text={m.content} streaming={m.streaming} />
                ) : (
                  etapa && (
                    <p className="mono flex items-center gap-2 text-[12px]" style={{ color: "var(--fg-dim)" }}>
                      <span
                        className="inline-block h-1.5 w-1.5 animate-pulse rounded-full"
                        style={{ background: "var(--accent)" }}
                        aria-hidden
                      />
                      {STAGE_LABELS[etapa] ?? "Redactando la respuesta"}…
                    </p>
                  )
                )}
                {m.sources && <Sources sources={m.sources} />}
                {m.telemetry && <Telemetry t={m.telemetry} />}

                {/* Las sugerencias pertenecen a ESTA respuesta, así que van
                    pegadas a ella y no flotando sobre el campo de entrada. */}
                {m.id === ultimo?.id && sugerencias.length > 0 && (
                  <div className="rise mt-3 flex flex-wrap gap-1.5">
                    {sugerencias.map((s) => (
                      <button
                        key={s}
                        onClick={() => enviar(s)}
                        className="rounded-full border px-3 py-1.5 text-[12.5px] transition-colors hover:brightness-[0.97]"
                        style={{ borderColor: "var(--border)", background: "var(--surface)" }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ),
          )}
          <div ref={finRef} />
        </div>
      )}

      {/* ---------------------------------------------------- composer --- */}
      <div className="sticky bottom-0 pb-4 pt-2" style={{ background: "var(--bg)" }}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            enviar(entrada);
          }}
          className="surface flex items-end gap-2 rounded-xl p-2 transition-shadow focus-within:shadow-[0_0_0_2px_var(--ring)]"
        >
          <textarea
            ref={areaRef}
            value={entrada}
            onChange={(e) => {
              setEntrada(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                enviar(entrada);
              }
            }}
            rows={1}
            maxLength={2000}
            placeholder="Escribe tu pregunta…"
            disabled={ocupado}
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] outline-none disabled:opacity-50"
            aria-label="Tu pregunta"
          />
          <button
            type="submit"
            disabled={!entrada.trim() || ocupado}
            className="grid h-9 w-9 shrink-0 place-items-center rounded-lg text-[15px] transition-opacity disabled:opacity-30"
            style={{ background: "var(--accent)", color: "var(--accent-fg)" }}
            aria-label="Enviar pregunta"
          >
            ↑
          </button>
        </form>

        <div className="mt-2 flex items-center justify-between px-1">
          <p className="mono text-[10px]" style={{ color: "var(--fg-dim)" }}>
            {conversacion}
          </p>
          {!vacio && (
            <button
              onClick={reiniciar}
              className="mono text-[10px] transition-colors hover:underline"
              style={{ color: "var(--fg-dim)" }}
            >
              nueva conversación
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
