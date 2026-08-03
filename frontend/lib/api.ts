import type { Analytics, Health } from "./types";

export const API =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} en ${path}`);
  return r.json() as Promise<T>;
}

export const getHealth = () => get<Health>("/health");
export const getAnalytics = (days?: number) =>
  get<Analytics>(`/analytics${days ? `?days=${days}` : ""}`);
export const getStarters = () =>
  get<{ questions: string[]; source: string }>("/chat/starters");
export const getConversations = () =>
  get<
    { id: string; title: string; message_count: number; total_tokens: number;
      updated_at: string }[]
  >("/chat/conversations?limit=60");

export async function sendFeedback(messageId: number, value: 1 | -1) {
  await fetch(`${API}/chat/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message_id: messageId, value }),
  });
}

/** Handlers de los eventos que emite `POST /chat/stream`. */
export interface StreamHandlers {
  onStart?: (d: { conversation_id: string }) => void;
  onStage?: (stage: string) => void;
  onSources?: (d: { sources: unknown[]; search_query: string; rewritten: boolean }) => void;
  onToken?: (text: string) => void;
  onDone?: (d: Record<string, unknown>) => void;
  onError?: (message: string) => void;
}

/**
 * Consume el stream SSE del chat.
 *
 * Se usa `fetch` + `ReadableStream` y no `EventSource` porque este endpoint es
 * un POST con cuerpo JSON, y `EventSource` solo sabe hacer GET. A cambio hay
 * que trocear el flujo a mano: los bloques SSE se separan por línea en blanco
 * y un bloque puede quedar partido entre dos chunks de red, así que se
 * mantiene un búfer con el resto.
 *
 * Devuelve una función para abortar (el usuario cambia de conversación o
 * cierra la pestaña a mitad de respuesta).
 */
export function streamChat(
  body: { message: string; conversation_id?: string; history_window?: number },
  h: StreamHandlers,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const r = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!r.ok || !r.body) {
        h.onError?.(`El servidor respondió ${r.status}.`);
        return;
      }

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let corte: number;
        while ((corte = buffer.indexOf("\n\n")) !== -1) {
          const bloque = buffer.slice(0, corte);
          buffer = buffer.slice(corte + 2);

          let evento = "message";
          let datos = "";
          for (const linea of bloque.split("\n")) {
            if (linea.startsWith("event: ")) evento = linea.slice(7).trim();
            else if (linea.startsWith("data: ")) datos += linea.slice(6);
          }
          if (!datos) continue;

          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(datos);
          } catch {
            continue;
          }

          switch (evento) {
            case "start":
              h.onStart?.(payload as { conversation_id: string });
              break;
            case "stage":
              h.onStage?.(payload.stage as string);
              break;
            case "sources":
              h.onSources?.(payload as never);
              break;
            case "token":
              h.onToken?.(payload.text as string);
              break;
            case "error":
              h.onError?.(payload.message as string);
              break;
            case "done":
              h.onDone?.(payload);
              break;
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        h.onError?.(
          "No se pudo contactar con el asistente. Comprueba que la API esté en marcha.",
        );
      }
    }
  })();

  return () => controller.abort();
}
