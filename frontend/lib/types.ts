export interface Source {
  index: number;
  url: string;
  title: string;
  score: number;
  rerank_score: number | null;
}

/** Telemetría del turno: llega en el evento `done` del stream. */
export interface Telemetry {
  message_id: number | null;
  grounded: boolean;
  reranked: boolean;
  history_used: number;
  search_query: string;
  rewritten: boolean;
  latency_ms: number;
  rewrite_ms: number;
  retrieval_ms: number;
  rerank_ms: number;
  llm_ms: number;
  suggestions_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  model: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  suggestions?: string[];
  telemetry?: Telemetry | null;
  streaming?: boolean;
  error?: boolean;
}

export interface Health {
  status: "ok" | "degraded" | "unhealthy";
  version: string;
  details: {
    indexed_chunks: number;
    history_window_size: number;
    llm: { provider: string; model: string; reachable: boolean };
    embeddings: { model: string };
    reranker: { enabled: boolean; model: string };
    vector_store: Record<string, unknown>;
  };
}

export interface Analytics {
  total_conversations: number;
  total_messages: number;
  total_questions: number;
  unique_users: number;
  avg_messages_per_conversation: number;
  multi_turn_conversation_rate: number;
  conversations_per_day: { date: string; conversations: number }[];
  grounded_rate: number;
  no_answer_rate: number;
  error_rate: number;
  avg_top_score: number;
  satisfaction_rate: number | null;
  feedback_positive: number;
  feedback_negative: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  avg_retrieval_ms: number;
  avg_rerank_ms: number;
  avg_llm_ms: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  estimated_cost_usd: number;
  cost_per_conversation_usd: number;
  models_used: Record<string, number>;
  untariffed_models: string[];
  top_topics: { term: string; count: number }[];
  top_cited_pages: { url: string; title: string; citations: number }[];
  unanswered_questions: string[];
}

/** Nombre legible de cada etapa del pipeline, para el indicador de progreso. */
export const STAGE_LABELS: Record<string, string> = {
  rewrite: "Interpretando la pregunta",
  retrieval: "Buscando en el contenido indexado",
  rerank: "Priorizando las fuentes más relevantes",
  truncate: "Seleccionando fuentes",
  prompt: "Preparando el contexto",
};
