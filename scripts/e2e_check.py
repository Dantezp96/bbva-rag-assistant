"""Validación E2E contra el stack real en Docker."""
import json, sys, time
import requests

API = "http://localhost:8000"
OK, FAIL = "[OK]", "[FALLO]"
errors = []


def check(name, cond, extra=""):
    print(f"{OK if cond else FAIL} {name}" + (f"  -> {extra}" if extra else ""))
    if not cond:
        errors.append(name)


def ask(msg, conv, **kw):
    r = requests.post(f"{API}/chat", json={"message": msg, "conversation_id": conv, **kw}, timeout=180)
    r.raise_for_status()
    return r.json()


print("=" * 78)
print("1. SALUD DEL SISTEMA")
print("=" * 78)
h = requests.get(f"{API}/health", timeout=30).json()
print(f"   estado={h['status']}  chunks={h['details']['indexed_chunks']}  "
      f"llm={h['details']['llm']['model']} ok={h['details']['llm']['reachable']}  "
      f"reranker={h['details']['reranker']['model']}")
check("estado 'ok' con corpus indexado", h["status"] == "ok" and h["details"]["indexed_chunks"] > 500)
check("LLM alcanzable", h["details"]["llm"]["reachable"])
check("ventana de memoria configurada", h["details"]["history_window_size"] == 6)

print()
print("=" * 78)
print("2. MEMORIA CONVERSACIONAL MULTI-TURNO  (requisito central)")
print("=" * 78)
CONV = f"e2e-{int(time.time())}"

t1 = ask("¿Qué créditos de vivienda ofrece BBVA Colombia?", CONV)
print(f"\n--- Turno 1 ---\n{t1['answer'][:420]}...")
print(f"    historial usado: {t1['history_used']} | fundamentada: {t1['grounded']} | "
      f"reranked: {t1['reranked']} | fuentes: {len(t1['sources'])}")
check("turno 1 sin historial previo", t1["history_used"] == 0)
check("turno 1 fundamentado en el corpus", t1["grounded"])
check("turno 1 devuelve fuentes citadas", len(t1["sources"]) > 0)
check("similitud vectorial en [0,1]", all(0 <= s["score"] <= 1 for s in t1["sources"]),
      f"scores={[round(s['score'],3) for s in t1['sources']]}")
check("puntuación del reranker presente y separada",
      any(s.get("rerank_score") is not None for s in t1["sources"]),
      f"rerank={[s.get('rerank_score') for s in t1['sources']]}")

# Pregunta que SOLO se puede resolver con el historial: no menciona "vivienda".
t2 = ask("¿Y hasta qué porcentaje financian?", CONV)
print(f"\n--- Turno 2 (referencia implícita, no dice 'vivienda') ---\n{t2['answer'][:420]}...")
print(f"    historial usado: {t2['history_used']} | fundamentada: {t2['grounded']}")
check("turno 2 recuerda los mensajes previos", t2["history_used"] == 2)
check("turno 2 sigue en la misma conversación", t2["conversation_id"] == CONV)

t3 = ask("Resume en una frase lo que te acabo de preguntar.", CONV)
print(f"\n--- Turno 3 ---\n{t3['answer'][:300]}")
check("turno 3 acumula historial (ventana N=6)", t3["history_used"] == 4)

print()
print("=" * 78)
print("3. AISLAMIENTO ENTRE CONVERSACIONES")
print("=" * 78)
otra = ask("¿Y hasta qué porcentaje financian?", f"{CONV}-aislada")
print(f"    historial usado en conversación nueva: {otra['history_used']}")
check("una conversación distinta no hereda memoria", otra["history_used"] == 0)

print()
print("=" * 78)
print("4. HONESTIDAD: PREGUNTA FUERA DEL CORPUS")
print("=" * 78)
fuera = ask("¿Cuál es la capital de Mongolia y su población exacta?", f"{CONV}-fuera")
print(f"    {fuera['answer'][:280]}")
print(f"    fundamentada: {fuera['grounded']} | fuentes: {len(fuera['sources'])}")
check("no inventa: se declara sin información",
      (not fuera["grounded"]) or "no encontré" in fuera["answer"].lower()
      or "no encuentro" in fuera["answer"].lower())

print()
print("=" * 78)
print("5. PERSISTENCIA DEL HISTORIAL")
print("=" * 78)
det = requests.get(f"{API}/chat/conversations/{CONV}", timeout=30).json()
print(f"    conversación '{det['id']}' | título: {det['title'][:60]!r}")
print(f"    mensajes persistidos: {len(det['messages'])} | tokens: {det['total_tokens']}")
check("los 3 turnos quedaron persistidos (6 mensajes)", len(det["messages"]) == 6)
check("el título es la primera pregunta", det["title"].startswith("¿Qué créditos de vivienda"))
check("404 para conversación inexistente",
      requests.get(f"{API}/chat/conversations/no-existe", timeout=30).status_code == 404)

print()
print("=" * 78)
print("6. FEEDBACK")
print("=" * 78)
mid = t1["message_id"]
r = requests.post(f"{API}/chat/feedback", json={"message_id": mid, "value": 1}, timeout=30)
check("feedback positivo aceptado (204)", r.status_code == 204, f"status={r.status_code}")
mid2 = t2["message_id"]
requests.post(f"{API}/chat/feedback", json={"message_id": mid2, "value": -1}, timeout=30)
bad = requests.post(f"{API}/chat/feedback", json={"message_id": mid, "value": 9}, timeout=30)
check("feedback inválido rechazado (400)", bad.status_code == 400, f"status={bad.status_code}")

print()
print("=" * 78)
print("7. ANALÍTICA DEL HISTÓRICO")
print("=" * 78)
a = requests.get(f"{API}/analytics", timeout=60).json()
print(f"    conversaciones={a['total_conversations']}  mensajes={a['total_messages']}  "
      f"preguntas={a['total_questions']}")
print(f"    fundamentadas={a['grounded_rate']:.0%}  sin info={a['no_answer_rate']:.0%}  "
      f"multi-turno={a['multi_turn_conversation_rate']:.0%}")
print(f"    latencia media={a['avg_latency_ms']:.0f}ms  p95={a['p95_latency_ms']}ms  "
      f"(recup={a['avg_retrieval_ms']:.0f} rerank={a['avg_rerank_ms']:.0f} llm={a['avg_llm_ms']:.0f})")
print(f"    score medio (coseno)={a['avg_top_score']:.3f}  tokens={a['total_prompt_tokens']+a['total_completion_tokens']}  "
      f"coste=US${a['estimated_cost_usd']:.5f}")
print(f"    satisfacción={a['satisfaction_rate']}  (+{a['feedback_positive']} / -{a['feedback_negative']})")
print(f"    temas top: {[t['term'] for t in a['top_topics'][:6]]}")
print(f"    páginas más citadas: {[p['url'].split('/')[-1] for p in a['top_cited_pages'][:4]]}")
print(f"    preguntas sin responder: {a['unanswered_questions'][:2]}")

check("recorre el histórico completo", a["total_conversations"] >= 3 and a["total_messages"] >= 8)
check("desglosa latencia por etapa", a["avg_retrieval_ms"] > 0 and a["avg_llm_ms"] > 0)
check("avg_top_score es una similitud coseno en [0,1]", 0 < a["avg_top_score"] <= 1,
      f"valor={a['avg_top_score']}")
check("estima coste", a["estimated_cost_usd"] > 0)
check("registra satisfacción", a["feedback_positive"] == 1 and a["feedback_negative"] == 1)
check("extrae temas consultados", len(a["top_topics"]) > 0)
check("extrae páginas más citadas", len(a["top_cited_pages"]) > 0)
check("detecta preguntas sin responder", len(a["unanswered_questions"]) > 0)

live = requests.get(f"{API}/analytics/live", timeout=30).json()
check("contadores en vivo del proceso", live.get("queries", 0) > 0, f"queries={live.get('queries')}")

print()
print("=" * 78)
print("8. CONFIGURACIÓN Y SECRETOS")
print("=" * 78)
cfg = requests.get(f"{API}/config", timeout=30).json()
check("la clave de OpenAI no se expone", cfg["openai_api_key"] == "***redacted***")
check("chunk size externalizado", cfg["chunk_size"] == 900)
check("ventana de historial externalizada", cfg["history_window_size"] == 6)

print()
print("=" * 78)
print(f"RESULTADO: {'TODO CORRECTO' if not errors else str(len(errors)) + ' FALLOS: ' + ', '.join(errors)}")
print("=" * 78)
sys.exit(1 if errors else 0)
