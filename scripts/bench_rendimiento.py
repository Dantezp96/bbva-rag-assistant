"""Banco de pruebas de rendimiento contra la API en marcha."""
import statistics as st
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

API = "http://localhost:8000"

PREGUNTAS = [
    "¿Qué tipos de cuenta de ahorro ofrece BBVA Colombia?",
    "¿Hasta qué porcentaje financia BBVA un crédito de vivienda?",
    "¿Qué plazos tiene el crédito de vehículo?",
    "¿Qué es un CDT y qué modalidades ofrece BBVA?",
    "¿Qué tarjetas de crédito tiene BBVA Colombia?",
    "¿Qué requisitos piden para un crédito de libranza?",
    "¿Qué es el adelanto de nómina?",
    "¿Qué fondos de inversión ofrece BBVA?",
    "¿Cómo funciona la cuenta de ahorro digital?",
    "¿Qué beneficios tiene la tarjeta débito?",
    "¿Qué seguros ofrece BBVA para vehículos?",
    "¿Qué es el leasing habitacional?",
]


def pedir(pregunta, conv=None, timeout=180):
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{API}/chat",
            json={"message": pregunta, "conversation_id": conv or f"perf-{uuid.uuid4().hex[:8]}"},
            timeout=timeout,
        )
        wall = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return {"ok": False, "wall": wall, "status": r.status_code}
        d = r.json()
        return {
            "ok": True, "wall": wall, "server": d["latency_ms"],
            "ret": d["retrieval_ms"], "rer": d["rerank_ms"], "llm": d["llm_ms"],
            "tok": d["prompt_tokens"] + d["completion_tokens"],
            "grounded": d["grounded"], "fuentes": len(d["sources"]),
        }
    except Exception as e:
        return {"ok": False, "wall": (time.perf_counter() - t0) * 1000, "error": type(e).__name__}


def pct(datos, p):
    if not datos:
        return 0
    o = sorted(datos)
    return o[min(len(o) - 1, int(len(o) * p / 100))]


def resumen(nombre, res):
    ok = [r for r in res if r["ok"]]
    fallos = len(res) - len(ok)
    if not ok:
        print(f"  {nombre}: 0/{len(res)} exitosas")
        return
    w = [r["wall"] for r in ok]
    print(f"  {nombre}: {len(ok)}/{len(res)} ok, {fallos} fallos")
    print(f"    wall-clock  min={min(w):7.0f}  p50={pct(w,50):7.0f}  p95={pct(w,95):7.0f}  "
          f"p99={pct(w,99):7.0f}  max={max(w):7.0f} ms")
    print(f"    etapas      recuperacion={st.mean(r['ret'] for r in ok):6.0f}  "
          f"rerank={st.mean(r['rer'] for r in ok):6.0f}  llm={st.mean(r['llm'] for r in ok):6.0f} ms")
    print(f"    tokens/resp {st.mean(r['tok'] for r in ok):.0f}   "
          f"fundamentadas {sum(r['grounded'] for r in ok)}/{len(ok)}")


print("=" * 84)
print("A. CALENTAMIENTO  (primera consulta tras arranque: carga de modelos ONNX)")
print("=" * 84)
frio = pedir(PREGUNTAS[0])
print(f"  1a consulta: {frio['wall']:.0f} ms  (rerank {frio.get('rer','?')} ms)")
seg = pedir(PREGUNTAS[1])
print(f"  2a consulta: {seg['wall']:.0f} ms  (rerank {seg.get('rer','?')} ms)")
print(f"  -> coste de calentamiento: {frio['wall'] - seg['wall']:+.0f} ms")

print()
print("=" * 84)
print("B. LATENCIA SECUENCIAL EN CALIENTE  (24 consultas, 1 a la vez)")
print("=" * 84)
t0 = time.perf_counter()
sec = [pedir(PREGUNTAS[i % len(PREGUNTAS)]) for i in range(24)]
dur = time.perf_counter() - t0
resumen("secuencial", sec)
print(f"    throughput  {24/dur:.2f} consultas/s  ({dur:.0f} s totales)")

print()
print("=" * 84)
print("C. CARGA CONCURRENTE  (misma API, concurrencia creciente)")
print("=" * 84)
for n in (2, 5, 10, 20):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda i: pedir(PREGUNTAS[i % len(PREGUNTAS)]), range(n * 2)))
    dur = time.perf_counter() - t0
    resumen(f"concurrencia={n:<2} ({n*2} consultas)", res)
    print(f"    throughput  {len(res)/dur:.2f} consultas/s  ({dur:.0f} s)")
    print()

print("=" * 84)
print("D. CONVERSACION LARGA  (12 turnos en el mismo hilo: coste de la memoria)")
print("=" * 84)
conv = f"largo-{uuid.uuid4().hex[:8]}"
for i in range(12):
    r = pedir(PREGUNTAS[i % len(PREGUNTAS)], conv=conv)
    if r["ok"]:
        print(f"  turno {i+1:2d}  {r['wall']:6.0f} ms   prompt+comp={r['tok']:5d} tok")
    else:
        print(f"  turno {i+1:2d}  FALLO {r}")
d = requests.get(f"{API}/chat/conversations/{conv}", timeout=30).json()
print(f"  -> mensajes persistidos: {len(d['messages'])}  tokens acumulados: {d['total_tokens']}")
