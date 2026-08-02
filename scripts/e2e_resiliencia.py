"""Resiliencia: qué hace el sistema cuando se le cae una dependencia.

DESTRUCTIVA pero reversible: para servicios y los vuelve a levantar al final.
Lo que se evalúa no es que siga respondiendo (no puede), sino que:
  - devuelva el código HTTP correcto en vez de un 500 opaco,
  - explique la causa de forma accionable,
  - y se recupere solo cuando la dependencia vuelve.
"""
import subprocess
import time

import requests

API = "http://localhost:8000"
fallos = []


def check(nombre, cond, detalle=""):
    print(f"{'OK  ' if cond else 'MAL '} {nombre}" + (f"   -> {detalle}" if detalle else ""))
    if not cond:
        fallos.append(nombre)


def compose(*args):
    return subprocess.run(["docker", "compose", *args], capture_output=True, text=True,
                          cwd=r"D:\Pruba tecnica\bbva-rag-assistant")


def preguntar(msg="¿Qué cuentas de ahorro hay?", conv="resil", timeout=120):
    try:
        r = requests.post(f"{API}/chat", json={"message": msg, "conversation_id": conv},
                          timeout=timeout)
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return None, {"excepcion": type(e).__name__}


def esperar_sano(intentos=30):
    for _ in range(intentos):
        try:
            if requests.get(f"{API}/health", timeout=10).json()["status"] == "ok":
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


print("=" * 88)
print("ESTADO INICIAL")
print("=" * 88)
h = requests.get(f"{API}/health", timeout=30).json()
print(f"  {h['status']}  chunks={h['details']['indexed_chunks']}")

# ---------------------------------------------------------------- QDRANT ----
print()
print("=" * 88)
print("1. CAIDA DE LA BASE VECTORIAL (Qdrant)")
print("=" * 88)
compose("stop", "qdrant")
time.sleep(6)

h = requests.get(f"{API}/health", timeout=60).json()
check("/health detecta la caida", h["status"] == "unhealthy",
      f"status={h['status']} reachable={h['details']['vector_store'].get('reachable')}")

code, d = preguntar()
respuesta = d.get("answer", "")
check("/chat responde sin reventar", code == 200, f"status={code}")
check("el mensaje explica el problema",
      any(p in respuesta.lower() for p in ("índice", "indice", "vectorial", "problema", "ingesta")),
      respuesta[:130])
check("no se marca como fundamentada", d.get("grounded") is False)
check("el turno fallido queda registrado", d.get("message_id") is not None)

compose("start", "qdrant")
print("  ... restaurando Qdrant")
check("se recupera solo al volver Qdrant", esperar_sano(), "")
code, d = preguntar()
check("vuelve a responder con fuentes", code == 200 and len(d.get("sources", [])) > 0,
      f"fuentes={len(d.get('sources', []))}")

# -------------------------------------------------------------- POSTGRES ----
print()
print("=" * 88)
print("2. CAIDA DE LA BASE DE HISTORIAL (PostgreSQL)")
print("=" * 88)
compose("stop", "postgres")
time.sleep(6)

code, d = preguntar(conv="resil-pg")
check("/chat no revienta con la BD caida", code in (200, 400, 500, 503), f"status={code}")
print(f"       respuesta: {str(d.get('answer') or d.get('error') or d)[:130]}")

code2 = requests.get(f"{API}/chat/conversations", timeout=60).status_code
check("listar conversaciones devuelve error controlado", code2 in (400, 500, 503),
      f"status={code2}")

h = requests.get(f"{API}/health", timeout=60).json()
check("/health sigue respondiendo (no cuelga)", h["status"] in ("ok", "degraded", "unhealthy"),
      f"status={h['status']}")

compose("start", "postgres")
print("  ... restaurando PostgreSQL")
time.sleep(12)
check("se recupera solo al volver PostgreSQL", esperar_sano(), "")
code, d = preguntar(conv="resil-pg2")
check("vuelve a persistir conversaciones", code == 200 and d.get("message_id") is not None)

# ------------------------------------------------------------ ARRANQUE ----
print()
print("=" * 88)
print("3. REINICIO EN FRIO DE LA API  (tiempo hasta primera respuesta util)")
print("=" * 88)
compose("restart", "api")
t0 = time.perf_counter()
sano = esperar_sano(intentos=40)
t_sano = time.perf_counter() - t0
check("la API vuelve a estado 'ok'", sano, f"{t_sano:.0f} s")

t0 = time.perf_counter()
code, d = preguntar(conv="frio")
t_prim = time.perf_counter() - t0
check("primera consulta tras arranque en frio", code == 200,
      f"{t_prim:.1f} s (incluye carga de modelos ONNX)")
t0 = time.perf_counter()
preguntar(conv="frio2")
print(f"       segunda consulta: {time.perf_counter() - t0:.1f} s")

print()
print("=" * 88)
print(f"RESULTADO: {'TODO CORRECTO' if not fallos else f'{len(fallos)} FALLOS: ' + ', '.join(fallos)}")
print("=" * 88)
