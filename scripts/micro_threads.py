"""Microbenchmark aislado: coste real del cross-encoder según número de hilos ONNX.

Se mide fuera de la API para separar el coste del modelo del resto del pipeline.
Hipótesis a validar: con un lote pequeño, 12 hilos sincronizan más de lo que
computan, y menos hilos es MÁS rápido, no menos.
"""
import os
import statistics as st
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

CACHE = "/app/models_cache"
MODELO = "Xenova/ms-marco-MiniLM-L-6-v2"
CONSULTA = "¿Hasta qué porcentaje financia BBVA un crédito de vivienda?"

# Pasajes representativos: ~900 caracteres, como los chunks reales.
BASE = (
    "Crédito de vivienda BBVA Colombia. Adquiere vivienda nueva o usada con plazos "
    "desde 5 hasta 30 años. Financiamos hasta el 70% del valor del inmueble para "
    "vivienda No VIS superior a 135 SMLMV, y hasta el 80% para vivienda VIS inferior "
    "a 150 SMLMV en las poblaciones decretadas por el Gobierno Nacional o 135 SMLMV "
    "para el resto del país. Aplican condiciones y restricciones. Simula tu crédito "
    "en línea y conoce la cuota mensual estimada según el monto y el plazo elegido. "
    "También financiamos proyectos de construcción y compra de cartera hipotecaria "
    "de otras entidades financieras con mejores condiciones de tasa. "
)
PASAJES_20 = [f"{BASE} Variante {i}." for i in range(20)]

print(f"nucleos visibles: {os.cpu_count()}")
print(f"{'hilos':>6} | {'top_k=20':>12} | {'top_k=12':>12} | {'top_k=5':>12}")
print("-" * 54)

for hilos in (None, 1, 2, 3, 4, 6, 8, 12):
    kwargs = {"model_name": MODELO, "cache_dir": CACHE}
    if hilos is not None:
        kwargs["threads"] = hilos
    modelo = TextCrossEncoder(**kwargs)
    list(modelo.rerank(CONSULTA, PASAJES_20[:2]))  # calentamiento

    fila = []
    for k in (20, 12, 5):
        tiempos = []
        for _ in range(5):
            t0 = time.perf_counter()
            list(modelo.rerank(CONSULTA, PASAJES_20[:k]))
            tiempos.append((time.perf_counter() - t0) * 1000)
        fila.append(st.median(tiempos))
    etiqueta = "auto" if hilos is None else str(hilos)
    print(f"{etiqueta:>6} | {fila[0]:9.0f} ms | {fila[1]:9.0f} ms | {fila[2]:9.0f} ms")
