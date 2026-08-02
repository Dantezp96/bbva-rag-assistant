"""Normalización de URLs, limpieza de HTML y almacenamiento local."""

from __future__ import annotations

import pytest

from rag_assistant.core.exceptions import ExtractionError
from rag_assistant.core.models import RawPage
from rag_assistant.scraping import ScrapeStorage, clean_page, normalize_url

PRODUCT_HTML = """
<html lang="es-CO">
<head>
  <title>Cuenta de Ahorro Digital | BBVA Colombia</title>
  <meta name="description" content="Abre tu cuenta de ahorro 100% digital.">
</head>
<body>
  <header><nav><a href="/personas">Personas</a><a href="/empresas">Empresas</a></nav></header>
  <div class="cookie-banner">Utilizamos cookies propias y de terceros para mejorar
    nuestros servicios y mostrar publicidad relacionada.</div>
  <main>
    <h1>Cuenta de Ahorro Digital</h1>
    <p>Abre tu cuenta en minutos desde la app, sin ir a una oficina y sin papeleo.</p>
    <h2>Beneficios de la cuenta</h2>
    <ul>
      <li>Sin cuota de manejo durante el primer año completo de vigencia</li>
      <li>Retiros gratuitos en toda la red de cajeros automáticos BBVA</li>
      <li>Transferencias ilimitadas a otras entidades mediante la aplicación</li>
    </ul>
    <h2>Requisitos de apertura</h2>
    <p>Ser mayor de edad, tener cédula de ciudadanía vigente y residir en Colombia.</p>
  </main>
  <footer><p>Aceptar cookies</p><p>Cerrar</p></footer>
  <script>console.log("analytics");</script>
</body>
</html>
"""


def _page(html: str, url: str = "https://www.bbva.com.co/personas/cuentas/ahorro.html") -> RawPage:
    return RawPage(url=url, html=html, status_code=200)


# ------------------------------------------------------- normalización -------
@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("https://WWW.BBVA.com.co/Personas/", "https://www.bbva.com.co/Personas"),
        ("https://www.bbva.com.co/a.html#seccion", "https://www.bbva.com.co/a.html"),
        ("https://www.bbva.com.co/a?utm_source=x&id=7", "https://www.bbva.com.co/a?id=7"),
        ("https://www.bbva.com.co", "https://www.bbva.com.co/"),
    ],
)
def test_normalize_url(entrada, esperado):
    assert normalize_url(entrada) == esperado


def test_urls_equivalentes_colapsan_a_la_misma_clave():
    """Es lo que evita rastrear y indexar la misma página varias veces."""
    variantes = [
        "https://www.bbva.com.co/personas/",
        "https://www.bbva.com.co/personas",
        "https://www.bbva.com.co/personas#top",
        "https://www.bbva.com.co/personas?utm_campaign=abc",
    ]
    assert len({normalize_url(u) for u in variantes}) == 1


# ------------------------------------------------------------ limpieza -------
def test_clean_page_extrae_contenido_y_metadatos():
    doc = clean_page(_page(PRODUCT_HTML))
    assert doc.title == "Cuenta de Ahorro Digital"
    assert doc.description.startswith("Abre tu cuenta")
    assert doc.language == "es"
    assert "Sin cuota de manejo" in doc.text
    assert "Requisitos de apertura" in doc.text
    assert doc.word_count > 30


def test_clean_page_elimina_navegacion_scripts_y_cookies():
    doc = clean_page(_page(PRODUCT_HTML))
    assert "analytics" not in doc.text
    assert "Utilizamos cookies propias" not in doc.text
    assert "Aceptar cookies" not in doc.text


def test_clean_page_deduplica_lineas_repetidas_por_plantilla():
    html = PRODUCT_HTML.replace(
        "</main>",
        "<p>Abre tu cuenta en minutos desde la app, sin ir a una oficina y sin papeleo.</p></main>",
    )
    doc = clean_page(_page(html))
    assert doc.text.count("sin ir a una oficina y sin papeleo") == 1


def test_pagina_sin_contenido_util_lanza_error_explicito():
    with pytest.raises(ExtractionError):
        clean_page(_page("<html><body><nav>Inicio</nav></body></html>"))


def test_seccion_se_deriva_del_path():
    doc = clean_page(_page(PRODUCT_HTML))
    assert doc.section == "personas/cuentas"


# ----------------------------------------------------------- storage --------
def test_storage_persiste_crudos_y_limpios(tmp_path):
    """Requisito de la prueba: almacenar los datos crudos y limpios en local."""
    storage = ScrapeStorage(tmp_path / "raw", tmp_path / "clean")
    page = _page(PRODUCT_HTML)

    raw_path = storage.save_raw(page)
    assert raw_path.exists()
    assert raw_path.read_text(encoding="utf-8") == PRODUCT_HTML

    storage.save_clean(clean_page(page))
    assert storage.stats() == {"raw_pages": 1, "clean_documents": 1}

    docs = storage.load_clean()
    assert len(docs) == 1
    assert docs[0].url == page.url


def test_reprocesar_desde_crudos_no_requiere_volver_a_rastrear(tmp_path):
    storage = ScrapeStorage(tmp_path / "raw", tmp_path / "clean")
    storage.save_raw(_page(PRODUCT_HTML))

    recuperadas = list(storage.iter_raw())
    assert len(recuperadas) == 1
    assert recuperadas[0].html == PRODUCT_HTML


def test_load_clean_deduplica_por_url(tmp_path):
    """Reindexar una página actualiza su versión, no crea un duplicado."""
    storage = ScrapeStorage(tmp_path / "raw", tmp_path / "clean")
    doc = clean_page(_page(PRODUCT_HTML))
    storage.save_clean(doc)
    storage.save_clean(doc)
    assert len(storage.load_clean()) == 1
