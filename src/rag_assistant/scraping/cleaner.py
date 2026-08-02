"""Limpieza y normalización del HTML crudo.

Convierte un `RawPage` (HTML tal cual) en un `CleanDocument` (texto útil +
metadatos). La calidad de este paso condiciona todo el RAG: si el texto llega
con menús, cookies y pies de página, los chunks se llenan de ruido y la
recuperación empeora.

Se ejecutan **dos extractores y se elige el más informativo**:

1. `trafilatura` — excelente para artículos y notas de prensa, muy bueno
   descartando boilerplate.
2. Extractor propio con BeautifulSoup — poda navegación, cookies y pies por
   selector, y reconstruye el texto respetando encabezados y listas.

Por qué los dos: medido sobre páginas de producto reales de BBVA, trafilatura
devolvía ~370 caracteres donde el extractor por DOM recuperaba ~6.900,
incluyendo justo lo que el usuario pregunta (plazos, porcentajes de
financiación, requisitos). Esas páginas son maquetación por componentes, no
prosa, y el algoritmo de densidad de texto de trafilatura las descarta. Al
revés ocurre en las páginas editoriales. Elegir la extracción más rica por
página, en vez de casarse con una librería, evita perder contenido.
"""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

from rag_assistant.core.exceptions import ExtractionError
from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import CleanDocument, RawPage

logger = get_logger(__name__)

#: Elementos que nunca contienen contenido informativo.
_DROP_TAGS = ("script", "style", "noscript", "svg", "iframe", "form", "template", "canvas")

#: Contenedores de navegación / legales, por etiqueta o por atributo.
_DROP_SELECTORS = (
    "header", "footer", "nav", "aside",
    "[role=navigation]", "[role=banner]", "[role=contentinfo]",
    "[class*=cookie]", "[id*=cookie]",
    "[class*=breadcrumb]", "[class*=menu]", "[class*=navbar]",
    "[class*=skip-link]", "[class*=social]", "[aria-hidden=true]",
)

#: Frases de boilerplate legal/UX que se repiten en todas las páginas.
_BOILERPLATE_LINES = re.compile(
    r"^(aceptar( todas)?( las)? cookies|configurar cookies|pol[ií]tica de cookies|"
    r"ir al contenido( principal)?|men[uú] principal|compartir|s[ií]guenos|"
    r"cerrar|volver|buscar|iniciar sesi[oó]n|reg[ií]strate|abre tu cuenta ya)$",
    re.IGNORECASE,
)

#: Bloques de texto legal idénticos en todas las páginas: solo añaden ruido a
#: los chunks y compiten con el contenido real en la búsqueda vectorial.
_BOILERPLATE_PREFIXES = (
    "utilizamos cookies propias y de terceros",
    "usamos cookies propias y de terceros",
    "este sitio web utiliza cookies",
)

_MIN_USEFUL_CHARS = 220

#: Cuánto más rica debe ser la extracción por DOM para preferirla a trafilatura.
#: Con empate técnico gana trafilatura, que limpia mejor el boilerplate.
_DOM_PREFERENCE_RATIO = 1.2


def _normalize_whitespace(text: str) -> str:
    """Colapsa espacios, unifica saltos y elimina caracteres de control."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _drop_boilerplate_lines(text: str) -> str:
    """Elimina navegación, avisos de cookies y líneas duplicadas por plantilla."""
    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue

        key = line.lower()
        if _BOILERPLATE_LINES.match(line) or key.startswith(_BOILERPLATE_PREFIXES):
            continue
        # Una línea idéntica repetida dentro de la misma página es maquetación
        # (el mismo bloque en cabecera, hero y pie), no contenido nuevo.
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return "\n".join(kept).strip()


def _extract_with_trafilatura(html: str, url: str) -> str:
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=False,
            favor_recall=True,
            no_fallback=False,
        )
        return extracted or ""
    except Exception as exc:  # noqa: BLE001 - trafilatura es best-effort
        logger.debug("cleaner.trafilatura_failed", url=url, error=str(exc))
        return ""


def _extract_with_soup(soup: BeautifulSoup) -> str:
    """Extractor de respaldo: poda el DOM y reconstruye el texto por bloques."""
    working = BeautifulSoup(str(soup), "lxml")

    for tag in working(_DROP_TAGS):
        tag.decompose()
    for selector in _DROP_SELECTORS:
        for node in working.select(selector):
            node.decompose()

    root = working.select_one("main") or working.select_one("[role=main]") or working.body
    if root is None:
        return ""

    blocks: list[str] = []
    for node in root.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "td", "th", "dd", "dt", "figcaption"]
    ):
        # Un <li> que contiene otra lista repetiría el texto de sus hijos
        # concatenado; se salta el contenedor y se recogen los hijos sueltos.
        if node.name == "li" and node.find(["ul", "ol"]):
            continue
        text = _normalize_whitespace(node.get_text(" ", strip=True))
        if not text or len(text) < 3:
            continue
        if node.name.startswith("h"):
            blocks.append(f"\n## {text}")
        elif node.name in ("li", "td", "th", "dd", "dt"):
            blocks.append(f"- {text}")
        else:
            blocks.append(text)
    return "\n".join(blocks)


def _guess_section(url: str) -> str:
    """Deriva una sección temática del path de la URL (útil como filtro/metadato)."""
    path = re.sub(r"^https?://[^/]+", "", url).strip("/")
    if not path:
        return "home"
    parts = [p for p in path.split("/") if p and not p.endswith(".html")]
    return "/".join(parts[:2]) if parts else "home"


def clean_page(page: RawPage) -> CleanDocument:
    """Convierte HTML crudo en un documento limpio listo para indexar."""
    soup = BeautifulSoup(page.html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = _normalize_whitespace(soup.title.string)
    if not title and (h1 := soup.find("h1")):
        title = _normalize_whitespace(h1.get_text(" ", strip=True))
    title = re.sub(r"\s*[|\-–]\s*BBVA.*$", "", title).strip() or page.url

    meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    description = _normalize_whitespace(meta_desc.get("content", "")) if meta_desc else ""

    headings = [
        _normalize_whitespace(h.get_text(" ", strip=True))
        for h in soup.find_all(["h1", "h2", "h3"])
    ]
    headings = [h for h in headings if h][:25]

    # Se ejecutan ambos extractores y se elige el más informativo por página.
    from_trafilatura = _extract_with_trafilatura(page.html, page.url)
    from_dom = _extract_with_soup(soup)
    text = (
        from_dom
        if len(from_dom) > len(from_trafilatura) * _DOM_PREFERENCE_RATIO
        else from_trafilatura or from_dom
    )
    logger.debug(
        "cleaner.extractor_chosen",
        url=page.url,
        trafilatura_chars=len(from_trafilatura),
        dom_chars=len(from_dom),
        chosen="dom" if text is from_dom else "trafilatura",
    )

    text = _drop_boilerplate_lines(_normalize_whitespace(text))

    if len(text) < _MIN_USEFUL_CHARS:
        raise ExtractionError(
            f"Contenido insuficiente en {page.url}",
            detail=f"Solo se extrajeron {len(text)} caracteres útiles",
        )

    # Encabezar con título y descripción mejora la recuperación: el chunk 0
    # queda auto-contenido y responde bien a preguntas del tipo "¿qué es X?".
    header = title if title else ""
    if description and description.lower() not in text.lower():
        header = f"{header}\n{description}" if header else description
    body = f"{header}\n\n{text}" if header else text

    lang_attr = (soup.html.get("lang") if soup.html else None) or "es"

    return CleanDocument(
        url=page.url,
        title=title,
        text=body,
        description=description,
        section=_guess_section(page.url),
        headings=headings,
        language=lang_attr.split("-")[0].lower(),
        scraped_at=page.fetched_at,
    )
