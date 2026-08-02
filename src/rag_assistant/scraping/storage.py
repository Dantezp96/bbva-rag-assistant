"""Persistencia local de los datos scrapeados (crudos y limpios).

Requisito de la prueba: "Almacenar los datos scrapeados en local (Crudos y
limpios)". Layout resultante:

    data/
      raw/                      HTML íntegro, tal cual llegó
        <slug>.html
        _manifest.jsonl         una línea por descarga (url, status, hash, ms)
      clean/                    texto limpio + metadatos
        <slug>.json
        corpus.jsonl            corpus completo, formato de una línea por doc

Se guarda el HTML crudo porque permite **reprocesar la limpieza sin volver a
rastrear el sitio**: si mañana mejora el extractor, se regenera `clean/` en
segundos y sin volver a molestar al servidor de BBVA.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from rag_assistant.core.logging import get_logger
from rag_assistant.core.models import CleanDocument, RawPage

logger = get_logger(__name__)

RAW_MANIFEST = "_manifest.jsonl"
CLEAN_CORPUS = "corpus.jsonl"


class ScrapeStorage:
    """Lectura/escritura del corpus en disco."""

    def __init__(self, raw_dir: Path | str, clean_dir: Path | str) -> None:
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- crudos --
    def save_raw(self, page: RawPage) -> Path:
        path = self.raw_dir / f"{page.slug}.html"
        path.write_text(page.html, encoding="utf-8")
        self._append_jsonl(
            self.raw_dir / RAW_MANIFEST,
            {
                "url": page.url,
                "file": path.name,
                "status_code": page.status_code,
                "fetched_at": page.fetched_at.isoformat(),
                "depth": page.depth,
                "fetcher": page.fetcher,
                "elapsed_ms": page.elapsed_ms,
                "bytes": len(page.html),
                "content_hash": page.content_hash,
            },
        )
        return path

    def iter_raw(self) -> Iterator[RawPage]:
        """Recorre el HTML ya descargado (permite re-limpiar sin re-crawlear)."""
        manifest = self.raw_dir / RAW_MANIFEST
        if not manifest.exists():
            return
        for entry in self._read_jsonl(manifest):
            path = self.raw_dir / entry["file"]
            if not path.exists():
                continue
            yield RawPage(
                url=entry["url"],
                html=path.read_text(encoding="utf-8"),
                status_code=entry.get("status_code", 200),
                fetched_at=datetime.fromisoformat(entry["fetched_at"]),
                depth=entry.get("depth", 0),
                fetcher=entry.get("fetcher", "unknown"),
            )

    # ------------------------------------------------------------ limpios --
    def save_clean(self, doc: CleanDocument) -> Path:
        slug = RawPage(url=doc.url, html="", status_code=200).slug
        path = self.clean_dir / f"{slug}.json"
        path.write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._append_jsonl(self.clean_dir / CLEAN_CORPUS, doc.to_dict())
        return path

    def load_clean(self) -> list[CleanDocument]:
        """Carga el corpus limpio, deduplicando por URL (gana la última versión)."""
        corpus = self.clean_dir / CLEAN_CORPUS
        by_url: dict[str, CleanDocument] = {}

        entries = (
            self._read_jsonl(corpus)
            if corpus.exists()
            else (
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(self.clean_dir.glob("*.json"))
            )
        )
        for entry in entries:
            by_url[entry["url"]] = CleanDocument(
                url=entry["url"],
                title=entry.get("title", ""),
                text=entry.get("text", ""),
                description=entry.get("description", ""),
                section=entry.get("section", ""),
                headings=entry.get("headings", []),
                language=entry.get("language", "es"),
                scraped_at=datetime.fromisoformat(
                    entry.get("scraped_at", datetime.now(UTC).isoformat())
                ),
                word_count=entry.get("word_count", 0),
                content_hash=entry.get("content_hash", ""),
            )
        return list(by_url.values())

    # ------------------------------------------------------------ utilidad --
    def reset(self, *, raw: bool = True, clean: bool = True) -> None:
        """Vacía los directorios de datos (usado por `--fresh` en la ingesta)."""
        targets = []
        if raw:
            targets.append(self.raw_dir)
        if clean:
            targets.append(self.clean_dir)
        for directory in targets:
            for path in directory.iterdir():
                if path.is_file() and path.name != ".gitkeep":
                    path.unlink()
        logger.info("storage.reset", raw=raw, clean=clean)

    def stats(self) -> dict[str, int]:
        raw_files = len(list(self.raw_dir.glob("*.html")))
        clean_files = len(list(self.clean_dir.glob("*.json")))
        return {"raw_pages": raw_files, "clean_documents": clean_files}

    @staticmethod
    def _append_jsonl(path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[dict]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
