"""Raw data ingestion (PHASE 4 — Phase 1: Creation feed).

Supports:
- text / dict / JSON / YAML / PDF
- normalizes input to a unified dict[str, Any]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def ingest_text(text: str, *, source: str = "<inline>") -> dict[str, Any]:
    """Wrap raw text as a normalized ingestion payload."""
    return {"source": source, "kind": "text", "content": text}


def ingest_json(path_or_str: str | Path, *, source: str | None = None) -> dict[str, Any]:
    """Load a JSON file or string."""
    if isinstance(path_or_str, Path) or (isinstance(path_or_str, str) and Path(path_or_str).is_file()):
        path = Path(path_or_str)
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"source": source or str(path), "kind": "json", "content": data}
    data = json.loads(path_or_str)
    return {"source": source or "<inline-json>", "kind": "json", "content": data}


def ingest_yaml(path_or_str: str | Path, *, source: str | None = None) -> dict[str, Any]:
    if isinstance(path_or_str, Path) or (isinstance(path_or_str, str) and Path(path_or_str).is_file()):
        path = Path(path_or_str)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {"source": source or str(path), "kind": "yaml", "content": data}
    return {"source": source or "<inline-yaml>", "kind": "yaml", "content": yaml.safe_load(path_or_str)}


def ingest_pdf(path: str | Path, *, source: str | None = None) -> dict[str, Any]:
    """Extract text from PDF using pypdf (optional dep)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires `pypdf`. Install via `pip install metadata-kg[pdf]`."
        ) from exc

    path = Path(path)
    reader = PdfReader(str(path))
    text_parts: list[str] = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning(f"PDF page extraction failed: {exc}")

    full_text = "\n\n".join(text_parts).strip()
    return {
        "source": source or str(path),
        "kind": "pdf",
        "content": full_text,
        "num_pages": len(reader.pages),
    }


def ingest_any(input: str | Path | dict[str, Any], *, source: str | None = None) -> dict[str, Any]:
    """Best-effort dispatcher based on type / file extension."""
    if isinstance(input, dict):
        return {"source": source or "<dict>", "kind": "dict", "content": input}

    if isinstance(input, (str, Path)):
        p = Path(input)
        if p.is_file():
            ext = p.suffix.lower()
            if ext == ".pdf":
                return ingest_pdf(p, source=source)
            if ext == ".json":
                return ingest_json(p, source=source)
            if ext in {".yaml", ".yml"}:
                return ingest_yaml(p, source=source)
            return ingest_text(p.read_text(encoding="utf-8"), source=source or str(p))
        # raw string
        return ingest_text(str(input), source=source or "<inline>")

    raise TypeError(f"Unsupported input type: {type(input)}")


__all__ = ["ingest_any", "ingest_json", "ingest_pdf", "ingest_text", "ingest_yaml"]
