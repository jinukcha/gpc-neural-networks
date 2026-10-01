"""Canonical JSON save and fresh-process reopen helpers."""
from __future__ import annotations

import json
from pathlib import Path

from ..document.model import PatternDocument
from ..document.resolver import require_resolved


def save_pattern_document(path: Path, document: PatternDocument) -> str:
    require_resolved(document)
    payload = document.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return str(payload["document_sha256"])


def load_pattern_document(path: Path) -> PatternDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = str(payload.pop("document_sha256"))
    document = PatternDocument.from_dict(payload)
    actual = str(document.to_dict()["document_sha256"])
    if stored != actual:
        raise ValueError(f"PatternDocument hash mismatch: {stored} != {actual}")
    require_resolved(document)
    return document
