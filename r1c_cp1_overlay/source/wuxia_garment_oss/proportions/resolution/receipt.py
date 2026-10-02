"""Canonical hashing and immutable JSON verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HASH_FIELDS = {
    "context_sha256",
    "resolved_set_sha256",
    "receipt_sha256",
    "definition_set_sha256",
}


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def payload_without_hash(payload: dict, hash_field: str) -> dict:
    result = dict(payload)
    result.pop(hash_field, None)
    return result


def verify_embedded_hash(payload: dict, hash_field: str) -> bool:
    recorded = payload.get(hash_field)
    return isinstance(recorded, str) and recorded == canonical_sha256(payload_without_hash(payload, hash_field))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_verified_json(path: Path, hash_field: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not verify_embedded_hash(payload, hash_field):
        raise ValueError(f"canonical hash mismatch: {path}")
    return payload
