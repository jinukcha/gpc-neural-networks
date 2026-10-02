"""Stable contracts and ordering rules for R1C completion and repair."""
from __future__ import annotations

import hashlib
import json


DISPOSITION_ORDER = {"SAFE_AUTO": 0, "GUIDED": 1, "HOLD": 2}


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_issue_id(category: str, owner_path: str) -> str:
    token = hashlib.sha256(f"{category}:{owner_path}".encode("utf-8")).hexdigest()[:12]
    return f"ISSUE_{category}_{token}"


def make_issue(
    category: str,
    owner_path: str,
    disposition: str,
    message: str,
    observed,
    expected,
    repair_targets: tuple[str, ...] = (),
    delta: float | None = None,
    budget: float | None = None,
) -> dict:
    if disposition not in DISPOSITION_ORDER:
        raise ValueError(f"invalid disposition: {disposition}")
    payload = {
        "issue_id": stable_issue_id(category, owner_path),
        "category": category,
        "owner_path": owner_path,
        "disposition": disposition,
        "message": message,
        "observed": observed,
        "expected": expected,
        "repair_targets": list(repair_targets),
        "delta": delta,
        "budget": budget,
    }
    payload["issue_sha256"] = canonical_sha256(payload)
    return payload


def strongest_disposition(values: list[str]) -> str:
    return max(values, key=lambda value: DISPOSITION_ORDER[value]) if values else "SAFE_AUTO"


def status_for_disposition(disposition: str, issue_count: int) -> str:
    if issue_count == 0:
        return "NOOP"
    return {
        "SAFE_AUTO": "READY_SAFE_AUTO",
        "GUIDED": "GUIDED_DECISION_REQUIRED",
        "HOLD": "HOLD_TOPOLOGY_CHANGE_REQUIRED",
    }[disposition]
