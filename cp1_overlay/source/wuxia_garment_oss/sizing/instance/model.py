"""Canonical CP1 selection receipt."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SelectionReceipt:
    request_id: str
    mode: str
    selected_size_id: str
    recommended_size_id: str
    shape_block: str
    height_block: str
    admission: str
    ranked_sizes: list[dict]
    grade_plan: list[dict]
    custom_alteration_plan: list[dict]
    residuals_m: dict[str, float]
    warnings: list[str]
    block_evidence: dict[str, float]

    def to_dict(self) -> dict:
        payload = {
            "contract": "GarmentSelectionReceipt/1",
            **asdict(self),
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        return payload
