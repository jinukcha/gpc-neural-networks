"""Before/after projection for repair plans without committing source state."""
from __future__ import annotations

from .model import canonical_sha256
from .repair import apply_operations


def build_repair_preview(candidate: dict, canonical: dict, plan: dict, preview_id: str) -> dict:
    projected = None
    projected_match = False
    if plan["status"] != "HOLD_TOPOLOGY_CHANGE_REQUIRED":
        projected = apply_operations(
            candidate,
            canonical,
            plan["operations"],
            f"{candidate['snapshot_id']}_PROJECTED",
        )
        projected_match = projected["state_sha256"] == canonical["state_sha256"]
    changed_paths = sorted({item["target_path"] for item in plan["operations"]})
    payload = {
        "contract": "RepairPreview/1",
        "preview_id": preview_id,
        "plan_sha256": plan["plan_sha256"],
        "before_snapshot_sha256": candidate["snapshot_sha256"],
        "before_state_sha256": candidate["state_sha256"],
        "projected_after_snapshot_sha256": projected["snapshot_sha256"] if projected else None,
        "projected_after_state_sha256": projected["state_sha256"] if projected else None,
        "projected_matches_canonical": projected_match,
        "operation_count": plan["operation_count"],
        "changed_owner_paths": changed_paths,
        "requires_user_approval": plan["status"] == "GUIDED_DECISION_REQUIRED",
        "blocked_by_topology_change": plan["status"] == "HOLD_TOPOLOGY_CHANGE_REQUIRED",
        "source_state_mutated": False,
    }
    payload["preview_sha256"] = canonical_sha256(payload)
    return payload
