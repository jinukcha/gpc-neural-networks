"""Build deterministic completion and repair plans from diagnosis reports."""
from __future__ import annotations

from .model import canonical_sha256, status_for_disposition, strongest_disposition


_CATEGORY_TO_OPERATION = {
    "MISSING_COMPONENT": "RESTORE_COMPONENT",
    "MISSING_INTERFACE": "RESTORE_SEAM",
    "MISSING_PARAMETER": "RESTORE_SEAM_FIELD",
    "MISSING_NOTCH": "RESTORE_NOTCH",
    "TANGENT_DRIFT": "RESTORE_SEGMENT",
    "ENDPOINT_DRIFT": "RESTORE_SEGMENT",
    "SEAM_LENGTH_MISMATCH": "RESTORE_SEGMENT",
}


def _operation_key(kind: str, target: str) -> str:
    return f"{kind}:{target}"


def _operation_id(kind: str, target: str) -> str:
    token = canonical_sha256({"kind": kind, "target": target})[:12]
    return f"OP_{kind}_{token}"


def build_repair_plan(report: dict, plan_id: str) -> dict:
    operations: dict[str, dict] = {}
    dispositions = []
    for issue in report["issues"]:
        disposition = issue["disposition"]
        dispositions.append(disposition)
        kind = _CATEGORY_TO_OPERATION.get(issue["category"])
        if kind is None or disposition == "HOLD":
            continue
        for target in issue["repair_targets"]:
            key = _operation_key(kind, target)
            entry = operations.get(key)
            if entry is None:
                operations[key] = {
                    "operation_id": _operation_id(kind, target),
                    "operation_kind": kind,
                    "target_path": target,
                    "source_path": f"canonical/{target}",
                    "disposition": disposition,
                    "issue_ids": [issue["issue_id"]],
                    "maximum_delta": issue.get("delta"),
                    "budget": issue.get("budget"),
                }
                continue
            entry["issue_ids"].append(issue["issue_id"])
            entry["disposition"] = strongest_disposition([entry["disposition"], disposition])
            if issue.get("delta") is not None:
                previous = entry.get("maximum_delta") or 0.0
                entry["maximum_delta"] = max(previous, issue["delta"])
    strongest = strongest_disposition(dispositions)
    ordered = sorted(operations.values(), key=lambda item: (item["target_path"], item["operation_kind"]))
    for operation in ordered:
        operation["issue_ids"] = sorted(set(operation["issue_ids"]))
    payload = {
        "contract": "RepairPlan/1",
        "plan_id": plan_id,
        "diagnosis_report_sha256": report["report_sha256"],
        "candidate_state_sha256": report["candidate_state_sha256"],
        "canonical_state_sha256": report["canonical_state_sha256"],
        "status": status_for_disposition(strongest, report["issue_count"]),
        "strongest_disposition": strongest,
        "operation_count": len(ordered),
        "operations": ordered,
        "hold_issue_ids": sorted(
            item["issue_id"] for item in report["issues"] if item["disposition"] == "HOLD"
        ),
        "guided_issue_ids": sorted(
            item["issue_id"] for item in report["issues"] if item["disposition"] == "GUIDED"
        ),
        "atomic_commit_required": True,
        "topology_change_auto_commit": False,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload
