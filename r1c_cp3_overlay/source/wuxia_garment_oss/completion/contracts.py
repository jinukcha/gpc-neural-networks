"""JSON Schema products for R1C CP3 completion and repair authority."""
from __future__ import annotations


DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _base(title: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": DRAFT,
        "title": title,
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": True,
    }


def cp3_schemas() -> dict[str, dict]:
    snapshot = _base(
        "PatternCompletionSnapshot/1",
        ["contract", "snapshot_id", "geometry_by_instance", "assembled_package", "state_sha256", "snapshot_sha256"],
        {
            "contract": {"const": "PatternCompletionSnapshot/1"},
            "snapshot_id": {"type": "string", "minLength": 1},
            "geometry_by_instance": {"type": "object", "minProperties": 1},
            "assembled_package": {"type": "object"},
            "resolved_parameter_set": {"type": "object"},
            "state_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "snapshot_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    )
    diagnosis = _base(
        "CompletionDiagnosisReport/1",
        ["contract", "diagnosis_id", "issue_count", "issues", "strongest_disposition", "report_sha256"],
        {
            "contract": {"const": "CompletionDiagnosisReport/1"},
            "diagnosis_id": {"type": "string"},
            "issue_count": {"type": "integer", "minimum": 0},
            "issues": {"type": "array", "items": {"type": "object"}},
            "strongest_disposition": {"enum": ["SAFE_AUTO", "GUIDED", "HOLD"]},
            "report_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    )
    plan = _base(
        "RepairPlan/1",
        ["contract", "plan_id", "status", "operation_count", "operations", "plan_sha256"],
        {
            "contract": {"const": "RepairPlan/1"},
            "plan_id": {"type": "string"},
            "status": {"enum": ["NOOP", "READY_SAFE_AUTO", "GUIDED_DECISION_REQUIRED", "HOLD_TOPOLOGY_CHANGE_REQUIRED"]},
            "operation_count": {"type": "integer", "minimum": 0},
            "operations": {"type": "array", "items": {"type": "object"}},
            "plan_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    )
    preview = _base(
        "RepairPreview/1",
        ["contract", "preview_id", "plan_sha256", "before_state_sha256", "source_state_mutated", "preview_sha256"],
        {
            "contract": {"const": "RepairPreview/1"},
            "preview_id": {"type": "string"},
            "plan_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "before_state_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "projected_after_state_sha256": {"type": ["string", "null"]},
            "source_state_mutated": {"const": False},
            "preview_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    )
    transaction = _base(
        "CompletionTransactionReceipt/1",
        ["contract", "transaction_id", "status", "before_state_sha256", "after_state_sha256", "partial_publication_count", "receipt_sha256"],
        {
            "contract": {"const": "CompletionTransactionReceipt/1"},
            "transaction_id": {"type": "string"},
            "status": {"enum": ["COMMITTED_SAFE_AUTO", "COMMITTED_GUIDED", "AWAITING_GUIDED_APPROVAL", "REJECTED_HOLD", "ROLLED_BACK_ATOMIC"]},
            "before_state_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "after_state_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "partial_publication_count": {"const": 0},
            "receipt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    )
    return {
        "pattern_completion_snapshot.schema.json": snapshot,
        "completion_diagnosis_report.schema.json": diagnosis,
        "repair_plan.schema.json": plan,
        "repair_preview.schema.json": preview,
        "completion_transaction_receipt.schema.json": transaction,
    }
