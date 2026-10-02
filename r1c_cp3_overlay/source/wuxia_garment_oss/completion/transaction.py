"""Atomic commit, guided wait, HOLD rejection, and rollback receipts."""
from __future__ import annotations

from .model import canonical_sha256
from .repair import apply_operation
from .snapshot import clone_snapshot, refresh_snapshot


def _blocked_receipt(candidate: dict, plan: dict, transaction_id: str, status: str) -> tuple[dict, dict]:
    receipt = {
        "contract": "CompletionTransactionReceipt/1",
        "transaction_id": transaction_id,
        "plan_sha256": plan["plan_sha256"],
        "status": status,
        "before_state_sha256": candidate["state_sha256"],
        "after_state_sha256": candidate["state_sha256"],
        "committed_operation_count": 0,
        "working_operation_count_before_failure": 0,
        "state_preserved": True,
        "partial_publication_count": 0,
        "rollback_performed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return candidate, receipt


def execute_transaction(
    candidate: dict,
    canonical: dict,
    plan: dict,
    transaction_id: str,
    guided_approved: bool = False,
    fail_after_operations: int | None = None,
) -> tuple[dict, dict]:
    if plan["status"] == "HOLD_TOPOLOGY_CHANGE_REQUIRED":
        return _blocked_receipt(candidate, plan, transaction_id, "REJECTED_HOLD")
    if plan["status"] == "GUIDED_DECISION_REQUIRED" and not guided_approved:
        return _blocked_receipt(candidate, plan, transaction_id, "AWAITING_GUIDED_APPROVAL")
    working = clone_snapshot(candidate, f"{candidate['snapshot_id']}_WORKING")
    applied = 0
    try:
        for operation in plan["operations"]:
            apply_operation(working, canonical, operation)
            applied += 1
            if fail_after_operations is not None and applied >= fail_after_operations:
                raise RuntimeError("INJECTED_TRANSACTION_FAILURE")
        working = refresh_snapshot(working, canonical)
        if working["state_sha256"] != canonical["state_sha256"]:
            raise RuntimeError("REPAIRED_STATE_DOES_NOT_MATCH_AUTHORITY")
    except Exception as error:
        receipt = {
            "contract": "CompletionTransactionReceipt/1",
            "transaction_id": transaction_id,
            "plan_sha256": plan["plan_sha256"],
            "status": "ROLLED_BACK_ATOMIC",
            "error": str(error),
            "before_state_sha256": candidate["state_sha256"],
            "after_state_sha256": candidate["state_sha256"],
            "committed_operation_count": 0,
            "working_operation_count_before_failure": applied,
            "state_preserved": True,
            "partial_publication_count": 0,
            "rollback_performed": True,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        return candidate, receipt
    committed = clone_snapshot(working, f"{candidate['snapshot_id']}_COMMITTED")
    status = "COMMITTED_GUIDED" if guided_approved else "COMMITTED_SAFE_AUTO"
    receipt = {
        "contract": "CompletionTransactionReceipt/1",
        "transaction_id": transaction_id,
        "plan_sha256": plan["plan_sha256"],
        "status": status,
        "before_state_sha256": candidate["state_sha256"],
        "after_state_sha256": committed["state_sha256"],
        "committed_operation_count": len(plan["operations"]),
        "working_operation_count_before_failure": 0,
        "state_preserved": False,
        "matches_canonical_authority": committed["state_sha256"] == canonical["state_sha256"],
        "partial_publication_count": 0,
        "rollback_performed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return committed, receipt
