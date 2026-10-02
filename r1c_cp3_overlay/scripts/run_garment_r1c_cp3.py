#!/usr/bin/env python3
"""Build CP3 completion diagnoses, repair plans, previews, and transactions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wuxia_garment_oss.completion.contracts import cp3_schemas
from wuxia_garment_oss.completion.diagnosis import (
    GUIDED_GEOMETRY_BUDGET_M,
    GUIDED_SEAM_RATIO_DELTA,
    SAFE_GEOMETRY_BUDGET_M,
    SAFE_SEAM_RATIO_DELTA,
    diagnose_completion,
)
from wuxia_garment_oss.completion.evidence import render_cp3_evidence
from wuxia_garment_oss.completion.fixtures import guided_candidate, hold_candidate, safe_auto_candidate
from wuxia_garment_oss.completion.model import canonical_sha256
from wuxia_garment_oss.completion.planning import build_repair_plan
from wuxia_garment_oss.completion.preview import build_repair_preview
from wuxia_garment_oss.completion.snapshot import load_cp2_snapshot, write_json
from wuxia_garment_oss.completion.transaction import execute_transaction


BUILD_REL = Path("build/r1c_cp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def publish_schemas(root: Path) -> None:
    for name, schema in cp3_schemas().items():
        write_json(root / "contracts/r1c_cp3" / name, schema)


def _case(canonical: dict, candidate: dict, name: str) -> tuple[dict, dict, dict]:
    report = diagnose_completion(canonical, candidate, f"R1C_CP3_{name.upper()}_DIAGNOSIS")
    plan = build_repair_plan(report, f"R1C_CP3_{name.upper()}_PLAN")
    preview = build_repair_preview(candidate, canonical, plan, f"R1C_CP3_{name.upper()}_PREVIEW")
    return report, plan, preview


def _publish_case(build: Path, name: str, candidate: dict, report: dict, plan: dict, preview: dict) -> None:
    write_json(build / "fixtures" / f"{name}_candidate_snapshot.json", candidate)
    write_json(build / "diagnosis" / f"{name}.json", report)
    write_json(build / "plans" / f"{name}.json", plan)
    write_json(build / "previews" / f"{name}.json", preview)


def completion_policy() -> dict:
    payload = {
        "contract": "CompletionRepairPolicy/1",
        "modes": ["STRICT", "GUIDED", "SAFE_AUTO"],
        "safe_geometry_budget_m": SAFE_GEOMETRY_BUDGET_M,
        "guided_geometry_budget_m": GUIDED_GEOMETRY_BUDGET_M,
        "safe_seam_ratio_delta": SAFE_SEAM_RATIO_DELTA,
        "guided_seam_ratio_delta": GUIDED_SEAM_RATIO_DELTA,
        "safe_completion": ["MIRROR_COMPONENT", "MISSING_INTERFACE", "MISSING_PARAMETER", "MISSING_NOTCH"],
        "bounded_repair": ["TANGENT_DRIFT", "ENDPOINT_DRIFT", "SEAM_LENGTH_MISMATCH"],
        "topology_auto_commit": False,
        "final_mesh_repair": "FORBIDDEN",
    }
    payload["policy_sha256"] = canonical_sha256(payload)
    return payload


def preliminary_receipt(
    safe_report: dict,
    safe_plan: dict,
    safe_tx: dict,
    guided_report: dict,
    guided_tx: dict,
    hold_report: dict,
    hold_tx: dict,
    rollback_tx: dict,
) -> dict:
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP3",
        "phase": "BUILT_PENDING_FRESH_PROCESS_REOPEN",
        "safe_issue_count": safe_report["issue_count"],
        "safe_operation_count": safe_plan["operation_count"],
        "guided_issue_count": guided_report["issue_count"],
        "hold_issue_count": hold_report["issue_count"],
        "safe_auto_commit_pass": safe_tx["status"] == "COMMITTED_SAFE_AUTO" and safe_tx.get("matches_canonical_authority") is True,
        "guided_wait_pass": guided_tx["status"] == "AWAITING_GUIDED_APPROVAL" and guided_tx["state_preserved"],
        "topology_hold_pass": hold_tx["status"] == "REJECTED_HOLD" and hold_tx["state_preserved"],
        "rollback_pass": rollback_tx["status"] == "ROLLED_BACK_ATOMIC" and rollback_tx["state_preserved"],
        "partial_publication_count": sum(
            item["partial_publication_count"]
            for item in (safe_tx, guided_tx, hold_tx, rollback_tx)
        ),
        "writer_process_id": os.getpid(),
        "triangulation_executed": False,
        "simulation_executed": False,
        "godot_executed": False,
        "r1c_cp2_predecessor_mutated": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    publish_schemas(root)
    canonical = load_cp2_snapshot(root)
    safe_candidate = safe_auto_candidate(canonical)
    guided = guided_candidate(canonical)
    hold = hold_candidate(canonical)
    safe_report, safe_plan, safe_preview = _case(canonical, safe_candidate, "safe_auto")
    guided_report, guided_plan, guided_preview = _case(canonical, guided, "guided")
    hold_report, hold_plan, hold_preview = _case(canonical, hold, "hold")
    repaired, safe_tx = execute_transaction(safe_candidate, canonical, safe_plan, "R1C_CP3_SAFE_COMMIT")
    _, guided_tx = execute_transaction(guided, canonical, guided_plan, "R1C_CP3_GUIDED_WAIT")
    _, hold_tx = execute_transaction(hold, canonical, hold_plan, "R1C_CP3_HOLD_REJECT")
    _, rollback_tx = execute_transaction(
        safe_candidate,
        canonical,
        safe_plan,
        "R1C_CP3_FORCED_ROLLBACK",
        fail_after_operations=2,
    )
    cases = (
        ("safe_auto", safe_candidate, safe_report, safe_plan, safe_preview),
        ("guided", guided, guided_report, guided_plan, guided_preview),
        ("hold", hold, hold_report, hold_plan, hold_preview),
    )
    for item in cases:
        _publish_case(build, *item)
    transactions = {
        "safe_commit": safe_tx,
        "guided_wait": guided_tx,
        "topology_hold": hold_tx,
        "forced_rollback": rollback_tx,
    }
    for name, payload in transactions.items():
        write_json(build / "transactions" / f"{name}.json", payload)
    write_json(build / "canonical_snapshot.json", canonical)
    write_json(build / "repaired/repaired_pattern_snapshot.json", repaired)
    write_json(build / "repaired/repaired_assembled_pattern_package.json", repaired["assembled_package"])
    write_json(build / "completion_repair_policy.json", completion_policy())
    receipt = preliminary_receipt(
        safe_report,
        safe_plan,
        safe_tx,
        guided_report,
        guided_tx,
        hold_report,
        hold_tx,
        rollback_tx,
    )
    write_json(build / "cp3_preliminary_receipt.json", receipt)
    render_cp3_evidence(
        build / "cp3_completion_repair_evidence.png",
        safe_candidate,
        repaired,
        safe_report,
        safe_plan,
        transactions,
        {"safe": safe_preview, "guided": guided_preview, "hold": hold_preview},
        receipt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
