from __future__ import annotations

import json
from pathlib import Path

from wuxia_garment_oss.completion.diagnosis import diagnose_completion
from wuxia_garment_oss.completion.fixtures import guided_candidate, hold_candidate, safe_auto_candidate
from wuxia_garment_oss.completion.planning import build_repair_plan
from wuxia_garment_oss.completion.snapshot import load_cp2_snapshot
from wuxia_garment_oss.completion.transaction import execute_transaction


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/r1c_cp3"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_safe_diagnosis_covers_completion_and_repair_classes() -> None:
    canonical = load_cp2_snapshot(ROOT)
    candidate = safe_auto_candidate(canonical)
    report = diagnose_completion(canonical, candidate, "TEST_SAFE")
    categories = {item["category"] for item in report["issues"]}
    assert {
        "MISSING_COMPONENT",
        "MISSING_INTERFACE",
        "MISSING_PARAMETER",
        "MISSING_NOTCH",
        "TANGENT_DRIFT",
        "ENDPOINT_DRIFT",
        "SEAM_LENGTH_MISMATCH",
    }.issubset(categories)
    assert report["strongest_disposition"] == "SAFE_AUTO"


def test_safe_plan_commits_to_canonical_state() -> None:
    canonical = load_cp2_snapshot(ROOT)
    candidate = safe_auto_candidate(canonical)
    report = diagnose_completion(canonical, candidate, "TEST_SAFE")
    plan = build_repair_plan(report, "TEST_SAFE_PLAN")
    repaired, receipt = execute_transaction(candidate, canonical, plan, "TEST_SAFE_TX")
    assert plan["status"] == "READY_SAFE_AUTO"
    assert receipt["status"] == "COMMITTED_SAFE_AUTO"
    assert repaired["state_sha256"] == canonical["state_sha256"]


def test_guided_candidate_waits_without_state_change() -> None:
    canonical = load_cp2_snapshot(ROOT)
    candidate = guided_candidate(canonical)
    report = diagnose_completion(canonical, candidate, "TEST_GUIDED")
    plan = build_repair_plan(report, "TEST_GUIDED_PLAN")
    result, receipt = execute_transaction(candidate, canonical, plan, "TEST_GUIDED_TX")
    assert plan["status"] == "GUIDED_DECISION_REQUIRED"
    assert receipt["status"] == "AWAITING_GUIDED_APPROVAL"
    assert result["state_sha256"] == candidate["state_sha256"]


def test_topology_hold_rejects_non_mirror_component_completion() -> None:
    canonical = load_cp2_snapshot(ROOT)
    candidate = hold_candidate(canonical)
    report = diagnose_completion(canonical, candidate, "TEST_HOLD")
    plan = build_repair_plan(report, "TEST_HOLD_PLAN")
    result, receipt = execute_transaction(candidate, canonical, plan, "TEST_HOLD_TX")
    assert plan["status"] == "HOLD_TOPOLOGY_CHANGE_REQUIRED"
    assert receipt["status"] == "REJECTED_HOLD"
    assert result["state_sha256"] == candidate["state_sha256"]


def test_injected_failure_rolls_back_atomically() -> None:
    canonical = load_cp2_snapshot(ROOT)
    candidate = safe_auto_candidate(canonical)
    report = diagnose_completion(canonical, candidate, "TEST_ROLLBACK")
    plan = build_repair_plan(report, "TEST_ROLLBACK_PLAN")
    result, receipt = execute_transaction(
        candidate, canonical, plan, "TEST_ROLLBACK_TX", fail_after_operations=2
    )
    assert receipt["status"] == "ROLLED_BACK_ATOMIC"
    assert receipt["partial_publication_count"] == 0
    assert result["state_sha256"] == candidate["state_sha256"]


def test_published_terminal_receipt() -> None:
    receipt = _load(BUILD / "cp3_receipt.json")
    assert receipt["terminal_decision"] == "CP3_COMPLETE_COMPLETION_REPAIR_TRANSACTION"
    assert receipt["cp3_acceptance"] is True
    assert receipt["partial_publication_count"] == 0
    assert receipt["triangulation_executed"] is False
    assert receipt["simulation_executed"] is False
