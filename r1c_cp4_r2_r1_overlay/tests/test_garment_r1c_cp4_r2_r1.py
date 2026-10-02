from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/r1c_cp4_r2_r1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_metric_decomposition_covers_all_structural_edges() -> None:
    before = _load(BUILD / "metric_decomposition_before.json")
    after = _load(BUILD / "metric_decomposition_after.json")
    assert before["edge_count"] == after["edge_count"] == 4014
    assert "sleeve_left" in before["by_component"]
    assert any("BOUNDARY" in key for key in before["by_component_region"])


def test_metric_fidelity_improves_raw_p95_and_p99() -> None:
    receipt = _load(BUILD / "isometric_arrangement_repair.json")
    assert receipt["global_metric_improved"] is True
    assert receipt["after"]["global_p95"] < receipt["before"]["global_p95"]
    assert receipt["after"]["global_p99"] < receipt["before"]["global_p99"]
    assert receipt["maximum_displacement_m"] <= 0.0180001


def test_bodice_rest_is_immutable() -> None:
    with np.load(ROOT / "build/r1c_cp4_r1/compiled_materialization.npz") as old:
        before = np.asarray(old["positions_rest"])
        component_ids = np.asarray(old["component_ids"])
    with np.load(BUILD / "metric_fidelity_materialization.npz") as new:
        after = np.asarray(new["positions_rest"])
    offsets = _load(ROOT / "build/r1c_cp4_r1/component_offsets.json")
    order = offsets["component_order"]
    bodice_ids = [index for index, name in enumerate(order) if name.startswith("bodice")]
    mask = np.isin(component_ids, bodice_ids)
    assert np.array_equal(before[mask], after[mask])


def test_technical_candidate_passes_without_vertex_repair() -> None:
    qualification = _load(BUILD / "technical_qualification_receipt.json")
    settling = _load(BUILD / "warp_settling_receipt.json")
    assert qualification["technical_pass"] is True
    assert qualification["seam"]["p95_m"] <= 0.004
    assert settling["post_settle_vertex_repair_count"] == 0


def test_before_after_evidence_is_pending_direct_audit() -> None:
    evidence = _load(BUILD / "before_after_evidence_receipt.json")
    candidate = _load(BUILD / "cp4_r2_r1_candidate_receipt.json")
    assert evidence["view_count"] == 12
    assert evidence["same_camera_definition"] is True
    assert evidence["direct_art_audit"] == "PENDING_MODEL_REVIEW"
    assert candidate["product_acceptance"] is False
