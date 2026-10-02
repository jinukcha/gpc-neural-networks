from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/r1c_cp4_r2_rev1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_metric_decomposition_and_improvement() -> None:
    before = _load(BUILD / "metric_decomposition_before.json")
    after = _load(BUILD / "metric_decomposition_after.json")
    assert before["edge_count"] == 4014
    assert after["edge_count"] == 4014
    assert after["target_components"]["log_error_p95"] < before["target_components"]["log_error_p95"]
    assert after["target_components"]["log_error_p99"] < before["target_components"]["log_error_p99"]


def test_isometric_repair_is_bounded() -> None:
    receipt = _load(BUILD / "isometric_repair_receipt.json")
    assert receipt["source_pattern_changed"] is False
    assert receipt["post_settle_vertex_repair_count"] == 0
    assert receipt["target_edge_count"] > 0
    assert receipt["after"]["log_error_p95"] < receipt["before"]["log_error_p95"]


def test_direct_glb_and_validator() -> None:
    package = _load(BUILD / "direct_glb_package.json")
    validator = _load(BUILD / "khronos_gltf_validation_report.json")
    assert package["blender_used"] is False
    assert package["fresh_reopen"]["pass"] is True
    assert int(validator["issues"]["numErrors"]) == 0
    assert (ROOT / package["path"]).is_file()


def test_godot_same_camera_evidence() -> None:
    godot = _load(BUILD / "godot_consumer_receipt.json")
    evidence = _load(BUILD / "godot_visual_evidence_receipt.json")
    assert godot["consumer_pass"] is True
    assert evidence["same_camera_fixture"] is True
    assert evidence["required_view_count_per_state"] == 12
    assert evidence["all_views_reviewable"] is True


def test_terminal_scope_is_blender_free() -> None:
    receipt = _load(BUILD / "cp4_r2_rev1_receipt.json")
    assert receipt["technical_pass"] is True
    assert receipt["khronos_validator_pass"] is True
    assert receipt["godot_consumer_pass"] is True
    assert receipt["blender_executed"] is False
    assert receipt["blend_generated"] is False
    assert receipt["cp4_r1_predecessor_mutated"] is False
