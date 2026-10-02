from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.rig.sleeve_generalization.construction import build_sleeve
from wuxia_garment_oss.rig.sleeve_generalization.measurements import load_arm_measurements
from wuxia_garment_oss.rig.sleeve_generalization.weights import sleeve_weight_field


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/rig_cp5"
SKELETON = ROOT / "build/rig_cp0/canonical_skeleton.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _joints() -> dict[str, np.ndarray]:
    payload = _load(SKELETON)
    return {name: np.asarray(value, dtype=np.float64) for name, value in payload["global_joint_positions"].items()}


def test_direct_measurements_follow_joint_chain() -> None:
    measurement, datums = load_arm_measurements(SKELETON)
    assert len(datums) == 6
    assert abs(measurement.sleeve_length_m - measurement.shoulder_to_elbow_m - measurement.elbow_to_wrist_m) < 1.0e-9
    assert measurement.cap_ease_ratio == 1.055


def test_sleeve_construction_publishes_cap_and_underarm() -> None:
    measurement, _ = load_arm_measurements(SKELETON)
    sleeve = build_sleeve("LEFT", _joints(), measurement, "FITTED_TUNIC")
    assert len(sleeve.positions) == 21 * 28
    assert len(sleeve.indices) == 20 * 28 * 2
    assert sleeve.metadata["sleeve_cap_curve_length_m"] > sleeve.metadata["armhole_curve_length_m"]
    assert sleeve.metadata["underarm_seam_vertex_modulo"] >= 0


def test_upper_arm_forearm_weights_are_normalized() -> None:
    measurement, _ = load_arm_measurements(SKELETON)
    sleeve = build_sleeve("RIGHT", _joints(), measurement, "STRAIGHT_ROBE")
    joints, weights, receipt = sleeve_weight_field(sleeve)
    assert joints.shape == weights.shape == (len(sleeve.positions), 4)
    assert np.max(np.abs(np.sum(weights, axis=1) - 1.0)) <= 1.0e-6
    assert receipt["zero_weight_vertex_count"] == 0
    assert "R_UPPER_ARM" in receipt["semantic_bones"]
    assert "R_FOREARM" in receipt["semantic_bones"]


def test_published_products_registry_and_outfits() -> None:
    tunic = _load(BUILD / "sleeved_tunic/rigged_garment_product.json")
    robe = _load(BUILD / "straight_sleeve_robe/rigged_garment_product.json")
    registry = _load(BUILD / "garment_library_registry.json")
    accepted = _load(BUILD / "outfits/sleeved_tunic_two_piece.json")
    rejected = _load(BUILD / "outfits/incompatible_double_upper.json")
    assert tunic["component_count"] == 2
    assert robe["component_count"] == 3
    assert len(registry["entries"]) == 5
    assert accepted["status"] == "ACCEPTED"
    assert rejected["status"] == "REJECTED_ATOMIC"
    assert rejected["rejection_reasons"]


def test_terminal_receipt_and_godot_runtime() -> None:
    receipt = _load(BUILD / "cp5_receipt.json")
    runtime = _load(BUILD / "godot_product/godot_sleeve_runtime_receipt.json")
    assert receipt["cp5_acceptance"] is True
    assert receipt["terminal_decision"] == "CP5_COMPLETE_SLEEVE_RIG_GENERALIZATION"
    assert receipt["secondary_motion_executed"] is False
    assert receipt["rig_aware_lod_executed"] is False
    assert runtime["consumer_pass"] is True
    assert runtime["sleeved_tunic"]["max_blend_shape_count"] >= 3
    assert runtime["straight_robe"]["max_blend_shape_count"] >= 3
