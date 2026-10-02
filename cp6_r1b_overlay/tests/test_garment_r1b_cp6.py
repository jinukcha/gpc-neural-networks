from __future__ import annotations

import json
from pathlib import Path

from wuxia_garment_oss.rig.runtime_product.lod import inspect_rigged_glb
from wuxia_garment_oss.rig.runtime_product.secondary_motion import product_profile


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/rig_cp6"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_secondary_motion_profiles_are_bounded_and_owned() -> None:
    tunic = product_profile("SLEEVED_TUNIC_RIGGED_R1B", "SLEEVED_TUNIC")
    robe = product_profile("STRAIGHT_SLEEVE_ROBE_RIGGED_R1B", "STRAIGHT_SLEEVE_ROBE")
    tunic.validate()
    robe.validate()
    assert len(tunic.domains) == 2
    assert len(robe.domains) == 3
    assert all(item.max_weight <= 0.72 for item in robe.domains)
    assert {item.owner_component for item in robe.domains} == {"LEFT_SLEEVE", "RIGHT_SLEEVE", "STRAIGHT_ROBE_SKIRT"}


def test_lod_products_preserve_skin_and_targets() -> None:
    for owner in ("sleeved_tunic", "straight_robe"):
        base = BUILD / "products" / owner
        metrics = tuple(inspect_rigged_glb(base / f"lod{index}.glb") for index in range(3))
        assert metrics[0]["vertex_count"] > metrics[1]["vertex_count"] > metrics[2]["vertex_count"]
        assert metrics[0]["triangle_count"] > metrics[1]["triangle_count"] > metrics[2]["triangle_count"]
        assert all(item["bone_count"] == 23 for item in metrics)
        assert all(item["zero_weight_vertex_count"] == 0 for item in metrics)
        assert all(item["cp5_target_names"] == metrics[0]["cp5_target_names"] for item in metrics)


def test_cp6_registry_uses_three_distance_bands() -> None:
    registry = _load(BUILD / "runtime_garment_lod_registry.json")
    assert registry["contract"] == "RuntimeGarmentLODRegistry/1"
    assert len(registry["products"]) == 2
    assert registry["distance_policy"]["near"]["lod"] == "LOD0"
    assert registry["distance_policy"]["mid"]["lod"] == "LOD1"
    assert registry["distance_policy"]["far"]["lod"] == "LOD2"
    assert registry["atomic_variant_swap"] is True


def test_exact_godot_runtime_preserves_state_across_lods() -> None:
    runtime = _load(BUILD / "godot_runtime_receipt.json")
    assert runtime["runtime_acceptance"] is True
    assert runtime["godot_version"]["string"].startswith("4.7.2-stable")
    for product in runtime["products"]:
        assert product["accepted"] is True
        assert product["secondary_motion_bounded"] is True
        assert product["secondary_state_transfer_pass"] is True
        assert product["corrective_and_secondary_names_preserved"] is True


def test_r1b_terminal_closeout_and_captures() -> None:
    receipt = _load(BUILD / "cp6_receipt.json")
    captures = _load(BUILD / "multi_distance_capture_receipt.json")
    assert receipt["r1b_complete"] is True
    assert receipt["terminal_decision"] == "GARMENT_CAD_PRO_R1B_COMPLETE"
    assert receipt["cp5_predecessor_mutated"] is False
    assert captures["capture_count"] == 6
    assert captures["all_captures_accepted"] is True
