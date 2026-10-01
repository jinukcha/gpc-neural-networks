from __future__ import annotations

import json
from pathlib import Path

from wuxia_garment_oss.rig.library.compiler import compile_outfit
from wuxia_garment_oss.rig.library.model import (
    CoverageRegion,
    GarmentFamilyEntry,
    GarmentLibraryRegistry,
)
from wuxia_garment_oss.rig.runtime.outfit_transaction import OutfitRuntimeState, commit_outfit


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/rig_cp4"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_registry() -> GarmentLibraryRegistry:
    required = ("ROOT", "PELVIS")
    top = GarmentFamilyEntry(
        "TOP", "TOP_FAMILY", "top.glb", "0" * 64, "MID", ("UPPER",),
        (CoverageRegion("TORSO", 0.004, True),), required,
        exclusive_slots=("UPPER_PRIMARY",),
    )
    bottom = GarmentFamilyEntry(
        "BOTTOM", "BOTTOM_FAMILY", "bottom.glb", "1" * 64, "BASE", ("LOWER",),
        (CoverageRegion("PELVIS", 0.004, True),), required,
        exclusive_slots=("LOWER_PRIMARY",),
    )
    blocker = GarmentFamilyEntry(
        "BLOCKER", "BLOCKER_FAMILY", "top.glb", "0" * 64, "MID", ("UPPER",),
        (CoverageRegion("TORSO", 0.012, True),), required,
        incompatible_families=("TOP_FAMILY",), exclusive_slots=("UPPER_PRIMARY",),
    )
    return GarmentLibraryRegistry(
        "TEST", (top, bottom, blocker),
        ("BASE", "INNER", "MID", "OUTER", "ARMOR", "ACCESSORY"),
        {"TORSO": 0.010, "PELVIS": 0.010},
    )


def test_compiler_accepts_compatible_two_piece() -> None:
    plan = compile_outfit(_fixture_registry(), "OK", "RIG", ("TOP", "BOTTOM"))
    assert plan.status == "ACCEPTED"
    assert plan.layer_order == ("BOTTOM", "TOP")
    assert plan.region_thickness_m == {"TORSO": 0.004, "PELVIS": 0.004}


def test_compiler_rejects_incompatible_without_partial_state() -> None:
    registry = _fixture_registry()
    accepted = compile_outfit(registry, "OK", "RIG", ("TOP", "BOTTOM"))
    rejected = compile_outfit(registry, "BAD", "RIG", ("TOP", "BLOCKER"))
    state = OutfitRuntimeState()
    first = commit_outfit(state, accepted, "mask")
    before = state.snapshot()
    second = commit_outfit(state, rejected, "mask")
    assert first["status"] == "COMMITTED"
    assert second["status"] == "REJECTED_ATOMIC"
    assert state.snapshot() == before


def test_published_registry_and_plans() -> None:
    registry = _load(BUILD / "garment_library_registry.json")
    accepted = _load(BUILD / "outfits/reference_two_piece.json")
    rejected = _load(BUILD / "outfits/incompatible_duplicate_tunic.json")
    assert registry["contract"] == "GarmentLibraryRegistry/1"
    assert len(registry["entries"]) == 3
    assert accepted["status"] == "ACCEPTED"
    assert accepted["layer_order"] == ["TROUSERS_RIGGED_R1B", "SLEEVELESS_TUNIC_RIGGED_R1B"]
    assert rejected["status"] == "REJECTED_ATOMIC"
    assert rejected["rejection_reasons"]


def test_body_occlusion_and_transactions_pass() -> None:
    mask = _load(BUILD / "body_occlusion/body_hide_mask.json")
    transactions = _load(BUILD / "outfit_transaction_suite.json")
    assert mask["mask_pass"] is True
    assert 0 < mask["hidden_triangle_count"] < mask["source_triangle_count"]
    assert transactions["compatible_commit_pass"] is True
    assert transactions["incompatible_rejection_pass"] is True
    assert transactions["rejected_state_preserved"] is True
    assert transactions["unequip_pass"] is True


def test_terminal_and_godot_consumer_pass() -> None:
    receipt = _load(BUILD / "cp4_receipt.json")
    runtime = _load(BUILD / "godot_product/godot_outfit_runtime_receipt.json")
    assert receipt["terminal_decision"] == "CP4_COMPLETE_MULTI_GARMENT_OUTFIT"
    assert receipt["cp4_acceptance"] is True
    assert receipt["godot_outfit_consumer_pass"] is True
    assert runtime["consumer_pass"] is True
    assert runtime["equipped_garment_count"] == 2
    assert runtime["atomic_rejection_pass"] is True
    assert runtime["secondary_motion_executed"] is False
    assert runtime["rig_aware_lod_executed"] is False
