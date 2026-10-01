from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/rig_cp3"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cp3_terminal_acceptance() -> None:
    receipt = _load(BUILD / "cp3_receipt.json")
    assert receipt["terminal_decision"] == "CP3_COMPLETE_GODOT_EQUIP_PRODUCT"
    assert receipt["cp3_scope_acceptance"] is True
    assert receipt["glb_fresh_reopen_pass"] is True
    assert receipt["godot_consumer_pass"] is True
    assert receipt["atomic_equip_pass"] is True
    assert receipt["character_swap_pass"] is True
    assert receipt["failed_swap_preserved"] is True


def test_rigged_products_have_canonical_skin() -> None:
    for garment, primitive_count in (("tunic", 14), ("trousers", 3)):
        product = _load(BUILD / garment / "rigged_garment_product.json")
        assert product["contract"] == "RiggedGarmentProduct/1"
        assert product["bone_count"] == 23
        assert product["skin_count"] == 1
        assert product["primitive_count"] == primitive_count
        assert product["inverse_bind_matrix_count"] == 23
        assert product["zero_weight_vertex_count"] == 0
        assert product["maximum_weight_sum_error"] <= 2.0e-6


def test_godot_atomic_transactions() -> None:
    receipt = _load(BUILD / "godot_product/godot_equip_receipt.json")
    results = [row["result"] for row in receipt["transactions"]]
    assert results == [
        "EQUIPPED",
        "SLOT_OCCUPIED",
        "EQUIPPED",
        "CHARACTER_SWAPPED",
        "MISSING_REQUIRED_BONE",
        "UNEQUIPPED",
        "EQUIPPED",
    ]
    assert receipt["final_state_count"] == 2
    assert receipt["tunic"]["skin_surfaces"] == 14
    assert receipt["trousers"]["skin_surfaces"] == 3


def test_scope_boundaries_and_recovery_provenance() -> None:
    receipt = _load(BUILD / "cp3_receipt.json")
    provenance = _load(BUILD / "restored_input_provenance.json")
    assert receipt["secondary_motion_executed"] is False
    assert receipt["lod_executed"] is False
    assert receipt["outfit_composition_executed"] is False
    assert receipt["mesh_scaling"] == "FORBIDDEN"
    assert provenance["exact_cp2_byte_continuity_verified"] is False
    assert provenance["r1a_products_mutated"] is False


def test_evidence_and_contracts_exist() -> None:
    evidence = BUILD / "cp3_godot_equip_evidence.png"
    assert evidence.is_file() and evidence.stat().st_size > 100_000
    schemas = sorted((ROOT / "contracts/rig_game_product").glob("*.schema.json"))
    assert len(schemas) == 4
