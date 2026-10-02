from __future__ import annotations

import json
from pathlib import Path

from wuxia_garment_oss.pattern_assembly.compiler import evaluate_recipe
from wuxia_garment_oss.r1c_cp0_fixtures.components import assembly_recipe, component_interfaces, component_registry
from wuxia_garment_oss.r1c_cp0_fixtures.visual import clean_observations, rejected_observations, visual_profile
from wuxia_garment_oss.visual_acceptance.evaluator import evaluate_visual_review


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/r1c_cp0"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _refs(profile, family: str) -> dict[str, list[str]]:
    views = [
        item.view_id for item in profile.required_views
        if "ALL" in item.applicable_families or family in item.applicable_families
    ]
    return {item.gate_id: views for item in profile.gates}


def test_component_registry_is_exact_2d_authority() -> None:
    registry = component_registry()
    registry.validate()
    assert len(registry.components) == 3
    for component in registry.components:
        payload = component.to_dict()
        assert payload["geometry_authority"] == "EXACT_2D_PATTERN"
        assert payload["three_dimensional_primitive_authority"] is False
        assert component.interface_ports


def test_complete_recipe_binds_every_sewn_boundary() -> None:
    receipt = evaluate_recipe(component_registry(), assembly_recipe(True), component_interfaces())
    assert receipt["accepted"] is True
    assert receipt["required_sewn_boundary_count"] == receipt["bound_sewn_boundary_count"]
    assert receipt["unbound_sewn_boundaries"] == []
    assert receipt["geometry_executed"] is False


def test_incomplete_recipe_is_rejected_without_geometry() -> None:
    receipt = evaluate_recipe(component_registry(), assembly_recipe(False), component_interfaces())
    assert receipt["accepted"] is False
    assert len(receipt["unbound_sewn_boundaries"]) == 2
    assert all("sleeve_right" in item for item in receipt["unbound_sewn_boundaries"])
    assert receipt["simulation_executed"] is False


def test_visual_authority_requires_technical_and_visual_pass() -> None:
    profile = visual_profile()
    profile.validate()
    refs = _refs(profile, "SLEEVED_TUNIC")
    clean = evaluate_visual_review(profile, "SLEEVED_TUNIC", True, clean_observations(), refs)
    technical_fail = evaluate_visual_review(profile, "SLEEVED_TUNIC", False, clean_observations(), refs)
    assert clean["visual_review"] == "PASS"
    assert clean["product_acceptance"] is True
    assert technical_fail["visual_review"] == "PASS"
    assert technical_fail["product_acceptance"] is False


def test_visual_defects_block_product_acceptance() -> None:
    profile = visual_profile()
    rejected = evaluate_visual_review(
        profile,
        "SLEEVED_TUNIC",
        True,
        rejected_observations(),
        _refs(profile, "SLEEVED_TUNIC"),
    )
    assert rejected["visual_review"] == "FAIL"
    assert rejected["product_acceptance"] is False
    assert "floating_component_count" in rejected["failed_gate_ids"]
    assert "sleeve_cap_armhole_gap_count" in rejected["failed_gate_ids"]
    assert "HOLD" in rejected["repair_dispositions"]


def test_published_cp0_terminal_receipt() -> None:
    receipt = _load(BUILD / "cp0_receipt.json")
    assert receipt["terminal_decision"] == "CP0_COMPLETE_VISUAL_AND_MODULAR_FOUNDATION"
    assert receipt["cp0_acceptance"] is True
    assert receipt["clean_visual_fixture_pass"] is True
    assert receipt["defective_visual_fixture_rejected"] is True
    assert receipt["geometry_executed"] is False
    assert receipt["godot_executed"] is False
