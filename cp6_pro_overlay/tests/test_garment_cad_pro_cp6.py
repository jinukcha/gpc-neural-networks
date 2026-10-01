from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/garment_cad_pro_cp6"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manufacturing_exports_are_one_to_one() -> None:
    tunic = _load(BUILD / "manufacturing/tunic/manufacturing_pattern_package.json")
    trousers = _load(BUILD / "manufacturing/trousers/manufacturing_pattern_package.json")
    assert tunic["contract"] == "ManufacturingPatternPackage/1"
    assert trousers["contract"] == "ManufacturingPatternPackage/1"
    assert tunic["scale"] == trousers["scale"] == "1:1"
    assert len(tunic["panels"]) >= 5
    assert len(trousers["panels"]) == 7
    assert (BUILD / "manufacturing/tunic/manufacturing_pattern_1to1.svg").is_file()
    assert (BUILD / "manufacturing/trousers/manufacturing_pattern_r12.dxf").is_file()


def test_trousers_pattern_has_required_professional_features() -> None:
    authority = _load(BUILD / "trousers/pattern_authority.json")
    assert authority["contract"] == "TrousersPatternAuthority/1"
    assert authority["dart_count"] == 4
    assert authority["gusset_panel_count"] == 1
    assert authority["waistband_panel_count"] == 2
    assert len(authority["panel_ids"]) == 7
    pom = authority["points_of_measure"]
    assert pom["finished_outseam"] > pom["finished_inseam"]
    assert pom["back_crotch_extension"] > pom["front_crotch_extension"]
    assert pom["back_waist_quarter"] > pom["front_waist_quarter"]
    assert pom["back_hip_quarter"] > pom["front_hip_quarter"]


def test_trousers_topology_is_product_admissible() -> None:
    receipt = _load(BUILD / "trousers/topology_receipt.json")
    assert receipt["topology_pass"] is True
    assert receipt["degenerate_triangle_count"] == 0
    assert receipt["non_finite_vertex_count"] == 0
    assert receipt["maximum_edge_incidence"] <= 2
    assert receipt["boundary_loop_count"] == receipt["expected_boundary_loop_count"] == 4
    assert set(receipt["expected_open_boundaries"]) == {
        "waistband_join",
        "left_ankle_hem",
        "right_ankle_hem",
        "crotch_gusset_insertion",
    }


def test_trousers_nine_motion_scenarios_pass() -> None:
    receipts = [_load(path) for path in sorted((BUILD / "trousers/fit").glob("*/*.json"))]
    assert len(receipts) == 9
    assert all(item["pose_pass"] for item in receipts)
    assert {item["pose_id"] for item in receipts} == {"SEATED", "SQUAT", "WALK_STRIDE"}
    assert len({item["material_id"] for item in receipts}) == 3


def test_glb_fresh_reopen_and_godot_consumer() -> None:
    fresh = _load(BUILD / "game_products/fresh_process_receipt.json")
    godot = _load(BUILD / "godot_product/godot_consumer_receipt.json")
    assert fresh["fresh_process_pass"] is True
    assert fresh["tunic"]["morph_target_counts"] == [10]
    assert fresh["trousers"]["morph_target_counts"] == [3, 3, 3]
    assert fresh["tunic"]["uv0_pass"] is True
    assert fresh["trousers"]["uv0_pass"] is True
    assert godot["consumer_pass"] is True
    assert godot["tunic_pass"] is True
    assert godot["trousers_pass"] is True


def test_cp6_terminal_scope_and_known_program_boundary() -> None:
    status = _load(ROOT / "PROFESSIONAL_STATUS.json")
    assert status["terminal_decision"] == "CP6_ACCEPTED_EXPORT_AND_TROUSERS_CLOSEOUT"
    assert status["cp6_scope_acceptance"] is True
    assert status["product_acceptance"] is True
    assert status["trousers_actual_product_topology"] is True
    assert status["tunic_cp3_feature_topology_embodied"] is False
    assert status["garment_cad_pro_r1a_complete"] is False
    assert status["thresholds_changed"] is False
    assert status["mesh_scaling"] == "FORBIDDEN"
