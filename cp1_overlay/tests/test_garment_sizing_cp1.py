from __future__ import annotations

from wuxia_garment_oss.garments.sleeveless_tunic.reference import (
    body_fixtures,
    reference_size_table,
)
from wuxia_garment_oss.garments.sleeveless_tunic.resolver import resolve_tunic_instance
from wuxia_garment_oss.sizing.selection.resolver import SelectionRequest, resolve_selection


def _auto(name: str):
    return resolve_selection(
        SelectionRequest(f"test_{name}", "AUTO_BODY_FIT"),
        reference_size_table(),
        body_fixtures()[name],
    )


def test_arbitrary_n_size_table() -> None:
    table = reference_size_table()
    assert table.ordered_ids() == ("S", "M", "L", "XL")


def test_reference_selects_m_without_alteration() -> None:
    receipt = _auto("REFERENCE")
    assert receipt.selected_size_id == "M"
    assert receipt.shape_block == "REGULAR"
    assert receipt.height_block == "REGULAR"
    assert receipt.admission == "NORMAL_GRADE"


def test_supported_shape_blocks() -> None:
    expectations = {
        "BROAD_SHOULDER": "BROAD_SHOULDER",
        "FULL_CHEST": "FULL_CHEST",
        "FULL_ABDOMEN": "FULL_ABDOMEN",
    }
    for fixture, block in expectations.items():
        receipt = _auto(fixture)
        assert receipt.shape_block == block
        assert receipt.admission == "CUSTOM_ALTERATION"


def test_supported_height_blocks() -> None:
    expectations = {"TALL": "TALL", "SHORT": "SHORT"}
    for fixture, block in expectations.items():
        receipt = _auto(fixture)
        assert receipt.height_block == block
        assert receipt.admission == "CUSTOM_ALTERATION"


def test_mild_custom_alteration_is_admitted() -> None:
    receipt = _auto("MILD_CUSTOM")
    assert receipt.shape_block == "REGULAR"
    assert receipt.height_block == "REGULAR"
    assert receipt.admission == "CUSTOM_ALTERATION"


def test_forced_custom_size_requests_alternate() -> None:
    receipt = resolve_selection(
        SelectionRequest("forced", "CUSTOM_MEASUREMENTS", "M"),
        reference_size_table(),
        body_fixtures()["NEAR_L_FORCED_M"],
    )
    assert receipt.recommended_size_id == "L"
    assert receipt.admission == "ALTERNATE_BLOCK_REQUIRED"


def test_topology_and_hold_are_explicit() -> None:
    assert _auto("COMBINED_TOPOLOGY").admission == "TOPOLOGY_CHANGE_REQUIRED"
    assert _auto("OUT_OF_RANGE").admission == "HOLD"


def test_grade_and_custom_plans_are_separate() -> None:
    receipt = _auto("BROAD_SHOULDER")
    assert all(row["operation"] == "GRADE" for row in receipt.grade_plan)
    assert all(row["operation"] == "CUSTOM_ALTERATION" for row in receipt.custom_alteration_plan)


def test_standard_size_and_instance_do_not_run_geometry() -> None:
    request = SelectionRequest("standard", "STANDARD_SIZE", "L")
    instance, receipt = resolve_tunic_instance(request, reference_size_table(), None)
    assert receipt["admission"] == "NORMAL_GRADE"
    assert instance["triangulation_executed"] is False
    assert instance["warp_simulation_executed"] is False
    assert instance["mesh_scaling"] == "FORBIDDEN"


def test_receipt_hash_is_deterministic() -> None:
    first = _auto("FULL_CHEST").to_dict()
    second = _auto("FULL_CHEST").to_dict()
    assert first["receipt_sha256"] == second["receipt_sha256"]
