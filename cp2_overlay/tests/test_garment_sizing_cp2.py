from __future__ import annotations

import pytest

from wuxia_garment_oss.garments.sleeveless_tunic.pattern.resolver import (
    resolve_pattern_parameters,
)
from wuxia_garment_oss.garments.sleeveless_tunic.reference import (
    body_fixtures,
    reference_size_table,
)
from wuxia_garment_oss.sizing.selection.resolver import SelectionRequest, resolve_selection


def _standard(size_id: str) -> dict:
    table = reference_size_table()
    receipt = resolve_selection(
        SelectionRequest(f"STANDARD_{size_id}", "STANDARD_SIZE", size_id), table, None
    ).to_dict()
    return resolve_pattern_parameters(receipt, table, None)


def _body_mode(body_id: str, mode: str, requested_size: str | None = None) -> dict:
    table = reference_size_table()
    body = body_fixtures()[body_id]
    request = SelectionRequest(f"{mode}_{body_id}", mode, requested_size)
    receipt = resolve_selection(request, table, body).to_dict()
    return resolve_pattern_parameters(receipt, table, body)


def _auto(body_id: str) -> dict:
    return _body_mode(body_id, "AUTO_BODY_FIT")


def test_standard_sml_poms_are_monotonic() -> None:
    packages = [_standard(size) for size in ("S", "M", "L")]
    for key in (
        "finished_chest_circumference_m", "finished_waist_circumference_m",
        "shoulder_half_m", "skirt_length_m", "hem_half_m",
    ):
        values = [package["poms"][key] for package in packages]
        assert values[0] < values[1] < values[2]


def test_reference_m_reproduces_cp2b_parameters() -> None:
    poms = _standard("M")["poms"]
    expected = {
        "front_chest_half_m": 0.275,
        "back_chest_half_m": 0.269,
        "front_waist_half_m": 0.244,
        "back_waist_half_m": 0.244,
        "shoulder_half_m": 0.205,
        "neck_half_m": 0.083,
        "underarm_height_m": 0.285,
        "front_shoulder_height_m": 0.455,
        "back_shoulder_height_m": 0.455,
        "front_top_height_m": 0.525,
        "back_top_height_m": 0.525,
        "front_neck_depth_m": 0.160,
        "back_neck_depth_m": 0.072,
        "skirt_length_m": 0.820,
        "hem_half_m": 0.310,
    }
    assert max(abs(poms[name] - value) for name, value in expected.items()) <= 1.0e-12


def test_custom_measurements_publish_supported_alteration() -> None:
    package = _body_mode("MILD_CUSTOM", "CUSTOM_MEASUREMENTS", "M")
    assert package["sizing_mode"] == "CUSTOM_MEASUREMENTS"
    assert package["selection"]["admission"] == "CUSTOM_ALTERATION"
    assert package["selection"]["selected_size_id"] == "M"


def test_full_chest_changes_front_distribution() -> None:
    reference = _auto("REFERENCE")["poms"]
    altered = _auto("FULL_CHEST")["poms"]
    assert altered["front_chest_half_m"] - reference["front_chest_half_m"] > 0.015
    assert altered["back_chest_half_m"] == pytest.approx(reference["back_chest_half_m"])


def test_full_abdomen_changes_front_waist_distribution() -> None:
    reference = _auto("REFERENCE")["poms"]
    altered = _auto("FULL_ABDOMEN")["poms"]
    assert altered["front_waist_half_m"] - reference["front_waist_half_m"] > 0.020
    assert altered["back_waist_half_m"] == pytest.approx(reference["back_waist_half_m"])


def test_shape_and_height_landmarks_change_independently() -> None:
    reference = _auto("REFERENCE")
    broad = _auto("BROAD_SHOULDER")
    tall = _auto("TALL")
    short = _auto("SHORT")
    assert broad["poms"]["shoulder_half_m"] > reference["poms"]["shoulder_half_m"]
    assert tall["poms"]["skirt_length_m"] > reference["poms"]["skirt_length_m"]
    assert short["poms"]["skirt_length_m"] < reference["poms"]["skirt_length_m"]


def test_panel_and_boundary_authority_is_complete() -> None:
    package = _standard("M")
    assert {panel["panel_id"] for panel in package["panels"]} == {
        "bodice_front", "bodice_back", "skirt_front", "skirt_back"
    }
    assert {pair["seam_id"] for pair in package["seam_pairs"]} == {
        "shoulder_left", "shoulder_right", "bodice_side_left", "bodice_side_right",
        "waist_front", "waist_back", "skirt_side_left", "skirt_side_right",
    }
    front = next(panel for panel in package["panels"] if panel["panel_id"] == "bodice_front")
    dispositions = {row["boundary_id"]: row["disposition"] for row in front["boundaries"]}
    assert dispositions["neckline"] == "OPEN"
    assert dispositions["armhole_left"] == "OPEN"
    assert dispositions["armhole_right"] == "OPEN"


def test_blocked_admission_cannot_publish_parameters() -> None:
    table = reference_size_table()
    body = body_fixtures()["COMBINED_TOPOLOGY"]
    receipt = resolve_selection(
        SelectionRequest("blocked", "AUTO_BODY_FIT"), table, body
    ).to_dict()
    assert receipt["admission"] == "TOPOLOGY_CHANGE_REQUIRED"
    with pytest.raises(ValueError):
        resolve_pattern_parameters(receipt, table, body)


def test_package_is_deterministic_and_geometry_is_deferred() -> None:
    first = _auto("MILD_CUSTOM")
    second = _auto("MILD_CUSTOM")
    assert first["package_id"] == second["package_id"]
    assert first["triangulation_executed"] is False
    assert first["warp_simulation_executed"] is False
    assert first["mesh_scaling"] == "FORBIDDEN"
