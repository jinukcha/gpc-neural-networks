from __future__ import annotations

from wuxia_garment_oss.garments.sleeveless_tunic.geometry import compile_geometry_package
from wuxia_garment_oss.garments.sleeveless_tunic.pattern.resolver import resolve_pattern_parameters
from wuxia_garment_oss.garments.sleeveless_tunic.reference import body_fixtures, reference_size_table
from wuxia_garment_oss.sizing.selection.resolver import SelectionRequest, resolve_selection


def _standard(size_id: str):
    table = reference_size_table()
    receipt = resolve_selection(
        SelectionRequest(f"STANDARD_{size_id}", "STANDARD_SIZE", size_id), table, None
    ).to_dict()
    parameter = resolve_pattern_parameters(receipt, table, None)
    return compile_geometry_package(parameter)


def _auto(body_id: str):
    table = reference_size_table()
    body = body_fixtures()[body_id]
    receipt = resolve_selection(
        SelectionRequest(f"AUTO_{body_id}", "AUTO_BODY_FIT"), table, body
    ).to_dict()
    parameter = resolve_pattern_parameters(receipt, table, body)
    return compile_geometry_package(parameter)


def _custom():
    table = reference_size_table()
    body = body_fixtures()["MILD_CUSTOM"]
    receipt = resolve_selection(
        SelectionRequest("CUSTOM_MILD", "CUSTOM_MEASUREMENTS", "M"), table, body
    ).to_dict()
    parameter = resolve_pattern_parameters(receipt, table, body)
    return compile_geometry_package(parameter)


def test_all_standard_geometry_is_admitted() -> None:
    for size in ("S", "M", "L"):
        package, arrays = _standard(size)
        assert package["triangulation_admission"] == "PASS"
        assert package["qualification"]["passed"] is True
        assert len(arrays) == 8


def test_boundary_spacing_and_open_disposition() -> None:
    package, _ = _standard("M")
    for panel in package["panels"]:
        for boundary in panel["boundaries"]:
            assert boundary["maximum_sample_spacing_m"] <= 0.012012
    dispositions = {
        (panel["panel_id"], boundary["boundary_id"]): boundary["disposition"]
        for panel in package["panels"] for boundary in panel["boundaries"]
    }
    assert dispositions[("bodice_front", "neckline")] == "OPEN"
    assert dispositions[("bodice_front", "armhole_left")] == "OPEN"
    assert dispositions[("skirt_front", "hem")] == "OPEN"


def test_seam_correspondence_is_complete_and_bounded() -> None:
    package, _ = _auto("FULL_CHEST")
    seams = package["seam_correspondence"]
    assert len(seams) == 8
    assert all(row["coverage"] == 1.0 for row in seams)
    assert all(row["pair_count"] >= 2 for row in seams)
    assert max(row["length_mismatch_ratio"] for row in seams) <= 0.03
    assert all(len(row["samples_a"]) == len(row["samples_b"]) for row in seams)


def test_triangulation_area_and_edge_gates() -> None:
    package, arrays = _auto("FULL_ABDOMEN")
    for panel in package["panels"]:
        quality = panel["triangulation"]
        assert quality["self_intersection_count"] == 0
        assert quality["degenerate_triangle_count"] == 0
        assert abs(quality["area_coverage_ratio"] - 1.0) <= 1.0e-8
        assert quality["maximum_edge_m"] <= 0.03000003
        panel_id = panel["panel_id"]
        assert arrays[f"{panel_id}__vertices"].shape[1] == 2
        assert arrays[f"{panel_id}__triangles"].shape[1] == 3


def test_standard_triangle_counts_are_size_aware() -> None:
    counts = [_standard(size)[0]["qualification"]["total_triangles"] for size in ("S", "M", "L")]
    assert counts[0] <= counts[1] <= counts[2]


def test_custom_measurements_publish_geometry() -> None:
    package, _ = _custom()
    assert package["selection"]["admission"] == "CUSTOM_ALTERATION"
    assert package["triangulation_admission"] == "PASS"


def test_geometry_id_is_deterministic() -> None:
    first, arrays_first = _auto("MILD_CUSTOM")
    second, arrays_second = _auto("MILD_CUSTOM")
    assert first["geometry_package_id"] == second["geometry_package_id"]
    assert first["array_hashes"] == second["array_hashes"]
    for key in arrays_first:
        assert (arrays_first[key] == arrays_second[key]).all()


def test_product_simulation_is_not_run() -> None:
    package, _ = _standard("M")
    assert package["warp_simulation_executed"] is False
    assert package["product_simulation_executed"] is False
    assert package["mesh_scaling"] == "FORBIDDEN"
