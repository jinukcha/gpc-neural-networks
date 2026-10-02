from __future__ import annotations

import json
from pathlib import Path

from wuxia_garment_oss.pattern_geometry import (
    assembly_recipe,
    build_geometry_library,
    compile_assembly,
    component_interfaces,
    component_registry,
    load_geometry_inputs,
    solve_interfaces,
)
from wuxia_garment_oss.pattern_geometry.curve import boundary_length
from wuxia_garment_oss.pattern_geometry.fixtures import shifted_front_pitch


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/r1c_cp2"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _products():
    inputs = load_geometry_inputs(ROOT)
    registry = component_registry()
    library, geometries = build_geometry_library(inputs)
    recipe = assembly_recipe()
    interfaces = component_interfaces()
    solved = solve_interfaces(registry, recipe, interfaces, geometries)
    return inputs, registry, library, geometries, recipe, interfaces, solved


def test_component_library_contains_bodice_sleeve_collar_cuff_gore() -> None:
    _, registry, library, geometries, *_ = _products()
    categories = {item.category for item in registry.components}
    assert {"BODICE", "SLEEVE", "COLLAR", "CUFF", "GORE"}.issubset(categories)
    assert library["authority_count"] == 9
    assert set(geometries) == {"bodice_front", "bodice_back", "sleeve_left", "sleeve_right", "collar", "cuff_left", "cuff_right", "gore_left", "gore_right"}


def test_sleeve_cap_and_notch_correspondence_pass() -> None:
    *_, solved = _products()
    cap_receipts = [item for item in solved["interface_receipts"] if item["semantic_role"] == "SLEEVE_CAP"]
    assert solved["accepted"] is True
    assert len(cap_receipts) == 4
    assert all(item["length_pass"] and item["notch_pass"] for item in cap_receipts)
    assert all(1.0 <= item["directed_or_symmetric_ratio"] <= 1.08 for item in cap_receipts)
    assert all(row["delta"] <= 0.025 for item in cap_receipts for row in item["notch_correspondence"])


def test_gore_geometry_is_exact_2d_and_not_triangulated() -> None:
    _, _, _, geometries, *_ = _products()
    gore = geometries["gore_left"]
    payload = gore.to_dict()
    lengths = [boundary_length(boundary, gore.segment_map()) for boundary in gore.boundaries]
    assert payload["contract"] == "ComponentGeometryAuthority/1"
    assert payload["three_dimensional_primitive_authority"] is False
    assert payload["triangulation_executed"] is False
    assert min(lengths) > 0.0


def test_assembly_compiles_without_triangulation() -> None:
    inputs, registry, _, geometries, recipe, _, solved = _products()
    package, receipt = compile_assembly(
        registry,
        recipe,
        geometries,
        solved,
        inputs.resolved_set_sha256,
        inputs.seam_allowance_m,
        inputs.turn_of_cloth_m,
    )
    assert receipt["accepted"] is True
    assert package is not None
    assert package["component_instance_count"] == 7
    assert package["seam_count"] == 16
    assert package["triangulation_executed"] is False
    assert package["simulation_executed"] is False


def test_notch_mismatch_rejects_atomically() -> None:
    inputs, registry, _, geometries, recipe, interfaces, _ = _products()
    broken = dict(geometries)
    broken["sleeve_left"] = shifted_front_pitch(geometries["sleeve_left"])
    solved = solve_interfaces(registry, recipe, interfaces, broken)
    package, receipt = compile_assembly(
        registry,
        recipe,
        broken,
        solved,
        inputs.resolved_set_sha256,
        inputs.seam_allowance_m,
        inputs.turn_of_cloth_m,
    )
    assert solved["accepted"] is False
    assert any("NOTCH_CORRESPONDENCE_MISMATCH" in item for item in solved["errors"])
    assert package is None
    assert receipt["status"] == "REJECTED_ATOMIC"
    assert receipt["partial_publication_count"] == 0


def test_terminal_cp2_products_pass() -> None:
    receipt = _load(BUILD / "cp2_receipt.json")
    checks = _load(ROOT / "R1C_CHECKS.json")
    assert receipt["cp2_acceptance"] is True
    assert receipt["terminal_decision"] == "CP2_COMPLETE_PATTERN_COMPONENT_ASSEMBLY"
    assert receipt["fresh_process_reopen_pass"] is True
    assert receipt["deterministic_rebuild_pass"] is True
    assert receipt["triangulation_executed"] is False
    assert receipt["simulation_executed"] is False
    assert checks["validation"] == "PASS"
