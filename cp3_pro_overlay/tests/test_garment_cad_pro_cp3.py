from __future__ import annotations

import json
import math
from pathlib import Path

from wuxia_garment_oss.construction.compiler import (
    compile_construction_package,
    construction_failure_probes,
)
from wuxia_garment_oss.garments.sleeveless_tunic.construction.professional import (
    tunic_construction_authority,
)
from wuxia_garment_oss.pattern_cad.document.model import PatternDocument
from wuxia_garment_oss.pattern_cad.document.resolver import require_resolved


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    base = ROOT / "build/pattern_cad_cp2"
    document = PatternDocument.from_dict(json.loads(
        (base / "features/composite_pattern_document.json").read_text()
    ))
    resolved = require_resolved(document)
    notch_set = json.loads((base / "grading/notch_set.json").read_text())
    notch_ids = {item["notch_id"] for item in notch_set["notches"]}
    return document, resolved, notch_ids


def _package() -> tuple[PatternDocument, dict, dict]:
    document, resolved, notch_ids = _inputs()
    authority = tunic_construction_authority()
    package = compile_construction_package(
        document, resolved, authority["seams"], authority["finishes"], authority["notches"],
        notch_ids, authority["closures"], authority["facings"], authority["layers"],
        authority["turns"], authority["graph"], authority["bill_of_materials"],
    )
    return document, authority, package


def test_construction_authority_counts() -> None:
    _, _, package = _package()
    assert len(package["seam_specs"]) == 12
    assert len(package["edge_finishes"]) == 8
    assert len(package["notch_correspondence"]) == 12
    assert len(package["closures"]) == 1
    assert len(package["facings"]) == 2
    assert len(package["layer_pieces"]) == 7
    assert len(package["turn_of_cloth"]) == 3


def test_stitch_and_cut_line_offset_matches_allowance() -> None:
    _, _, package = _package()
    for row in package["seam_lines"]:
        for side in (row["side_a"], row["side_b"]):
            distances = [math.dist(a, b) for a, b in zip(side["stitch_line"], side["cut_line"])]
            assert max(abs(value - side["allowance_m"]) for value in distances) <= 1.0e-9


def test_notch_correspondence_preserves_cp2_and_generated_identity() -> None:
    _, _, package = _package()
    inherited = [row for row in package["notch_correspondence"] if not row["side_a_notch_id"].startswith("CP3_")]
    generated = [row for row in package["notch_correspondence"] if row["side_a_notch_id"].startswith("CP3_")]
    assert len(inherited) == 6
    assert len(generated) == 6
    assert all(row["role"] for row in package["notch_correspondence"])


def test_gather_and_gusset_construction_semantics() -> None:
    _, _, package = _package()
    seams = {row["seam_id"]: row for row in package["seam_specs"]}
    assert seams["waist_back"]["seam_type"] == "GATHERED_SEAM"
    assert seams["waist_back"]["gather_ratio"] == 1.12
    assert sum(row["seam_type"] == "GUSSET_INSERTION" for row in seams.values()) == 4


def test_closure_facing_lining_interfacing_and_turn_of_cloth() -> None:
    _, _, package = _package()
    closure = package["closures"][0]
    assert math.dist(*closure["internal_cut_line"]) == closure["length_m"]
    assert {row["layer_type"] for row in package["layer_pieces"]} == {"LINING", "INTERFACING"}
    assert all(row["turn_of_cloth_m"] > 0.0 for row in package["facings"])
    assert all(row["allowance_m"] > 0.0 for row in package["turn_of_cloth"])


def test_assembly_graph_is_cycle_free_and_dependency_ordered() -> None:
    _, _, package = _package()
    plan = package["assembly_plan"]
    assert plan["cycle_free"] is True
    positions = {row["operation_id"]: row["sequence"] for row in plan["sequence"]}
    for row in plan["sequence"]:
        assert all(positions[parent] < row["sequence"] for parent in row["depends_on"])
    assert plan["operation_count"] == 21


def test_construction_failures_are_atomic() -> None:
    document, authority, _ = _package()
    before = document.to_dict()["document_sha256"]
    result = construction_failure_probes(document, authority["seams"][0], authority["graph"])
    assert result["probe_count"] == 4
    assert result["all_rejected"] is True
    assert result["all_atomic"] is True
    assert document.to_dict()["document_sha256"] == before


def test_cp3_does_not_mesh_or_simulate_or_scale() -> None:
    document, _, package = _package()
    assert package["source_document_mutated"] is False
    assert package["triangulation_executed"] is False
    assert package["warp_simulation_executed"] is False
    assert package["mesh_scaling"] == "FORBIDDEN"
    assert document.to_dict()["document_sha256"] == package["source_pattern_document_sha256"]
