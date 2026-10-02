from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/r1c_cp4_r1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_component_triangulation_and_rest_metric() -> None:
    triangulation = load(BUILD / "triangulation_receipt.json")
    rest = load(BUILD / "compiled_rest_metric_receipt.json")
    assert triangulation["component_count"] == 7
    assert triangulation["vertex_count"] > 900
    assert triangulation["triangle_count"] > 1400
    assert rest["settling_rest_owner"] == "ARRANGEMENT_REST_METRIC"
    assert rest["pattern_metric_preserved_as_diagnostic"] is True


def test_orientation_aware_seam_map_and_cap_patch() -> None:
    seam = load(BUILD / "seam_correspondence_receipt.json")
    cap = load(BUILD / "cap_patch_arrangement_receipt.json")
    assert seam["interface_count"] == 16
    assert seam["reversed_count"] >= 1
    assert cap["source_pattern_changed"] is False
    assert cap["post_alignment_seam_p95_m"] <= 1.0e-8


def test_warp_settling_is_bounded() -> None:
    settling = load(BUILD / "warp_settling_receipt.json")
    qualification = load(BUILD / "technical_qualification_receipt.json")
    assert settling["runtime"] == "warp-lang"
    assert settling["material"]["profile_id"] == "WOOL_TWILL_MEDIUM_REFERENCE"
    assert settling["post_settle_vertex_repair_count"] == 0
    assert qualification["technical_pass"] is True
    assert qualification["seam"]["p95_m"] <= 0.004
    assert qualification["edge_strain"]["p95"] <= 0.08


def test_blender_visual_product_closeout() -> None:
    visual = load(BUILD / "visual_evidence_receipt.json")
    blender = load(BUILD / "blender_visual_product_receipt.json")
    receipt = load(BUILD / "cp4_r1_receipt.json")
    assert blender["blender_version"] == "5.2.2"
    assert visual["visual_review"] == "PASS"
    assert visual["view_count"] == 12
    assert receipt["product_acceptance"] is True


def test_published_arrays_are_finite() -> None:
    with np.load(BUILD / "compiled_materialization.npz", allow_pickle=False) as data:
        assert np.isfinite(data["positions_final"]).all()
        assert len(data["triangles"]) > 0
        assert len(data["seam_pairs"]) > 0
