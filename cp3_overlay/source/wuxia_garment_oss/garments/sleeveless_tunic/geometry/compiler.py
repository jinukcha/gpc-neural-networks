"""Compile CP2 tunic parameters into curves, seam pairs, and 2D mesh admission."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ....sizing.geometry.curve import count_for_length
from ....sizing.geometry.triangulation import triangulate_panel
from ....sizing.instance.model import canonical_sha256
from .curves import TARGET_BOUNDARY_EDGE_M, assemble_outline, compile_panel_curves

MAX_TRIANGLE_EDGE_M = 0.030
MAX_SEAM_MISMATCH_RATIO = 0.030


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _boundary(panel_payload: dict, boundary_id: str) -> dict:
    return next(row for row in panel_payload["boundaries"] if row["boundary_id"] == boundary_id)


def _compile_seam(
    seam: dict,
    panel_payloads: dict[str, dict],
    curve_maps: dict[str, dict],
    target_edge_m: float,
) -> dict:
    panel_a, panel_b = seam["a"]["panel_id"], seam["b"]["panel_id"]
    boundary_a, boundary_b = seam["a"]["boundary_id"], seam["b"]["boundary_id"]
    source_a = _boundary(panel_payloads[panel_a], boundary_a)
    source_b = _boundary(panel_payloads[panel_b], boundary_b)
    if source_a["disposition"] != "SEWN" or source_b["disposition"] != "SEWN":
        raise ValueError(f"seam references an open boundary: {seam['seam_id']}")
    length_a, length_b = float(source_a["length_m"]), float(source_b["length_m"])
    pair_count = count_for_length(max(length_a, length_b), target_edge_m, 2)
    points_a = curve_maps[panel_a][boundary_a].sample_by_arclength(pair_count)
    points_b = curve_maps[panel_b][boundary_b].sample_by_arclength(pair_count)
    mismatch = abs(length_a - length_b) / max(length_a, length_b)
    return {
        "seam_id": seam["seam_id"],
        "a": seam["a"],
        "b": seam["b"],
        "orientation_relation": "SAME_NORMALIZED_ARC_DIRECTION",
        "pair_count": pair_count,
        "length_a_m": length_a,
        "length_b_m": length_b,
        "length_mismatch_ratio": mismatch,
        "coverage": 1.0,
        "samples_a": [[float(x), float(y)] for x, y in points_a],
        "samples_b": [[float(x), float(y)] for x, y in points_b],
    }


def _compile_panel(panel: dict, target_edge_m: float, maximum_mesh_edge_m: float) -> tuple[dict, dict, dict]:
    payload, curves = compile_panel_curves(panel, target_edge_m)
    outline = assemble_outline(payload)
    vertices, triangles, mesh = triangulate_panel(outline, maximum_mesh_edge_m)
    vertex_array = np.asarray(vertices, dtype=np.float32)
    triangle_array = np.asarray(triangles, dtype=np.int32)
    payload["outline"] = [[float(x), float(y)] for x, y in outline]
    payload["triangulation"] = {
        **mesh,
        "maximum_allowed_edge_m": maximum_mesh_edge_m,
        "vertices_sha256": _array_sha256(vertex_array),
        "triangles_sha256": _array_sha256(triangle_array),
    }
    return payload, curves, {"vertices": vertex_array, "triangles": triangle_array}


def _qualification(panels: list[dict], seams: list[dict]) -> dict:
    boundaries = [boundary for panel in panels for boundary in panel["boundaries"]]
    triangulations = [panel["triangulation"] for panel in panels]
    critical_open = {
        ("bodice_front", "neckline"), ("bodice_back", "neckline"),
        ("bodice_front", "armhole_left"), ("bodice_front", "armhole_right"),
        ("bodice_back", "armhole_left"), ("bodice_back", "armhole_right"),
        ("skirt_front", "hem"), ("skirt_back", "hem"),
    }
    dispositions = {
        (panel["panel_id"], boundary["boundary_id"]): boundary["disposition"]
        for panel in panels for boundary in panel["boundaries"]
    }
    maximum_spacing = max(float(row["maximum_sample_spacing_m"]) for row in boundaries)
    maximum_mismatch = max(float(row["length_mismatch_ratio"]) for row in seams)
    gates = {
        "panel_count_4": len(panels) == 4,
        "seam_count_8": len(seams) == 8,
        "critical_open_boundaries": all(dispositions.get(key) == "OPEN" for key in critical_open),
        "boundary_spacing": maximum_spacing <= TARGET_BOUNDARY_EDGE_M * 1.001,
        "seam_coverage": all(float(row["coverage"]) == 1.0 for row in seams),
        "seam_length_mismatch": maximum_mismatch <= MAX_SEAM_MISMATCH_RATIO,
        "panel_self_intersection": all(row["self_intersection_count"] == 0 for row in triangulations),
        "degenerate_triangles": all(row["degenerate_triangle_count"] == 0 for row in triangulations),
        "mesh_area_coverage": all(abs(float(row["area_coverage_ratio"]) - 1.0) <= 1.0e-8 for row in triangulations),
        "maximum_mesh_edge": all(float(row["maximum_edge_m"]) <= MAX_TRIANGLE_EDGE_M * 1.000001 for row in triangulations),
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "maximum_boundary_spacing_m": maximum_spacing,
        "maximum_seam_length_mismatch_ratio": maximum_mismatch,
        "total_boundary_samples": sum(int(row["sample_count"]) for row in boundaries),
        "total_seam_pairs": sum(int(row["pair_count"]) for row in seams),
        "total_mesh_vertices": sum(int(row["mesh_vertex_count"]) for row in triangulations),
        "total_triangles": sum(int(row["triangle_count"]) for row in triangulations),
    }


def compile_geometry_package(
    parameter_package: dict,
    target_edge_m: float = TARGET_BOUNDARY_EDGE_M,
    maximum_mesh_edge_m: float = MAX_TRIANGLE_EDGE_M,
) -> tuple[dict, dict[str, np.ndarray]]:
    if parameter_package["contract"] != "GarmentPatternParameterPackage/1":
        raise ValueError("CP3 requires GarmentPatternParameterPackage/1")
    panel_payloads: dict[str, dict] = {}
    curve_maps: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for panel in parameter_package["panels"]:
        payload, curves, mesh_arrays = _compile_panel(panel, target_edge_m, maximum_mesh_edge_m)
        panel_id = panel["panel_id"]
        panel_payloads[panel_id] = payload
        curve_maps[panel_id] = curves
        arrays[f"{panel_id}__vertices"] = mesh_arrays["vertices"]
        arrays[f"{panel_id}__triangles"] = mesh_arrays["triangles"]
    seams = [
        _compile_seam(seam, panel_payloads, curve_maps, target_edge_m)
        for seam in parameter_package["seam_pairs"]
    ]
    panels = [panel_payloads[panel["panel_id"]] for panel in parameter_package["panels"]]
    qualification = _qualification(panels, seams)
    payload: dict[str, Any] = {
        "contract": "GarmentGeometryPackage/1",
        "source_parameter_package_id": parameter_package["package_id"],
        "garment_design_id": parameter_package["garment_design_id"],
        "topology_class": parameter_package["topology_class"],
        "selection": parameter_package["selection"],
        "target_boundary_edge_m": target_edge_m,
        "maximum_mesh_edge_m": maximum_mesh_edge_m,
        "panels": panels,
        "seam_correspondence": seams,
        "qualification": qualification,
        "triangulation_admission": "PASS" if qualification["passed"] else "FAIL",
        "warp_simulation_executed": False,
        "product_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "array_hashes": {name: _array_sha256(array) for name, array in sorted(arrays.items())},
    }
    payload["geometry_package_id"] = canonical_sha256(payload)
    return payload, arrays
