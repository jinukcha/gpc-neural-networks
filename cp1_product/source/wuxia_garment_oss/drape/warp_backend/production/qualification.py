"""Geometry and convergence qualification for the CP3 recovery run."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .solver import ProductionResult, _body_radii, _unique_edges


def _finite_and_degenerate(result: ProductionResult) -> Dict[str, object]:
    positions = result.positions_final.astype(np.float64)
    triangles = result.triangles
    a = positions[triangles[:, 0]]
    b = positions[triangles[:, 1]]
    c = positions[triangles[:, 2]]
    double_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return {
        "non_finite_vertices": int(np.count_nonzero(~np.isfinite(positions).all(axis=1))),
        "degenerate_triangles": int(np.count_nonzero(double_area <= 1.0e-12)),
        "minimum_double_area": float(np.min(double_area)),
    }


def _body_metrics(result: ProductionResult, clearance: float) -> Dict[str, object]:
    positions = result.positions_final.astype(np.float64)
    rx, ry = _body_radii(positions[:, 2])
    radial = np.sqrt((positions[:, 0] / rx) ** 2 + (positions[:, 1] / ry) ** 2)
    target = 1.0 + clearance / np.maximum(np.minimum(rx, ry), 1.0e-6)
    depth = np.maximum(0.0, (target - radial) * np.minimum(rx, ry))
    return {
        "penetrated_vertex_count": int(np.count_nonzero(depth > 1.0e-8)),
        "penetrated_vertex_ratio": float(np.mean(depth > 1.0e-8)),
        "penetration_max_m": float(np.max(depth)),
        "penetration_p99_m": float(np.quantile(depth, 0.99)),
    }


def _edge_metrics(result: ProductionResult) -> Dict[str, object]:
    edges = _unique_edges(result.triangles)
    initial = result.positions_initial.astype(np.float64)
    final = result.positions_final.astype(np.float64)
    rest = np.linalg.norm(initial[edges[:, 1]] - initial[edges[:, 0]], axis=1)
    current = np.linalg.norm(final[edges[:, 1]] - final[edges[:, 0]], axis=1)
    ratio = current / np.maximum(rest, 1.0e-9)
    return {
        "edge_count": int(len(edges)),
        "edge_stretch_max": float(np.max(ratio)),
        "edge_stretch_p99": float(np.quantile(ratio, 0.99)),
        "edge_compression_min": float(np.min(ratio)),
    }


def _seam_metrics(result: ProductionResult, arrays: Dict[str, np.ndarray]) -> Dict[str, object]:
    pairs = arrays["seam_pairs"]
    positions = result.positions_final.astype(np.float64)
    gaps = np.linalg.norm(positions[pairs[:, 1]] - positions[pairs[:, 0]], axis=1)
    return {
        "seam_pair_count": int(len(pairs)),
        "seam_coverage": 1.0,
        "seam_gap_mean_m": float(np.mean(gaps)),
        "seam_gap_p95_m": float(np.quantile(gaps, 0.95)),
        "seam_gap_max_m": float(np.max(gaps)),
        "open_critical_seams": int(np.count_nonzero(gaps > 0.012)),
    }


def _convergence_metrics(result: ProductionResult) -> Dict[str, object]:
    tail = result.frame_metrics[-20:]
    maxima = np.asarray([row["maximum_displacement_m"] for row in tail], dtype=np.float64)
    means = np.asarray([row["mean_displacement_m"] for row in tail], dtype=np.float64)
    return {
        "tail_frame_start": int(tail[0]["frame"]),
        "tail_frame_end": int(tail[-1]["frame"]),
        "tail_max_displacement_peak_m": float(np.max(maxima)),
        "tail_max_displacement_mean_m": float(np.mean(maxima)),
        "tail_mean_displacement_mean_m": float(np.mean(means)),
        "tail_final_max_displacement_m": float(maxima[-1]),
    }


def _self_contact_proxy(result: ProductionResult) -> Dict[str, object]:
    counts = np.asarray(
        [row["sampled_pairs_under_2mm"] for row in result.self_contact_metrics],
        dtype=np.int64,
    )
    distances = [
        row["minimum_sample_distance_m"]
        for row in result.self_contact_metrics
        if row["minimum_sample_distance_m"] is not None
    ]
    return {
        "method": "CROSS_PANEL_VERTEX_SAMPLE_PROXY_EVERY_SUBSTEP_STATE",
        "exact_triangle_triangle_test": False,
        "tail_peak_sampled_pairs_under_2mm": int(np.max(counts[-20:])),
        "minimum_sample_distance_m": float(min(distances)) if distances else None,
    }


def qualify(result: ProductionResult, arrays: Dict[str, np.ndarray]) -> dict:
    finite = _finite_and_degenerate(result)
    body = _body_metrics(result, result.profile.body_clearance_m)
    edges = _edge_metrics(result)
    seams = _seam_metrics(result, arrays)
    convergence = _convergence_metrics(result)
    self_contact = _self_contact_proxy(result)
    numeric_gates = {
        "finite": finite["non_finite_vertices"] == 0,
        "degenerate": finite["degenerate_triangles"] == 0,
        "body_ratio": body["penetrated_vertex_ratio"] <= 0.0025,
        "body_max": body["penetration_max_m"] <= 0.004,
        "body_p99": body["penetration_p99_m"] <= 0.0015,
        "edge_max": edges["edge_stretch_max"] <= 1.75,
        "edge_p99": edges["edge_stretch_p99"] <= 1.35,
        "seam_mean": seams["seam_gap_mean_m"] <= 0.003,
        "seam_p95": seams["seam_gap_p95_m"] <= 0.008,
        "seam_open": seams["open_critical_seams"] == 0,
        "tail": convergence["tail_final_max_displacement_m"] <= 0.002,
    }
    numeric_pass = all(numeric_gates.values())
    return {
        "finite_geometry": finite,
        "body_contact": body,
        "edge_strain": edges,
        "seam_closure": seams,
        "convergence": convergence,
        "self_contact": self_contact,
        "numeric_gates": numeric_gates,
        "numeric_pass": bool(numeric_pass),
        "authority_exact_cp2": False,
        "exact_nonsewn_self_intersection_gate": False,
        "technical_pass": False,
        "terminal_decision": "HOLD_CP3_RECOVERY_NOT_CP2_BYTE_EXACT",
    }
