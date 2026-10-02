"""Technical and visual-admission qualification for CP4-R1."""
from __future__ import annotations

import numpy as np

from .body import BodyProfile, arm_datums, arm_radius, torso_radii
from .model import ComponentMesh, SeamMap, canonical_sha256
from .seams import seam_gap_metrics


def qualify_product(
    meshes: dict[str, ComponentMesh],
    seam_maps: list[SeamMap],
    arrays: dict,
    final_positions: np.ndarray,
    settle_receipt: dict,
    profile: BodyProfile,
) -> dict:
    seam = _final_seam_metrics(meshes, seam_maps, arrays, final_positions)
    strain = _edge_strain(arrays, final_positions)
    geometry = _geometry_metrics(arrays, final_positions)
    penetration = _penetration_metrics(arrays, final_positions, profile)
    gates = {
        "seam_mean": seam["mean_m"] <= 0.0015,
        "seam_p95": seam["p95_m"] <= 0.0040,
        "edge_strain_p95": strain["p95"] <= 0.080,
        "edge_strain_p99": strain["p99"] <= 0.180,
        "body_penetration_p99": penetration["p99_m"] <= 0.0010,
        "non_finite_vertex": geometry["non_finite_vertex_count"] == 0,
        "degenerate_triangle": geometry["degenerate_triangle_count"] == 0,
        "normal_inversion": geometry["normal_inversion_count"] == 0,
        "stable_tail": settle_receipt["tail_peak_frame_displacement_m"] <= 0.0015,
        "post_settle_vertex_repair": settle_receipt["post_settle_vertex_repair_count"] == 0,
    }
    payload = {
        "contract": "CP4R1TechnicalQualificationReceipt/1",
        "seam": seam,
        "edge_strain": strain,
        "geometry": geometry,
        "body_penetration": penetration,
        "settling_tail": {
            "peak_m": settle_receipt["tail_peak_frame_displacement_m"],
            "mean_m": settle_receipt["tail_mean_frame_displacement_m"],
        },
        "gates": gates,
        "technical_pass": all(gates.values()),
        "visual_review": "PENDING_BLENDER_REVIEW",
        "product_acceptance": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _final_seam_metrics(meshes, seam_maps, arrays, final_positions):
    for instance_id, offset in arrays["offsets"].items():
        count = len(meshes[instance_id].vertices_3d)
        meshes[instance_id].vertices_3d = final_positions[offset : offset + count]
    return seam_gap_metrics(meshes, seam_maps)


def _edge_strain(arrays, positions):
    edges = arrays["edges"]
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    rest = arrays["arrangement_rest_lengths"]
    valid = rest > 1.0e-9
    values = np.abs(lengths[valid] / rest[valid] - 1.0)
    return {
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _geometry_metrics(arrays, positions):
    triangles = arrays["triangles"]
    current_cross = np.cross(positions[triangles[:, 1]] - positions[triangles[:, 0]], positions[triangles[:, 2]] - positions[triangles[:, 0]])
    rest = arrays["positions_rest"]
    rest_cross = np.cross(rest[triangles[:, 1]] - rest[triangles[:, 0]], rest[triangles[:, 2]] - rest[triangles[:, 0]])
    current_area = 0.5 * np.linalg.norm(current_cross, axis=1)
    dot = np.einsum("ij,ij->i", current_cross, rest_cross)
    return {
        "non_finite_vertex_count": int(np.count_nonzero(~np.isfinite(positions).all(axis=1))),
        "degenerate_triangle_count": int(np.count_nonzero(current_area <= 1.0e-10)),
        "normal_inversion_count": int(np.count_nonzero(dot < 0.0)),
        "minimum_triangle_area_m2": float(current_area.min()),
    }


def _penetration_metrics(arrays, positions, profile):
    penetration = np.zeros(len(positions), dtype=np.float64)
    for component_index, instance_id in enumerate(arrays["component_order"]):
        indices = np.flatnonzero(arrays["component_ids"] == component_index)
        if instance_id.startswith("bodice") or instance_id == "collar":
            penetration[indices] = _torso_penetration(positions[indices], profile)
        elif instance_id.endswith("left"):
            penetration[indices] = _arm_penetration(positions[indices], profile, "LEFT")
        elif instance_id.endswith("right"):
            penetration[indices] = _arm_penetration(positions[indices], profile, "RIGHT")
    return {
        "mean_m": float(penetration.mean()),
        "p95_m": float(np.quantile(penetration, 0.95)),
        "p99_m": float(np.quantile(penetration, 0.99)),
        "maximum_m": float(penetration.max()),
        "penetrating_vertex_count": int(np.count_nonzero(penetration > 0.0)),
    }


def _torso_penetration(points, profile):
    result = np.zeros(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        rx, rz = torso_radii(profile, float(point[1]))
        ratio = np.sqrt((point[0] / rx) ** 2 + (point[2] / rz) ** 2)
        result[index] = max(0.0, (1.0 - ratio) * min(rx, rz))
    return result


def _arm_penetration(points, profile, side):
    shoulder, elbow, wrist = arm_datums(profile, side)
    result = np.zeros(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        distance, t = _nearest_distance(point, shoulder, elbow, wrist)
        result[index] = max(0.0, arm_radius(profile, t) - distance)
    return result


def _nearest_distance(point, shoulder, elbow, wrist):
    first = _segment_distance(point, shoulder, elbow, 0.0, 0.52)
    second = _segment_distance(point, elbow, wrist, 0.52, 1.0)
    return first if first[0] <= second[0] else second


def _segment_distance(point, start, end, t0, t1):
    vector = end - start
    local = float(np.dot(point - start, vector) / np.dot(vector, vector))
    local = min(max(local, 0.0), 1.0)
    distance = float(np.linalg.norm(point - (start + vector * local)))
    return distance, t0 * (1.0 - local) + t1 * local
