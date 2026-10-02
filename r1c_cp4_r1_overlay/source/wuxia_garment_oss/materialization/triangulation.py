"""Component-aware constrained-outline triangulation."""
from __future__ import annotations

import math

from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.spatial import Delaunay

from .curves import boundary_length, sample_outline
from .model import ComponentMesh


def seam_boundary_counts(package: dict, spacing_m: float = 0.024) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for seam in package["seams"]:
        count = max(7, int(math.ceil(max(seam["length_a_m"], seam["length_b_m"]) / spacing_m)) + 1)
        endpoint_a = seam["endpoint_a"]
        endpoint_b = seam["endpoint_b"]
        counts[(endpoint_a["component_instance_id"], endpoint_a["boundary_id"])] = count
        counts[(endpoint_b["component_instance_id"], endpoint_b["boundary_id"])] = count
    return counts


def _deduplicate(points: np.ndarray, tolerance: float = 1.0e-9) -> tuple[np.ndarray, np.ndarray]:
    unique: list[np.ndarray] = []
    remap = np.empty(len(points), dtype=np.int32)
    buckets: dict[tuple[int, int], list[int]] = {}
    scale = 1.0 / tolerance
    for index, point in enumerate(points):
        key = (int(round(point[0] * scale)), int(round(point[1] * scale)))
        found = None
        for candidate in buckets.get(key, []):
            if np.linalg.norm(unique[candidate] - point) <= tolerance:
                found = candidate
                break
        if found is None:
            found = len(unique)
            unique.append(point)
            buckets.setdefault(key, []).append(found)
        remap[index] = found
    return np.asarray(unique, dtype=np.float64), remap


def _interior_points(outline: np.ndarray, spacing_m: float) -> np.ndarray:
    lower = outline.min(axis=0)
    upper = outline.max(axis=0)
    xs = np.arange(lower[0] + spacing_m, upper[0], spacing_m)
    ys = np.arange(lower[1] + spacing_m, upper[1], spacing_m)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    candidates = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    inside = PolygonPath(outline, closed=True).contains_points(candidates, radius=-1.0e-7)
    return candidates[inside]


def _filter_triangles(vertices: np.ndarray, triangles: np.ndarray, outline: np.ndarray) -> np.ndarray:
    path = PolygonPath(outline, closed=True)
    centroids = vertices[triangles].mean(axis=1)
    inside = path.contains_points(centroids, radius=-1.0e-9)
    selected = triangles[inside]
    a = vertices[selected[:, 1]] - vertices[selected[:, 0]]
    b = vertices[selected[:, 2]] - vertices[selected[:, 0]]
    signed = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    nondegenerate = np.abs(signed) > 1.0e-12
    selected = selected[nondegenerate]
    signed = signed[nondegenerate]
    selected[signed < 0.0, [1, 2]] = selected[signed < 0.0, [2, 1]]
    return selected.astype(np.int32)


def triangulate_component(
    geometry: dict,
    counts: dict[tuple[str, str], int],
    interior_spacing_m: float = 0.030,
) -> ComponentMesh:
    local_counts = {
        boundary_id: count
        for (instance_id, boundary_id), count in counts.items()
        if instance_id == geometry["instance_id"]
    }
    outline, boundary_map = sample_outline(geometry, local_counts)
    interior = _interior_points(outline, interior_spacing_m)
    combined = np.concatenate((outline, interior), axis=0)
    vertices, remap = _deduplicate(combined)
    outline_remap = remap[: len(outline)]
    remapped_boundaries = {
        boundary_id: outline_remap[indices]
        for boundary_id, indices in boundary_map.items()
    }
    if len(vertices) < 3:
        raise ValueError(f"insufficient triangulation points: {geometry['instance_id']}")
    delaunay = Delaunay(vertices, qhull_options="Qbb Qc Qz Q12")
    triangles = _filter_triangles(vertices, np.asarray(delaunay.simplices, dtype=np.int32), vertices[outline_remap])
    fixed = np.zeros(len(vertices), dtype=np.bool_)
    for indices in remapped_boundaries.values():
        fixed[indices] = True
    mesh = ComponentMesh(
        instance_id=geometry["instance_id"],
        component_id=geometry["component_id"],
        vertices_2d=vertices,
        triangles=triangles,
        boundary_indices=remapped_boundaries,
        fixed_mask=fixed,
        source_geometry_sha256=geometry["geometry_sha256"],
        metadata={
            "triangulation": "SCIPY_DELAUNAY_FILTERED_BY_EXACT_OUTLINE",
            "interior_spacing_m": interior_spacing_m,
            "outline_vertex_count": int(len(np.unique(outline_remap))),
        },
    )
    mesh.validate_2d()
    return mesh


def triangulate_snapshot(snapshot: dict) -> tuple[dict[str, ComponentMesh], dict]:
    package = snapshot["assembled_package"]
    counts = seam_boundary_counts(package)
    meshes = {
        instance_id: triangulate_component(geometry, counts)
        for instance_id, geometry in snapshot["geometry_by_instance"].items()
        if instance_id in {item["instance_id"] for item in package["component_instances"]}
    }
    receipt = {
        "contract": "ComponentTriangulationReceipt/1",
        "component_count": len(meshes),
        "vertex_count": sum(len(mesh.vertices_2d) for mesh in meshes.values()),
        "triangle_count": sum(len(mesh.triangles) for mesh in meshes.values()),
        "components": [mesh.to_summary() for mesh in meshes.values()],
        "triangulation_executed": True,
        "cloth_simulation_executed": False,
    }
    return meshes, receipt
