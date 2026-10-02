"""Compile a body-clear arrangement rest before Warp settling."""
from __future__ import annotations

import numpy as np

from .body import BodyProfile
from .contact_geometry import project_point
from .model import ComponentMesh, SeamMap, canonical_sha256
from .seam_groups import seam_groups


def compile_body_clear_rest(
    meshes: dict[str, ComponentMesh],
    seam_maps: list[SeamMap],
    profile: BodyProfile,
    clearance_m: float = 0.010,
) -> dict:
    before = penetration_summary(meshes, profile)
    count = project_components(meshes, profile, clearance_m)
    groups = seam_groups(seam_maps)
    iterations = project_groups(meshes, groups, profile, clearance_m)
    relax_interiors(meshes)
    count += project_components(meshes, profile, clearance_m, movable_only=True)
    after = penetration_summary(meshes, profile)
    payload = {
        "contract": "CompiledBodyClearRestReceipt/1",
        "clearance_m": clearance_m,
        "source_pattern_changed": False,
        "post_settle_vertex_repair": False,
        "projected_vertex_operation_count": count,
        "seam_equivalence_class_count": len(groups),
        "maximum_seam_junction_size": max(len(item) for item in groups),
        "seam_junction_projection_iterations": iterations,
        "penetration_before": before,
        "penetration_after": after,
        "rest_metric_compile_admitted": after["p99_m"] <= 1.0e-9,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    if not payload["rest_metric_compile_admitted"]:
        raise RuntimeError(f"body-clear rest compilation failed: {after}")
    return payload


def project_components(meshes, profile, clearance, movable_only=False):
    count = 0
    for instance_id, mesh in meshes.items():
        indices = np.arange(len(mesh.vertices_3d), dtype=np.int32)
        if movable_only:
            indices = indices[~mesh.fixed_mask]
        for index in indices:
            projected = project_point(mesh.vertices_3d[index], instance_id, profile, clearance)
            if np.linalg.norm(projected - mesh.vertices_3d[index]) > 1.0e-12:
                mesh.vertices_3d[index] = projected
                count += 1
    return count


def project_groups(meshes, groups, profile, clearance):
    maximum_iterations = 0
    for members in groups:
        point = np.mean([meshes[owner].vertices_3d[index] for owner, index in members], axis=0)
        owners = tuple(sorted({owner for owner, _ in members}))
        iteration = 0
        for iteration in range(1, 25):
            previous = point.copy()
            for owner in owners:
                point = project_point(point, owner, profile, clearance)
            if np.linalg.norm(point - previous) <= 1.0e-10:
                break
        maximum_iterations = max(maximum_iterations, iteration)
        for owner, index in members:
            meshes[owner].vertices_3d[index] = point
            meshes[owner].fixed_mask[index] = True
    return maximum_iterations


def relax_interiors(meshes, iterations=55):
    for mesh in meshes.values():
        adjacency = adjacency_list(len(mesh.vertices_3d), mesh.triangles)
        anchor = mesh.vertices_3d.copy()
        current = anchor.copy()
        for _ in range(iterations):
            updated = current.copy()
            for index, neighbours in enumerate(adjacency):
                if mesh.fixed_mask[index] or not neighbours:
                    continue
                average = current[np.fromiter(neighbours, dtype=np.int32)].mean(axis=0)
                updated[index] = 0.68 * average + 0.32 * anchor[index]
            current = updated
        mesh.vertices_3d = current
        mesh.metadata["body_clear_rest_relax_iterations"] = iterations


def penetration_summary(meshes, profile):
    values = []
    for instance_id, mesh in meshes.items():
        for point in mesh.vertices_3d:
            values.append(float(np.linalg.norm(project_point(point, instance_id, profile, 0.0) - point)))
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_m": float(array.mean()),
        "p95_m": float(np.quantile(array, 0.95)),
        "p99_m": float(np.quantile(array, 0.99)),
        "maximum_m": float(array.max()),
        "penetrating_vertex_count": int(np.count_nonzero(array > 1.0e-12)),
    }


def adjacency_list(vertex_count, triangles):
    result = [set() for _ in range(vertex_count)]
    for triangle in triangles:
        a, b, c = map(int, triangle)
        result[a].update((b, c))
        result[b].update((a, c))
        result[c].update((a, b))
    return result
