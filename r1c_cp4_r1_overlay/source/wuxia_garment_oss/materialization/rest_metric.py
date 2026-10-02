"""Compile arrangement-space rest metric and global simulation arrays."""
from __future__ import annotations

import numpy as np

from .model import ComponentMesh, SeamMap, canonical_sha256


def compile_rest_metric(meshes: dict[str, ComponentMesh], seam_maps: list[SeamMap]):
    order = list(meshes)
    offsets: dict[str, int] = {}
    positions_2d, positions_3d, fixed, component_ids = [], [], [], []
    triangles, edges = [], set()
    cursor = 0
    for component_index, instance_id in enumerate(order):
        mesh = meshes[instance_id]
        mesh.validate_3d()
        offsets[instance_id] = cursor
        positions_2d.append(mesh.vertices_2d)
        positions_3d.append(mesh.vertices_3d)
        fixed.append(mesh.fixed_mask)
        component_ids.append(np.full(len(mesh.vertices_3d), component_index, dtype=np.int16))
        shifted = mesh.triangles + cursor
        triangles.append(shifted)
        for triangle in shifted:
            a, b, c = map(int, triangle)
            edges.update((min(a, b), max(a, b)) for a, b in ((a, b), (b, c), (c, a)))
        cursor += len(mesh.vertices_3d)
    global_2d = np.concatenate(positions_2d).astype(np.float64)
    global_3d = np.concatenate(positions_3d).astype(np.float64)
    global_triangles = np.concatenate(triangles).astype(np.int32)
    global_edges = np.asarray(sorted(edges), dtype=np.int32)
    pattern_rest = np.linalg.norm(global_2d[global_edges[:, 0]] - global_2d[global_edges[:, 1]], axis=1)
    arrangement_rest = np.linalg.norm(global_3d[global_edges[:, 0]] - global_3d[global_edges[:, 1]], axis=1)
    seam_pairs = []
    for seam in seam_maps:
        first = seam.vertex_pairs[:, 0] + offsets[seam.component_a]
        second = seam.vertex_pairs[:, 1] + offsets[seam.component_b]
        seam_pairs.append(np.column_stack((first, second)))
    seam_pairs_array = np.concatenate(seam_pairs).astype(np.int32)
    valid = pattern_rest > 1.0e-8
    ratio = arrangement_rest[valid] / pattern_rest[valid]
    arrays = {
        "positions_2d": global_2d,
        "positions_rest": global_3d,
        "triangles": global_triangles,
        "edges": global_edges,
        "pattern_rest_lengths": pattern_rest,
        "arrangement_rest_lengths": arrangement_rest,
        "seam_pairs": seam_pairs_array,
        "fixed_mask": np.concatenate(fixed).astype(np.bool_),
        "component_ids": np.concatenate(component_ids),
        "component_order": order,
        "offsets": offsets,
    }
    receipt = {
        "contract": "CompiledArrangementRestMetric/1",
        "vertex_count": int(len(global_3d)),
        "triangle_count": int(len(global_triangles)),
        "edge_count": int(len(global_edges)),
        "seam_pair_count": int(len(seam_pairs_array)),
        "pattern_to_arrangement_ratio_p50": float(np.quantile(ratio, 0.50)),
        "pattern_to_arrangement_ratio_p95": float(np.quantile(ratio, 0.95)),
        "pattern_to_arrangement_ratio_p99": float(np.quantile(ratio, 0.99)),
        "settling_rest_owner": "ARRANGEMENT_REST_METRIC",
        "pattern_metric_preserved_as_diagnostic": True,
        "source_pattern_changed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return arrays, receipt
