"""Product-topology qualification for shell, waistband, and gusset primitives."""
from __future__ import annotations

import numpy as np

from ...pattern_cad.document.model import canonical_sha256
from .mesh import TrousersMesh


def _edge_counts(triangles: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = tuple(sorted((int(first), int(second))))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _boundary_component_count(counts: dict[tuple[int, int], int]) -> int:
    adjacency: dict[int, set[int]] = {}
    for (first, second), incidence in counts.items():
        if incidence != 1:
            continue
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    visited: set[int] = set()
    components = 0
    for start in adjacency:
        if start in visited:
            continue
        components += 1
        stack = [start]
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            stack.extend(adjacency.get(vertex, ()))
    return components


def topology_receipt(mesh: TrousersMesh) -> dict:
    positions = mesh.positions
    triangles = mesh.triangles
    first = positions[triangles[:, 0]]
    second = positions[triangles[:, 1]]
    third = positions[triangles[:, 2]]
    area = np.linalg.norm(np.cross(second - first, third - first), axis=1) * 0.5
    counts = _edge_counts(triangles)
    boundary_count = _boundary_component_count(counts)
    payload = {
        "contract": "TrousersTopologyReceipt/1",
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(triangles)),
        "panel_triangle_count": {
            panel: mesh.triangle_panel_ids.count(panel)
            for panel in sorted(set(mesh.triangle_panel_ids))
        },
        "non_finite_vertex_count": int(np.count_nonzero(~np.isfinite(positions))),
        "degenerate_triangle_count": int(np.count_nonzero(area <= 1.0e-10)),
        "maximum_edge_incidence": max(counts.values(), default=0),
        "boundary_loop_count": boundary_count,
        "expected_open_boundaries": [
            "waistband_join",
            "left_ankle_hem",
            "right_ankle_hem",
            "crotch_gusset_insertion",
        ],
        "expected_boundary_loop_count": 4,
        "waistband_triangle_count": int(len(mesh.waistband_triangles)),
        "gusset_triangle_count": int(len(mesh.gusset_triangles)),
        "separate_primitive_contract": "SHELL_PLUS_WAISTBAND_PLUS_GUSSET",
    }
    payload["topology_pass"] = (
        payload["non_finite_vertex_count"] == 0
        and payload["degenerate_triangle_count"] == 0
        and payload["maximum_edge_incidence"] <= 2
        and boundary_count == payload["expected_boundary_loop_count"]
        and payload["waistband_triangle_count"] > 0
        and payload["gusset_triangle_count"] > 0
    )
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
