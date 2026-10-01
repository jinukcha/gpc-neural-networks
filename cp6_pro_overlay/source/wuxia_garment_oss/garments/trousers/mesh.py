"""Generate a welded four-panel pair-of-pants surface with waistband and gusset."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...pattern_cad.document.model import canonical_sha256


@dataclass(frozen=True)
class TrousersMesh:
    positions: np.ndarray
    triangles: np.ndarray
    triangle_panel_ids: tuple[str, ...]
    waistband_positions: np.ndarray
    waistband_triangles: np.ndarray
    gusset_positions: np.ndarray
    gusset_triangles: np.ndarray
    authority: dict


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1.0e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _section_values(pom: dict[str, float], z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outseam = pom["finished_outseam"]
    crotch_z = outseam - pom["finished_inseam"]
    hip_z = outseam - pom["hip_depth"]
    knee_z = pom["knee_height"]
    hip_half = pom["finished_hip"] * 0.25
    waist_half = pom["finished_waist"] * 0.25
    top_blend = _smoothstep(hip_z, outseam, z)
    pelvis_half = hip_half * (1.0 - top_blend) + waist_half * top_blend
    leg_blend = _smoothstep(0.0, max(knee_z, 1.0e-6), z)
    thigh_half = (pom["front_thigh_half"] + pom["back_thigh_half"]) * 0.50
    ankle_half = (pom["front_ankle_half"] + pom["back_ankle_half"]) * 0.50
    leg_half = ankle_half * (1.0 - leg_blend) + thigh_half * leg_blend
    split = 1.0 - _smoothstep(crotch_z - 0.03, crotch_z + 0.11, z)
    return pelvis_half, leg_half, split


def _panel_grid(
    pom: dict[str, float],
    side: float,
    front: bool,
    vertical_count: int,
    horizontal_count: int,
) -> np.ndarray:
    outseam = pom["finished_outseam"]
    z = np.linspace(outseam, 0.0, vertical_count)
    u = np.linspace(0.0, 1.0, horizontal_count)
    zz, uu = np.meshgrid(z, u, indexing="ij")
    pelvis_half, leg_half, split = _section_values(pom, zz)
    leg_center = side * pom["finished_hip"] * 0.14
    inner_leg = leg_center - side * leg_half * 0.42
    outer_leg = leg_center + side * leg_half * 0.58
    inner = split * inner_leg
    outer = (1.0 - split) * side * pelvis_half + split * outer_leg
    x = inner + uu * (outer - inner)
    depth_scale = pom["finished_hip"] * (0.105 if front else 0.125)
    crotch_z = outseam - pom["finished_inseam"]
    center_depth = depth_scale * (1.0 - split) * (0.52 if front else 0.68)
    depth = depth_scale * (0.82 + 0.18 * _smoothstep(crotch_z, outseam, zz))
    face = 1.0 if front else -1.0
    y = face * (center_depth * (1.0 - uu) + depth * np.sin(np.pi * uu))
    z_shape = zz + (0.004 if front else -0.003) * np.sin(np.pi * uu) * (1.0 - split)
    return np.column_stack((x.reshape(-1), y.reshape(-1), z_shape.reshape(-1)))


def _grid_triangles(vertical_count: int, horizontal_count: int) -> np.ndarray:
    rows = []
    for row in range(vertical_count - 1):
        for column in range(horizontal_count - 1):
            a = row * horizontal_count + column
            b = a + 1
            c = a + horizontal_count
            d = c + 1
            rows.extend(((a, c, b), (b, c, d)))
    return np.asarray(rows, dtype=np.int64)


def _orient(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    result = triangles.copy()
    a = positions[result[:, 0]]
    b = positions[result[:, 1]]
    c = positions[result[:, 2]]
    normal = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    radial = centroid.copy()
    radial[:, 2] = 0.0
    flip = np.sum(normal * radial, axis=1) < 0.0
    result[flip, 1], result[flip, 2] = result[flip, 2].copy(), result[flip, 1].copy()
    return result


def _weld(positions: np.ndarray, triangles: np.ndarray, tolerance: float = 1.0e-7) -> tuple[np.ndarray, np.ndarray]:
    keys = np.round(positions / tolerance).astype(np.int64)
    mapping: dict[tuple[int, int, int], int] = {}
    remap = np.empty(len(positions), dtype=np.int64)
    unique = []
    for index, key in enumerate(map(tuple, keys)):
        target = mapping.get(key)
        if target is None:
            target = len(unique)
            mapping[key] = target
            unique.append(positions[index])
        remap[index] = target
    welded = remap[triangles]
    nondegenerate = np.all(
        np.column_stack((welded[:, 0] != welded[:, 1], welded[:, 1] != welded[:, 2], welded[:, 2] != welded[:, 0])),
        axis=1,
    )
    return np.asarray(unique, dtype=np.float64), welded[nondegenerate]


def _waistband(pom: dict[str, float], segments: int = 72) -> tuple[np.ndarray, np.ndarray]:
    height = pom["waistband_height"]
    bottom = pom["finished_outseam"] - 0.005
    top = bottom + height
    rx = pom["finished_waist"] / (2.0 * np.pi) * 1.18
    ry = pom["finished_waist"] / (2.0 * np.pi) * 0.72
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    lower = np.column_stack((rx * np.cos(angles), ry * np.sin(angles), np.full(segments, bottom)))
    upper = np.column_stack((rx * np.cos(angles), ry * np.sin(angles), np.full(segments, top)))
    positions = np.vstack((lower, upper))
    triangles = []
    for index in range(segments):
        next_index = (index + 1) % segments
        triangles.extend(((index, next_index, segments + index), (next_index, segments + next_index, segments + index)))
    return positions, np.asarray(triangles, dtype=np.int64)


def _gusset(pom: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    z = pom["finished_outseam"] - pom["finished_inseam"] - 0.010
    half_width = pom["gusset_width"] * 0.5
    half_length = pom["gusset_length"] * 0.5
    positions = np.asarray(
        ((0.0, half_length, z), (half_width, 0.0, z - 0.020), (0.0, -half_length, z), (-half_width, 0.0, z - 0.020)),
        dtype=np.float64,
    )
    return positions, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)


def _edge_counts(triangles: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            key = tuple(sorted((int(a), int(b))))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _boundary_loop_count(triangles: np.ndarray) -> int:
    boundary = [edge for edge, count in _edge_counts(triangles).items() if count == 1]
    adjacency: dict[int, set[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    loops = 0
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        loops += 1
        stack = [start]
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            stack.extend(adjacency.get(vertex, ()))
    return loops


def topology_receipt(mesh: TrousersMesh) -> dict:
    positions = mesh.positions
    triangles = mesh.triangles
    a = positions[triangles[:, 0]]
    b = positions[triangles[:, 1]]
    c = positions[triangles[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) * 0.5
    counts = _edge_counts(triangles)
    payload = {
        "contract": "TrousersTopologyReceipt/1",
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(triangles)),
        "panel_triangle_count": {panel: mesh.triangle_panel_ids.count(panel) for panel in sorted(set(mesh.triangle_panel_ids))},
        "non_finite_vertex_count": int(np.count_nonzero(~np.isfinite(positions))),
        "degenerate_triangle_count": int(np.count_nonzero(area <= 1.0e-10)),
        "maximum_edge_incidence": max(counts.values(), default=0),
        "boundary_loop_count": _boundary_loop_count(triangles),
        "expected_open_boundaries": ["waist", "left_ankle", "right_ankle"],
        "waistband_triangle_count": int(len(mesh.waistband_triangles)),
        "gusset_triangle_count": int(len(mesh.gusset_triangles)),
    }
    payload["topology_pass"] = (
        payload["non_finite_vertex_count"] == 0
        and payload["degenerate_triangle_count"] == 0
        and payload["maximum_edge_incidence"] <= 2
        and payload["boundary_loop_count"] == 3
    )
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def build_trousers_mesh(pattern_authority: dict, vertical_count: int = 46, horizontal_count: int = 18) -> TrousersMesh:
    pom = pattern_authority["points_of_measure"]
    grid_triangles = _grid_triangles(vertical_count, horizontal_count)
    position_rows = []
    triangle_rows = []
    panel_ids = []
    offset = 0
    for panel_id, side, front in (
        ("front_left", -1.0, True), ("front_right", 1.0, True),
        ("back_left", -1.0, False), ("back_right", 1.0, False),
    ):
        positions = _panel_grid(pom, side, front, vertical_count, horizontal_count)
        triangles = _orient(positions, grid_triangles) + offset
        position_rows.append(positions)
        triangle_rows.append(triangles)
        panel_ids.extend([panel_id] * len(triangles))
        offset += len(positions)
    positions, triangles = _weld(np.vstack(position_rows), np.vstack(triangle_rows))
    waistband_positions, waistband_triangles = _waistband(pom)
    gusset_positions, gusset_triangles = _gusset(pom)
    authority = {
        "generator": "FOUR_PANEL_BRANCH_SURFACE_V1",
        "vertical_count": vertical_count,
        "horizontal_count": horizontal_count,
        "source_pattern_sha256": pattern_authority["authority_sha256"],
    }
    authority["authority_sha256"] = canonical_sha256(authority)
    return TrousersMesh(
        positions,
        triangles,
        tuple(panel_ids[:len(triangles)]),
        waistband_positions,
        waistband_triangles,
        gusset_positions,
        gusset_triangles,
        authority,
    )
