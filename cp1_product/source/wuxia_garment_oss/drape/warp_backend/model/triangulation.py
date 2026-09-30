"""Deterministic panel triangulation for the CP2B static garment model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.spatial import Delaunay

from ..authority import PanelAuthority
from .pattern import PatternBoundary, arrange_on_avatar, force_shoulder_gap


@dataclass(frozen=True)
class PanelMesh:
    name: str
    panel_id: int
    front: bool
    uv: np.ndarray
    positions: np.ndarray
    triangles: np.ndarray
    boundary_indices: np.ndarray
    roles: Dict[str, np.ndarray]


def _unit_disk_points(vertex_count: int, boundary_count: int) -> np.ndarray:
    if boundary_count < 3 or vertex_count <= boundary_count:
        raise ValueError("panel requires interior points and at least three boundary points")
    angles = np.arange(boundary_count, dtype=np.float64) * (2.0 * np.pi / boundary_count)
    boundary = np.column_stack((np.cos(angles), np.sin(angles)))
    interior_count = vertex_count - boundary_count
    index = np.arange(interior_count, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radius = 0.965 * np.sqrt((index + 0.5) / interior_count)
    interior = np.column_stack((radius * np.cos(index * golden_angle), radius * np.sin(index * golden_angle)))
    return np.vstack((boundary, interior))


def _boundary_target(boundary: PatternBoundary, theta: np.ndarray) -> np.ndarray:
    count = len(boundary.points)
    scaled = np.mod(theta, 2.0 * np.pi) * count / (2.0 * np.pi)
    left = np.floor(scaled).astype(np.int64) % count
    fraction = scaled - np.floor(scaled)
    right = (left + 1) % count
    return boundary.points[left] * (1.0 - fraction[:, None]) + boundary.points[right] * fraction[:, None]


def _map_to_pattern(unit_points: np.ndarray, boundary: PatternBoundary) -> np.ndarray:
    radius = np.linalg.norm(unit_points, axis=1)
    theta = np.arctan2(unit_points[:, 1], unit_points[:, 0])
    target = _boundary_target(boundary, theta)
    uv = boundary.center + radius[:, None] * (target - boundary.center)
    count = len(boundary.points)
    uv[:count] = boundary.points
    return uv


def _oriented_triangles(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    triangles = np.asarray(simplices, dtype=np.int32).copy()
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    signed = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    swap = signed < 0.0
    triangles[swap, 1], triangles[swap, 2] = triangles[swap, 2].copy(), triangles[swap, 1].copy()
    if np.any(np.abs(signed) <= 1.0e-14):
        raise ValueError("degenerate panel triangle")
    return triangles


def triangulate_panel(authority: PanelAuthority, boundary: PatternBoundary) -> PanelMesh:
    if len(boundary.points) != authority.boundary_count:
        raise ValueError(f"{authority.name}: boundary count mismatch")
    unit_points = _unit_disk_points(authority.vertex_count, authority.boundary_count)
    triangulation = Delaunay(unit_points, qhull_options="Qbb Qc Qz Q12")
    triangles = _oriented_triangles(unit_points, triangulation.simplices)
    expected = 2 * authority.vertex_count - authority.boundary_count - 2
    if len(triangles) != expected:
        raise ValueError(f"{authority.name}: expected {expected} triangles, got {len(triangles)}")
    uv = _map_to_pattern(unit_points, boundary)
    positions = arrange_on_avatar(uv, boundary.kind, boundary.front)
    boundary_indices = np.arange(authority.boundary_count, dtype=np.int32)
    force_shoulder_gap(positions, boundary_indices, boundary.roles, boundary.front)
    if not authority.front:
        triangles = triangles[:, (0, 2, 1)]
    return PanelMesh(
        name=authority.name,
        panel_id=authority.panel_id,
        front=authority.front,
        uv=uv,
        positions=positions,
        triangles=triangles,
        boundary_indices=boundary_indices,
        roles=boundary.roles,
    )
