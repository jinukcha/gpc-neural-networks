"""Dual-area mass ownership for the static garment model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MassModel:
    triangle_area: np.ndarray
    vertex_dual_area: np.ndarray
    vertex_mass: np.ndarray
    inverse_mass: np.ndarray
    total_area_m2: float
    total_mass_kg: float


def build_dual_area_mass(
    positions: np.ndarray,
    triangles: np.ndarray,
    areal_density_kg_m2: float,
) -> MassModel:
    points = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must be [V,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangles must be [F,3]")
    if not np.isfinite(points).all() or areal_density_kg_m2 <= 0.0:
        raise ValueError("non-finite positions or non-positive density")
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    triangle_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if np.any(triangle_area <= 1.0e-12):
        raise ValueError("degenerate rest triangle prevents mass construction")
    dual = np.zeros(len(points), dtype=np.float64)
    contribution = triangle_area / 3.0
    for corner in range(3):
        np.add.at(dual, faces[:, corner], contribution)
    if np.any(dual <= 0.0):
        raise ValueError("unowned or zero-area garment vertex")
    mass = dual * float(areal_density_kg_m2)
    inverse = 1.0 / mass
    total_area = float(np.sum(triangle_area, dtype=np.float64))
    expected_mass = total_area * float(areal_density_kg_m2)
    actual_mass = float(np.sum(mass, dtype=np.float64))
    if not np.isclose(actual_mass, expected_mass, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("dual-area mass is not conservative")
    return MassModel(
        triangle_area=triangle_area.astype(np.float32),
        vertex_dual_area=dual.astype(np.float32),
        vertex_mass=mass.astype(np.float32),
        inverse_mass=inverse.astype(np.float32),
        total_area_m2=total_area,
        total_mass_kg=actual_mass,
    )
