"""UV-derived grain and rest-strain basis construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GrainModel:
    rest_uv_inverse: np.ndarray
    rest_triangle_basis: np.ndarray
    grain_axes: np.ndarray
    uv_determinant: np.ndarray


def _normalize(vectors: np.ndarray, label: str) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError(f"zero-length {label} grain axis")
    return vectors / lengths[:, None]


def build_grain_model(
    positions: np.ndarray,
    uv: np.ndarray,
    triangles: np.ndarray,
) -> GrainModel:
    points = np.asarray(positions, dtype=np.float64)
    tex = np.asarray(uv, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    p0, p1, p2 = (points[faces[:, index]] for index in range(3))
    t0, t1, t2 = (tex[faces[:, index]] for index in range(3))
    du1, du2 = t1 - t0, t2 - t0
    determinant = du1[:, 0] * du2[:, 1] - du2[:, 0] * du1[:, 1]
    if np.any(np.abs(determinant) <= 1.0e-12):
        raise ValueError("singular UV triangle prevents grain basis construction")
    inverse = np.empty((len(faces), 2, 2), dtype=np.float64)
    inverse[:, 0, 0] = du2[:, 1] / determinant
    inverse[:, 0, 1] = -du2[:, 0] / determinant
    inverse[:, 1, 0] = -du1[:, 1] / determinant
    inverse[:, 1, 1] = du1[:, 0] / determinant
    ds = np.stack((p1 - p0, p2 - p0), axis=2)
    deformation = np.einsum("fij,fjk->fik", ds, inverse)
    weft = deformation[:, :, 0]
    warp = deformation[:, :, 1]
    grain_axes = np.stack((_normalize(warp, "warp"), _normalize(weft, "weft")), axis=1)
    rest_basis = np.transpose(deformation, (0, 2, 1))
    if not np.isfinite(rest_basis).all() or not np.isfinite(grain_axes).all():
        raise ValueError("non-finite grain basis")
    return GrainModel(
        rest_uv_inverse=inverse.astype(np.float32),
        rest_triangle_basis=rest_basis.astype(np.float32),
        grain_axes=grain_axes.astype(np.float32),
        uv_determinant=determinant.astype(np.float32),
    )
