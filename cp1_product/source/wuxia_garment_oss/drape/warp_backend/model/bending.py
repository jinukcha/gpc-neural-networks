"""Interior-edge ownership and rest-dihedral construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

EdgeEntry = Tuple[int, int]


@dataclass(frozen=True)
class BendingModel:
    interior_edges: np.ndarray
    interior_edge_faces: np.ndarray
    rest_edge_length: np.ndarray
    rest_dihedral: np.ndarray
    boundary_edge_count: int


def _face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("degenerate triangle prevents bending construction")
    return normals / lengths[:, None]


def build_bending_model(positions: np.ndarray, triangles: np.ndarray) -> BendingModel:
    points = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    adjacency: Dict[Tuple[int, int], List[EdgeEntry]] = {}
    for face_id, (a, b, c) in enumerate(faces):
        for left, right, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            key = tuple(sorted((int(left), int(right))))
            adjacency.setdefault(key, []).append((face_id, int(opposite)))
    if any(len(entries) > 2 for entries in adjacency.values()):
        raise ValueError("non-manifold garment edge")
    normals = _face_normals(points, faces)
    edge_rows, face_rows, lengths, angles = [], [], [], []
    boundary_count = 0
    for (v0, v1), entries in sorted(adjacency.items()):
        if len(entries) == 1:
            boundary_count += 1
            continue
        (face0, opposite0), (face1, opposite1) = entries
        direction = points[v1] - points[v0]
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-12:
            raise ValueError("zero-length interior edge")
        axis = direction / length
        n0, n1 = normals[face0], normals[face1]
        sine = float(np.dot(axis, np.cross(n0, n1)))
        cosine = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
        edge_rows.append((v0, v1, opposite0, opposite1))
        face_rows.append((face0, face1))
        lengths.append(length)
        angles.append(float(np.arctan2(sine, cosine)))
    if not edge_rows:
        raise ValueError("garment model has no interior bending edges")
    return BendingModel(
        interior_edges=np.asarray(edge_rows, dtype=np.int32),
        interior_edge_faces=np.asarray(face_rows, dtype=np.int32),
        rest_edge_length=np.asarray(lengths, dtype=np.float32),
        rest_dihedral=np.asarray(angles, dtype=np.float32),
        boundary_edge_count=boundary_count,
    )
