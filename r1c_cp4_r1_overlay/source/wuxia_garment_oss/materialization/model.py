"""Data owners for triangulated and materialized garment components."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class ComponentMesh:
    instance_id: str
    component_id: str
    vertices_2d: np.ndarray
    triangles: np.ndarray
    boundary_indices: dict[str, np.ndarray]
    vertices_3d: np.ndarray | None = None
    fixed_mask: np.ndarray | None = None
    source_geometry_sha256: str = ""
    metadata: dict = field(default_factory=dict)

    def validate_2d(self) -> None:
        if self.vertices_2d.ndim != 2 or self.vertices_2d.shape[1] != 2:
            raise ValueError(f"invalid 2D vertices: {self.instance_id}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"invalid triangles: {self.instance_id}")
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(self.vertices_2d)):
            raise ValueError(f"triangle index out of range: {self.instance_id}")
        if not np.isfinite(self.vertices_2d).all():
            raise ValueError(f"non-finite 2D geometry: {self.instance_id}")
        for boundary_id, indices in self.boundary_indices.items():
            if not boundary_id or len(indices) < 2:
                raise ValueError(f"invalid boundary map: {self.instance_id}.{boundary_id}")
            if np.any(indices < 0) or np.any(indices >= len(self.vertices_2d)):
                raise ValueError(f"boundary index out of range: {self.instance_id}.{boundary_id}")

    def validate_3d(self) -> None:
        self.validate_2d()
        if self.vertices_3d is None or self.vertices_3d.shape != (len(self.vertices_2d), 3):
            raise ValueError(f"invalid 3D vertices: {self.instance_id}")
        if not np.isfinite(self.vertices_3d).all():
            raise ValueError(f"non-finite 3D geometry: {self.instance_id}")
        if self.fixed_mask is None or self.fixed_mask.shape != (len(self.vertices_2d),):
            raise ValueError(f"missing fixed mask: {self.instance_id}")

    def boundary_positions(self, boundary_id: str) -> np.ndarray:
        if self.vertices_3d is None:
            raise ValueError("3D arrangement not published")
        return self.vertices_3d[self.boundary_indices[boundary_id]]

    def to_summary(self) -> dict:
        payload = {
            "instance_id": self.instance_id,
            "component_id": self.component_id,
            "vertex_count": int(len(self.vertices_2d)),
            "triangle_count": int(len(self.triangles)),
            "boundary_vertex_counts": {
                key: int(len(value)) for key, value in sorted(self.boundary_indices.items())
            },
            "source_geometry_sha256": self.source_geometry_sha256,
            "metadata": self.metadata,
        }
        payload["summary_sha256"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class SeamMap:
    interface_id: str
    component_a: str
    boundary_a: str
    component_b: str
    boundary_b: str
    orientation: str
    vertex_pairs: np.ndarray
    direct_mean_m: float
    reversed_mean_m: float
    initial_mean_m: float
    initial_p95_m: float

    def to_dict(self) -> dict:
        payload = {
            "interface_id": self.interface_id,
            "endpoint_a": [self.component_a, self.boundary_a],
            "endpoint_b": [self.component_b, self.boundary_b],
            "orientation": self.orientation,
            "pair_count": int(len(self.vertex_pairs)),
            "direct_mean_m": self.direct_mean_m,
            "reversed_mean_m": self.reversed_mean_m,
            "initial_mean_m": self.initial_mean_m,
            "initial_p95_m": self.initial_p95_m,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
