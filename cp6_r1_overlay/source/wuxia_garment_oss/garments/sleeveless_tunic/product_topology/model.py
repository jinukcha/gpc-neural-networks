"""Contracts for the CP6-R1 feature-complete tunic product mesh."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


LAYER_SHELL = 0
LAYER_LINING = 1
LAYER_FACING = 2
LAYER_INTERFACING = 3
LAYER_HARDWARE = 4

FEATURE_BASE = 0
FEATURE_DART = 1
FEATURE_PLEAT = 2
FEATURE_GATHER = 3
FEATURE_GUSSET = 4
FEATURE_FACING = 5
FEATURE_INTERFACING = 6
FEATURE_CLOSURE = 7


@dataclass(frozen=True)
class ComponentRange:
    name: str
    vertex_start: int
    vertex_count: int
    triangle_start: int
    triangle_count: int
    layer_id: int
    feature_id: int
    source_owner_id: str
    bonded: bool = False

    @property
    def vertex_stop(self) -> int:
        return self.vertex_start + self.vertex_count

    @property
    def triangle_stop(self) -> int:
        return self.triangle_start + self.triangle_count

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vertex_start": self.vertex_start,
            "vertex_count": self.vertex_count,
            "triangle_start": self.triangle_start,
            "triangle_count": self.triangle_count,
            "layer_id": self.layer_id,
            "feature_id": self.feature_id,
            "source_owner_id": self.source_owner_id,
            "bonded": self.bonded,
        }


@dataclass(frozen=True)
class TunicProductMesh:
    positions: np.ndarray
    triangles: np.ndarray
    uv: np.ndarray
    panel_ids: np.ndarray
    face_panel_ids: np.ndarray
    layer_ids: np.ndarray
    face_layer_ids: np.ndarray
    feature_ids: np.ndarray
    face_feature_ids: np.ndarray
    driver_indices: np.ndarray
    driver_weights: np.ndarray
    driver_offsets: np.ndarray
    seam_pairs: np.ndarray
    seam_ids: np.ndarray
    seam_rest_lengths: np.ndarray
    seam_names: tuple[str, ...]
    attachment_indices: np.ndarray
    components: tuple[ComponentRange, ...]
    base_vertex_count: int
    authority: dict

    def component(self, name: str) -> ComponentRange:
        return next(item for item in self.components if item.name == name)

    def component_positions(self, name: str, positions: np.ndarray | None = None) -> np.ndarray:
        owner = self.component(name)
        source = self.positions if positions is None else positions
        return source[owner.vertex_start:owner.vertex_stop]

    def component_triangles(self, name: str) -> np.ndarray:
        owner = self.component(name)
        value = self.triangles[owner.triangle_start:owner.triangle_stop]
        return value - owner.vertex_start

    def component_uv(self, name: str) -> np.ndarray:
        owner = self.component(name)
        return self.uv[owner.vertex_start:owner.vertex_stop]

    def to_receipt(self) -> dict:
        return {
            "contract": "FeatureCompleteTunicTopologyReceipt/1",
            "vertex_count": int(len(self.positions)),
            "triangle_count": int(len(self.triangles)),
            "base_vertex_count": self.base_vertex_count,
            "added_vertex_count": int(len(self.positions) - self.base_vertex_count),
            "component_count": len(self.components),
            "components": [item.to_dict() for item in self.components],
            "seam_pair_count": int(len(self.seam_pairs)),
            "seam_group_count": len(self.seam_names),
            "attachment_count": int(len(self.attachment_indices)),
            "layer_ids": sorted(int(value) for value in np.unique(self.layer_ids)),
            "feature_ids": sorted(int(value) for value in np.unique(self.feature_ids)),
            "authority": self.authority,
        }


def component_dicts(components: Sequence[ComponentRange]) -> list[dict]:
    return [item.to_dict() for item in components]
