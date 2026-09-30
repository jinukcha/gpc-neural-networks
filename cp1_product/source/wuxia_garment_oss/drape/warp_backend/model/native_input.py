"""Assemble the four-panel static input consumed by the Warp model compiler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from ..authority import (
    EXPECTED_ATTACHMENTS,
    EXPECTED_PANEL_COUNT,
    EXPECTED_SEAM_PAIRS,
    EXPECTED_TRIANGLES,
    EXPECTED_VERTICES,
    PANELS,
    SEAM_COUNTS,
)
from .pattern import bodice_boundary, skirt_boundary
from .triangulation import PanelMesh, triangulate_panel


@dataclass(frozen=True)
class NativeInput:
    positions: np.ndarray
    velocities: np.ndarray
    triangles: np.ndarray
    uv: np.ndarray
    panel_ids: np.ndarray
    face_panel_ids: np.ndarray
    seam_pairs: np.ndarray
    seam_ids: np.ndarray
    seam_names: Tuple[str, ...]
    attachment_indices: np.ndarray
    attachment_targets: np.ndarray
    attachment_ids: np.ndarray


def _panel_meshes() -> Dict[str, PanelMesh]:
    result: Dict[str, PanelMesh] = {}
    for authority in PANELS:
        boundary = bodice_boundary(authority.front) if authority.name.startswith("bodice") else skirt_boundary(authority.front)
        result[authority.name] = triangulate_panel(authority, boundary)
    return result


def _offset_meshes(meshes: Mapping[str, PanelMesh]) -> Tuple[dict, Dict[str, int]]:
    positions, uv, triangles = [], [], []
    panel_ids, face_panel_ids = [], []
    offsets: Dict[str, int] = {}
    for authority in PANELS:
        mesh = meshes[authority.name]
        offset = sum(len(item) for item in positions)
        offsets[mesh.name] = offset
        positions.append(mesh.positions)
        uv.append(mesh.uv)
        triangles.append(mesh.triangles + offset)
        panel_ids.append(np.full(len(mesh.positions), mesh.panel_id, dtype=np.int32))
        face_panel_ids.append(np.full(len(mesh.triangles), mesh.panel_id, dtype=np.int32))
    arrays = {
        "positions": np.vstack(positions).astype(np.float32),
        "uv": np.vstack(uv).astype(np.float32),
        "triangles": np.vstack(triangles).astype(np.int32),
        "panel_ids": np.concatenate(panel_ids),
        "face_panel_ids": np.concatenate(face_panel_ids),
    }
    return arrays, offsets


def _global_role(mesh: PanelMesh, offset: int, role: str, reverse: bool = False) -> np.ndarray:
    local = mesh.boundary_indices[mesh.roles[role]]
    if reverse:
        local = local[::-1]
    return local.astype(np.int32) + int(offset)


def _pair(name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    expected = SEAM_COUNTS[name]
    if len(a) != expected or len(b) != expected:
        raise ValueError(f"{name}: expected {expected} endpoints, got {len(a)} and {len(b)}")
    return np.column_stack((a, b)).astype(np.int32)


def _seams(meshes: Mapping[str, PanelMesh], offsets: Mapping[str, int]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    bf, bb = meshes["bodice_front"], meshes["bodice_back"]
    sf, sb = meshes["skirt_front"], meshes["skirt_back"]
    role = lambda mesh, key, rev=False: _global_role(mesh, offsets[mesh.name], key, rev)
    named: Sequence[Tuple[str, np.ndarray]] = (
        ("shoulder_left", _pair("shoulder_left", role(bf, "shoulder_left"), role(bb, "shoulder_left"))),
        ("shoulder_right", _pair("shoulder_right", role(bf, "shoulder_right"), role(bb, "shoulder_right"))),
        ("bodice_side_left", _pair("bodice_side_left", role(bf, "side_left"), role(bb, "side_left"))),
        ("bodice_side_right", _pair("bodice_side_right", role(bf, "side_right"), role(bb, "side_right"))),
        ("waist_front", _pair("waist_front", role(bf, "waist"), role(sf, "waist", True))),
        ("waist_back", _pair("waist_back", role(bb, "waist"), role(sb, "waist", True))),
        ("skirt_side_left", _pair("skirt_side_left", role(sf, "side_left"), role(sb, "side_left"))),
        ("skirt_side_right", _pair("skirt_side_right", role(sf, "side_right"), role(sb, "side_right"))),
    )
    pairs = np.vstack([value for _name, value in named]).astype(np.int32)
    ids = np.concatenate([np.full(len(value), index, dtype=np.int32) for index, (_name, value) in enumerate(named)])
    names = tuple(name for name, _value in named)
    if len(pairs) != EXPECTED_SEAM_PAIRS:
        raise ValueError(f"expected {EXPECTED_SEAM_PAIRS} seam pairs, got {len(pairs)}")
    if len({tuple(sorted(map(int, pair))) for pair in pairs}) != len(pairs):
        raise ValueError("duplicate seam pair")
    return pairs, ids, names


def _attachments(positions: np.ndarray, seam_pairs: np.ndarray, seam_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shoulder_mask = seam_ids < 2
    shoulder_pairs = seam_pairs[shoulder_mask]
    indices, targets, ids = [], [], []
    for constraint_id, (a, b) in enumerate(shoulder_pairs):
        target = 0.5 * (positions[a] + positions[b])
        indices.extend((int(a), int(b)))
        targets.extend((target, target))
        ids.extend((constraint_id, constraint_id))
    index_array = np.asarray(indices, dtype=np.int32)
    if len(index_array) != EXPECTED_ATTACHMENTS or len(np.unique(index_array)) != EXPECTED_ATTACHMENTS:
        raise ValueError("shoulder attachment ownership is not exactly 52 unique vertices")
    return index_array, np.asarray(targets, dtype=np.float32), np.asarray(ids, dtype=np.int32)


def build_native_input() -> NativeInput:
    meshes = _panel_meshes()
    arrays, offsets = _offset_meshes(meshes)
    seam_pairs, seam_ids, seam_names = _seams(meshes, offsets)
    attachment_indices, attachment_targets, attachment_ids = _attachments(
        arrays["positions"], seam_pairs, seam_ids
    )
    if len(meshes) != EXPECTED_PANEL_COUNT:
        raise ValueError("panel count mismatch")
    if len(arrays["positions"]) != EXPECTED_VERTICES:
        raise ValueError("vertex count mismatch")
    if len(arrays["triangles"]) != EXPECTED_TRIANGLES:
        raise ValueError("triangle count mismatch")
    return NativeInput(
        positions=arrays["positions"],
        velocities=np.zeros_like(arrays["positions"], dtype=np.float32),
        triangles=arrays["triangles"],
        uv=arrays["uv"],
        panel_ids=arrays["panel_ids"],
        face_panel_ids=arrays["face_panel_ids"],
        seam_pairs=seam_pairs,
        seam_ids=seam_ids,
        seam_names=seam_names,
        attachment_indices=attachment_indices,
        attachment_targets=attachment_targets,
        attachment_ids=attachment_ids,
    )
