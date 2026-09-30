"""Array-level contract for WarpGarmentModelPackage."""
from __future__ import annotations

import hashlib
from typing import Mapping, Tuple

import numpy as np

from ..authority import (
    EXPECTED_ATTACHMENTS,
    EXPECTED_PANEL_COUNT,
    EXPECTED_SEAM_PAIRS,
    EXPECTED_TRIANGLES,
    EXPECTED_VERTICES,
)

ArraySpec = Tuple[np.dtype, Tuple[int | None, ...]]


_SPECS: Mapping[str, ArraySpec] = {
    "positions_initial": (np.dtype("float32"), (EXPECTED_VERTICES, 3)),
    "velocities_initial": (np.dtype("float32"), (EXPECTED_VERTICES, 3)),
    "triangles": (np.dtype("int32"), (EXPECTED_TRIANGLES, 3)),
    "uv": (np.dtype("float32"), (EXPECTED_VERTICES, 2)),
    "panel_ids": (np.dtype("int32"), (EXPECTED_VERTICES,)),
    "face_panel_ids": (np.dtype("int32"), (EXPECTED_TRIANGLES,)),
    "triangle_area": (np.dtype("float32"), (EXPECTED_TRIANGLES,)),
    "vertex_dual_area": (np.dtype("float32"), (EXPECTED_VERTICES,)),
    "vertex_mass": (np.dtype("float32"), (EXPECTED_VERTICES,)),
    "inverse_mass": (np.dtype("float32"), (EXPECTED_VERTICES,)),
    "rest_uv_inverse": (np.dtype("float32"), (EXPECTED_TRIANGLES, 2, 2)),
    "rest_triangle_basis": (np.dtype("float32"), (EXPECTED_TRIANGLES, 2, 3)),
    "grain_axes": (np.dtype("float32"), (EXPECTED_TRIANGLES, 2, 3)),
    "uv_determinant": (np.dtype("float32"), (EXPECTED_TRIANGLES,)),
    "interior_edges": (np.dtype("int32"), (None, 4)),
    "interior_edge_faces": (np.dtype("int32"), (None, 2)),
    "rest_edge_length": (np.dtype("float32"), (None,)),
    "rest_dihedral": (np.dtype("float32"), (None,)),
    "seam_pairs": (np.dtype("int32"), (EXPECTED_SEAM_PAIRS, 2)),
    "seam_ids": (np.dtype("int32"), (EXPECTED_SEAM_PAIRS,)),
    "seam_rest_length": (np.dtype("float32"), (EXPECTED_SEAM_PAIRS,)),
    "seam_group_offsets": (np.dtype("int32"), (9,)),
    "attachment_indices": (np.dtype("int32"), (EXPECTED_ATTACHMENTS,)),
    "attachment_targets": (np.dtype("float32"), (EXPECTED_ATTACHMENTS, 3)),
    "attachment_ids": (np.dtype("int32"), (EXPECTED_ATTACHMENTS,)),
    "attachment_rest_distance": (np.dtype("float32"), (EXPECTED_ATTACHMENTS,)),
}


def _shape_matches(actual: Tuple[int, ...], expected: Tuple[int | None, ...]) -> bool:
    return len(actual) == len(expected) and all(want is None or got == want for got, want in zip(actual, expected))


def canonical_array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_model_arrays(arrays: Mapping[str, np.ndarray]) -> dict:
    missing = sorted(set(_SPECS) - set(arrays))
    extra = sorted(set(arrays) - set(_SPECS))
    if missing or extra:
        raise ValueError(f"model array keys differ; missing={missing}, extra={extra}")
    for name, (dtype, shape) in _SPECS.items():
        array = arrays[name]
        if array.dtype != dtype or not _shape_matches(array.shape, shape):
            raise ValueError(f"{name}: expected {dtype} {shape}, got {array.dtype} {array.shape}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"{name}: non-finite values")
    if not np.array_equal(np.unique(arrays["panel_ids"]), np.arange(EXPECTED_PANEL_COUNT, dtype=np.int32)):
        raise ValueError("vertex panel ownership is incomplete")
    if not np.array_equal(np.unique(arrays["face_panel_ids"]), np.arange(EXPECTED_PANEL_COUNT, dtype=np.int32)):
        raise ValueError("face panel ownership is incomplete")
    if np.min(arrays["triangles"]) < 0 or np.max(arrays["triangles"]) >= EXPECTED_VERTICES:
        raise ValueError("triangle index out of range")
    if np.min(arrays["interior_edges"]) < 0 or np.max(arrays["interior_edges"]) >= EXPECTED_VERTICES:
        raise ValueError("bending index out of range")
    if np.any(arrays["vertex_mass"] <= 0.0) or np.any(arrays["inverse_mass"] <= 0.0):
        raise ValueError("non-positive mass ownership")
    if int(arrays["seam_group_offsets"][-1]) != EXPECTED_SEAM_PAIRS:
        raise ValueError("seam group offsets do not close")
    return {
        "schema": "WarpGarmentModelPackage/1",
        "arrays": len(arrays),
        "vertices": EXPECTED_VERTICES,
        "triangles": EXPECTED_TRIANGLES,
        "seam_pairs": EXPECTED_SEAM_PAIRS,
        "attachments": EXPECTED_ATTACHMENTS,
        "interior_edges": int(len(arrays["interior_edges"])),
        "canonical_array_sha256": canonical_array_hash(arrays),
    }
