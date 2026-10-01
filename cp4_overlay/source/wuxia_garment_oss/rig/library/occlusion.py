"""Compile per-triangle body hide masks from outfit semantic coverage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import canonical_sha256


_REGION_NAME_TO_ID = {
    "TORSO": 0,
    "NECK": 1,
    "HEAD": 2,
    "LEFT_ARM": 3,
    "RIGHT_ARM": 4,
    "LEFT_HAND": 5,
    "RIGHT_HAND": 6,
    "LEFT_LEG": 7,
    "RIGHT_LEG": 8,
    "LEFT_FOOT": 9,
    "RIGHT_FOOT": 10,
}


def _first_array(data: np.lib.npyio.NpzFile, names: tuple[str, ...]):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def _load_body_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = sorted((root / "build/rig_cp0/body_skin_source").glob("*.npz"))
    if not candidates:
        raise FileNotFoundError("body skin-source NPZ not found")
    with np.load(candidates[0], allow_pickle=False) as data:
        positions = _first_array(data, ("positions", "vertices", "body_positions"))
        triangles = _first_array(data, ("triangles", "faces", "body_triangles"))
        region_ids = _first_array(data, ("semantic_region_ids", "region_ids", "vertex_region_ids"))
    if positions is None or triangles is None:
        raise KeyError("body skin-source geometry arrays missing")
    if region_ids is None:
        region_ids = infer_regions(np.asarray(positions, dtype=np.float64))
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(triangles, dtype=np.int32),
        np.asarray(region_ids, dtype=np.int16),
    )


def infer_regions(positions: np.ndarray) -> np.ndarray:
    """Fallback region inference for the canonical CP0 fixture only."""
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    result = np.full(len(positions), _REGION_NAME_TO_ID["TORSO"], dtype=np.int16)
    result[z > 1.58] = _REGION_NAME_TO_ID["HEAD"]
    result[(z > 1.42) & (z <= 1.58)] = _REGION_NAME_TO_ID["NECK"]
    arm = (np.abs(x) > 0.30) & (z > 0.78)
    result[arm & (x < 0)] = _REGION_NAME_TO_ID["LEFT_ARM"]
    result[arm & (x >= 0)] = _REGION_NAME_TO_ID["RIGHT_ARM"]
    leg = z < 0.86
    result[leg & (x < 0)] = _REGION_NAME_TO_ID["LEFT_LEG"]
    result[leg & (x >= 0)] = _REGION_NAME_TO_ID["RIGHT_LEG"]
    foot = z < 0.12
    result[foot & (x < 0)] = _REGION_NAME_TO_ID["LEFT_FOOT"]
    result[foot & (x >= 0)] = _REGION_NAME_TO_ID["RIGHT_FOOT"]
    return result


def _coverage_to_body_regions(coverage: tuple[str, ...]) -> set[str]:
    mapping = {
        "TORSO": {"TORSO"},
        "CHEST": {"TORSO"},
        "WAIST": {"TORSO"},
        "PELVIS": {"TORSO", "LEFT_LEG", "RIGHT_LEG"},
        "LEFT_THIGH": {"LEFT_LEG"},
        "RIGHT_THIGH": {"RIGHT_LEG"},
        "LEFT_KNEE": {"LEFT_LEG"},
        "RIGHT_KNEE": {"RIGHT_LEG"},
        "LEFT_CALF": {"LEFT_LEG"},
        "RIGHT_CALF": {"RIGHT_LEG"},
    }
    result: set[str] = set()
    for item in coverage:
        result.update(mapping.get(item, {item} if item in _REGION_NAME_TO_ID else set()))
    return result


def compile_body_hide_mask(root: Path, hidden_regions: tuple[str, ...], safety_regions: tuple[str, ...]) -> dict:
    positions, triangles, vertex_regions = _load_body_arrays(root)
    triangle_regions = np.stack([vertex_regions[triangles[:, index]] for index in range(3)], axis=1)
    hidden_body = _coverage_to_body_regions(hidden_regions)
    safety_body = _coverage_to_body_regions(safety_regions)
    hide_ids = {_REGION_NAME_TO_ID[item] for item in hidden_body - safety_body}
    hide = np.asarray([all(int(value) in hide_ids for value in row) for row in triangle_regions], dtype=np.bool_)
    boundary = np.asarray([any(int(value) in hide_ids for value in row) and not all(int(value) in hide_ids for value in row) for row in triangle_regions], dtype=np.bool_)
    hide[boundary] = False
    target = root / "build/rig_cp4/body_occlusion"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "body_hide_mask.npz",
        hide_triangle_mask=hide,
        triangle_region_ids=triangle_regions,
        positions=positions.astype(np.float32),
        triangles=triangles,
    )
    payload = {
        "contract": "BodyOcclusionMask/1",
        "source_vertex_count": int(len(positions)),
        "source_triangle_count": int(len(triangles)),
        "hidden_semantic_regions": sorted(hidden_body),
        "preserved_safety_regions": sorted(safety_body),
        "hidden_triangle_count": int(np.count_nonzero(hide)),
        "boundary_preserved_triangle_count": int(np.count_nonzero(boundary)),
        "visible_triangle_count": int(len(triangles) - np.count_nonzero(hide)),
        "mask_path": "build/rig_cp4/body_occlusion/body_hide_mask.npz",
        "mask_pass": bool(np.count_nonzero(hide) > 0 and np.count_nonzero(hide) < len(triangles)),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    (target / "body_hide_mask.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
