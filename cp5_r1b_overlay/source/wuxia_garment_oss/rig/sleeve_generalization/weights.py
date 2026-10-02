"""Component-owned upper-arm, forearm, and robe weight transfer."""
from __future__ import annotations

import numpy as np

from wuxia_garment_oss.rig.game_product.glb_skin import BONE_INDEX, _normalize

from .construction import ConstructedPrimitive


def _sleeve_row(prefix: str, t: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    underarm = max(0.0, -np.sin(angle))
    if t <= 0.16:
        cap = t / 0.16
        return _normalize((
            ("CHEST", 0.18 * (1.0 - cap) * underarm),
            (f"{prefix}_CLAVICLE", 0.38 * (1.0 - 0.45 * cap)),
            (f"{prefix}_UPPER_ARM", 0.62 + 0.18 * cap),
        ))
    if t <= 0.48:
        blend = (t - 0.16) / 0.32
        return _normalize(((f"{prefix}_CLAVICLE", 0.10 * (1.0 - blend)), (f"{prefix}_UPPER_ARM", 0.88 - 0.18 * blend), (f"{prefix}_FOREARM", 0.02 + 0.28 * blend)))
    if t <= 0.68:
        blend = (t - 0.48) / 0.20
        return _normalize(((f"{prefix}_UPPER_ARM", 0.70 * (1.0 - blend)), (f"{prefix}_FOREARM", 0.30 + 0.65 * blend), (f"{prefix}_HAND", 0.05 * blend)))
    blend = (t - 0.68) / 0.32
    return _normalize(((f"{prefix}_FOREARM", 0.95 - 0.35 * blend), (f"{prefix}_HAND", 0.05 + 0.35 * blend)))


def sleeve_weight_field(primitive: ConstructedPrimitive) -> tuple[np.ndarray, np.ndarray, dict]:
    prefix = "L" if primitive.metadata["side"] == "LEFT" else "R"
    joints = np.empty((len(primitive.positions), 4), dtype=np.uint16)
    weights = np.empty((len(primitive.positions), 4), dtype=np.float32)
    for index, (t, angle) in enumerate(zip(primitive.longitudinal, primitive.angular, strict=True)):
        joint_row, weight_row = _sleeve_row(prefix, float(t), float(angle))
        joints[index], weights[index] = joint_row, weight_row
    metrics = _metrics(joints, weights, prefix)
    return joints, weights, metrics


def robe_skirt_weight_field(primitive: ConstructedPrimitive) -> tuple[np.ndarray, np.ndarray, dict]:
    joints = np.empty((len(primitive.positions), 4), dtype=np.uint16)
    weights = np.empty((len(primitive.positions), 4), dtype=np.float32)
    for index, (t, angle) in enumerate(zip(primitive.longitudinal, primitive.angular, strict=True)):
        left = 0.5 + 0.5 * np.cos(float(angle))
        right = 1.0 - left
        lower = float(t)
        joints[index], weights[index] = _normalize((
            ("PELVIS", 0.72 - 0.30 * lower),
            ("SPINE_01", 0.20 * (1.0 - lower)),
            ("L_THIGH", 0.38 * lower * left),
            ("R_THIGH", 0.38 * lower * right),
        ))
    metrics = _metrics(joints, weights, "SKIRT")
    return joints, weights, metrics


def _metrics(joints: np.ndarray, weights: np.ndarray, owner: str) -> dict:
    sums = np.sum(weights, axis=1, dtype=np.float64)
    nonzero = np.count_nonzero(weights > 0.0, axis=1)
    active = {int(value) for value in joints[weights > 0.0]}
    semantic = sorted(name for name, index in BONE_INDEX.items() if index in active)
    return {
        "contract": "SkinWeightField/1",
        "owner": owner,
        "vertex_count": int(len(weights)),
        "maximum_influences": int(np.max(nonzero)),
        "maximum_weight_sum_error": float(np.max(np.abs(sums - 1.0))),
        "zero_weight_vertex_count": int(np.count_nonzero(sums <= 0.0)),
        "negative_weight_count": int(np.count_nonzero(weights < 0.0)),
        "semantic_bones": semantic,
        "transfer_mode": "COMPONENT_LOCAL_SEMANTIC_CHAIN",
    }
