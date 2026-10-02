"""Sparse shoulder, underarm, and elbow corrective deformation fields."""
from __future__ import annotations

import numpy as np

from .construction import ConstructedPrimitive


TARGET_NAMES = ("SHOULDER_RAISE", "UNDERARM_REACH", "ELBOW_BEND")


def sleeve_correctives(primitive: ConstructedPrimitive) -> tuple[list[np.ndarray], dict]:
    t = primitive.longitudinal.astype(np.float64)
    angle = primitive.angular.astype(np.float64)
    side_sign = -1.0 if primitive.metadata["side"] == "LEFT" else 1.0
    shoulder = np.exp(-np.square(t / 0.23))
    shoulder_delta = np.zeros_like(primitive.positions, dtype=np.float32)
    shoulder_delta[:, 1] = (0.026 * shoulder).astype(np.float32)
    shoulder_delta[:, 0] = (0.008 * side_sign * shoulder).astype(np.float32)

    underarm_profile = np.exp(-np.square((t - 0.14) / 0.18)) * np.maximum(0.0, -np.sin(angle))
    underarm_delta = np.zeros_like(primitive.positions, dtype=np.float32)
    underarm_delta[:, 1] = (-0.012 * underarm_profile).astype(np.float32)
    underarm_delta[:, 2] = (0.022 * underarm_profile).astype(np.float32)

    elbow_profile = np.exp(-np.square((t - 0.53) / 0.12))
    elbow_fold = elbow_profile * (0.35 + 0.65 * np.maximum(0.0, np.cos(angle)))
    elbow_delta = np.zeros_like(primitive.positions, dtype=np.float32)
    elbow_delta[:, 1] = (0.010 * elbow_fold).astype(np.float32)
    elbow_delta[:, 2] = (0.018 * elbow_profile * np.sin(angle)).astype(np.float32)

    targets = [shoulder_delta, underarm_delta, elbow_delta]
    return targets, corrective_receipt(primitive.name, targets)


def zero_correctives(primitive: ConstructedPrimitive) -> tuple[list[np.ndarray], dict]:
    targets = [np.zeros_like(primitive.positions, dtype=np.float32) for _ in TARGET_NAMES]
    return targets, corrective_receipt(primitive.name, targets)


def corrective_receipt(owner: str, targets: list[np.ndarray]) -> dict:
    entries = []
    for name, values in zip(TARGET_NAMES, targets, strict=True):
        magnitude = np.linalg.norm(values.astype(np.float64), axis=1)
        entries.append({
            "driver_id": name,
            "affected_vertex_count": int(np.count_nonzero(magnitude > 1.0e-8)),
            "maximum_displacement_m": float(np.max(magnitude)),
            "mean_active_displacement_m": float(np.mean(magnitude[magnitude > 1.0e-8])) if np.any(magnitude > 1.0e-8) else 0.0,
        })
    return {
        "contract": "CorrectiveDeformationSet/1",
        "owner_component": owner,
        "composition_rule": "ADDITIVE_SPARSE_LOCAL",
        "targets": entries,
        "full_pose_replacement": False,
        "secondary_motion": "NOT_EXECUTED",
        "lod_transfer": "NOT_EXECUTED",
    }


def corrective_driver_set() -> dict:
    return {
        "contract": "CorrectiveDriverSet/1",
        "drivers": [
            {"driver_id": "SHOULDER_RAISE", "semantic_joints": ["CLAVICLE", "UPPER_ARM"], "activation": "RAISE_0_TO_95_DEG"},
            {"driver_id": "UNDERARM_REACH", "semantic_joints": ["CHEST", "CLAVICLE", "UPPER_ARM"], "activation": "ABDUCTION_AND_CROSS_BODY"},
            {"driver_id": "ELBOW_BEND", "semantic_joints": ["UPPER_ARM", "FOREARM"], "activation": "FLEXION_35_TO_145_DEG"},
        ],
        "interpolation": "PIECEWISE_SMOOTH_BOUNDED",
        "supported_adapter_range": ["CANONICAL_EXACT_V1", "MIXAMO_STYLE_V1"],
    }
