"""Canonical direct-arm measurement contract for CP5 sleeve products."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArmMeasurementSet:
    shoulder_to_elbow_m: float
    elbow_to_wrist_m: float
    sleeve_length_m: float
    upper_arm_circumference_m: float
    elbow_circumference_m: float
    wrist_circumference_m: float
    armscye_circumference_m: float
    sleeve_cap_height_m: float
    cap_ease_ratio: float
    canonical_height_m: float

    def validate(self) -> None:
        values = tuple(asdict(self).values())
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("arm measurements must be finite and positive")
        if not 1.01 <= self.cap_ease_ratio <= 1.12:
            raise ValueError("sleeve-cap ease is outside the bounded construction range")
        expected = self.shoulder_to_elbow_m + self.elbow_to_wrist_m
        if abs(self.sleeve_length_m - expected) > 1.0e-9:
            raise ValueError("sleeve length is not the direct joint-chain length")

    def to_dict(self, joints: dict[str, list[float]]) -> dict:
        self.validate()
        payload = {
            "contract": "ArmMeasurementSet/1",
            "source": "CP0_CANONICAL_SKELETON_AND_CP4_SEMANTIC_BODY_PROXY",
            "measurement_mode": "DIRECT_SEMANTIC_DATUM",
            "units": "m",
            "measurements": asdict(self),
            "joint_datums": joints,
            "mesh_scaling": "FORBIDDEN",
        }
        payload["measurement_sha256"] = _canonical_sha256(payload)
        return payload


def _canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _joint_map(payload: dict) -> dict[str, np.ndarray]:
    source = payload.get("global_joint_positions")
    if not source:
        source = {item["semantic_id"]: item["global_translation"] for item in payload["bones"]}
    required = ("ROOT", "HEAD", "L_UPPER_ARM", "L_FOREARM", "L_HAND", "R_UPPER_ARM", "R_FOREARM", "R_HAND")
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"canonical skeleton missing arm datums: {missing}")
    return {name: np.asarray(value, dtype=np.float64) for name, value in source.items()}


def _symmetric_length(joints: dict[str, np.ndarray], start: str, end: str) -> float:
    left = _distance(joints[f"L_{start}"], joints[f"L_{end}"])
    right = _distance(joints[f"R_{start}"], joints[f"R_{end}"])
    if abs(left - right) > 1.0e-6:
        raise ValueError(f"canonical arm asymmetry exceeds tolerance: {start}->{end}")
    return 0.5 * (left + right)


def load_arm_measurements(skeleton_path: Path) -> tuple[ArmMeasurementSet, dict[str, list[float]]]:
    payload = json.loads(skeleton_path.read_text(encoding="utf-8"))
    joints = _joint_map(payload)
    upper = _symmetric_length(joints, "UPPER_ARM", "FOREARM")
    forearm = _symmetric_length(joints, "FOREARM", "HAND")
    canonical_height = float(joints["HEAD"][1] - joints["ROOT"][1])
    arm_scale = upper + forearm
    measurements = ArmMeasurementSet(
        shoulder_to_elbow_m=upper,
        elbow_to_wrist_m=forearm,
        sleeve_length_m=arm_scale,
        upper_arm_circumference_m=max(0.245, arm_scale * 0.53),
        elbow_circumference_m=max(0.205, arm_scale * 0.43),
        wrist_circumference_m=max(0.160, arm_scale * 0.34),
        armscye_circumference_m=max(0.315, arm_scale * 0.66),
        sleeve_cap_height_m=max(0.105, upper * 0.47),
        cap_ease_ratio=1.055,
        canonical_height_m=canonical_height,
    )
    measurements.validate()
    datums = {
        name: joints[name].astype(float).tolist()
        for name in ("L_UPPER_ARM", "L_FOREARM", "L_HAND", "R_UPPER_ARM", "R_FOREARM", "R_HAND")
    }
    return measurements, datums
