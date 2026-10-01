"""Separate standard grade movement from body-specific alterations."""
from __future__ import annotations

from dataclasses import asdict

from ..body_profile.model import BodyMeasurements, MEASUREMENT_FIELDS
from .blocks import BlockSelection


CUSTOM_LIMITS = {
    "stature": 0.050,
    "chest_circumference": 0.040,
    "front_chest_arc": 0.025,
    "back_chest_arc": 0.025,
    "waist_circumference": 0.045,
    "front_waist_arc": 0.030,
    "back_waist_arc": 0.030,
    "hip_circumference": 0.050,
    "shoulder_width": 0.015,
    "front_torso_length": 0.020,
    "back_torso_length": 0.020,
    "armscye_depth": 0.012,
}
TOPOLOGY_LIMITS = {
    name: value * 2.4 for name, value in CUSTOM_LIMITS.items()
}
HARD_RANGES = {
    "stature": (1.30, 2.20),
    "chest_circumference": (0.65, 1.60),
    "waist_circumference": (0.55, 1.55),
    "hip_circumference": (0.70, 1.70),
    "shoulder_width": (0.30, 0.62),
    "armscye_depth": (0.15, 0.36),
}


def difference_plan(
    source: BodyMeasurements,
    target: BodyMeasurements,
    operation: str,
) -> list[dict]:
    rows = []
    for name in MEASUREMENT_FIELDS:
        delta = float(getattr(target, name) - getattr(source, name))
        if abs(delta) < 1.0e-10:
            continue
        rows.append({"operation": operation, "measurement": name, "delta_m": delta})
    return rows


def residuals(body: BodyMeasurements, block_target: BodyMeasurements) -> dict[str, float]:
    return {
        name: float(getattr(body, name) - getattr(block_target, name))
        for name in MEASUREMENT_FIELDS
    }


def hard_range_violations(body: BodyMeasurements) -> list[str]:
    values = asdict(body)
    return [
        name
        for name, (minimum, maximum) in HARD_RANGES.items()
        if not (minimum <= float(values[name]) <= maximum)
    ]


def classify_admission(
    mode: str,
    body: BodyMeasurements | None,
    selection: BlockSelection,
    residual_by_name: dict[str, float],
    selected_size_id: str,
    recommended_size_id: str,
    selected_score: float,
    recommended_score: float,
) -> tuple[str, list[str]]:
    if mode == "STANDARD_SIZE":
        return "NORMAL_GRADE", []
    assert body is not None
    hard = hard_range_violations(body)
    if hard:
        return "HOLD", [f"out_of_supported_range:{name}" for name in hard]
    topology = [
        name for name, value in residual_by_name.items()
        if abs(value) > TOPOLOGY_LIMITS[name]
    ]
    if len(selection.shape_flags) > 1:
        topology.append("multiple_shape_blocks")
    if topology:
        return "TOPOLOGY_CHANGE_REQUIRED", sorted(set(topology))
    excessive = [
        name for name, value in residual_by_name.items()
        if abs(value) > CUSTOM_LIMITS[name]
    ]
    score_advantage = selected_score - recommended_score
    if selected_size_id != recommended_size_id and score_advantage > 0.35:
        return "ALTERNATE_BLOCK_REQUIRED", [f"recommended_size:{recommended_size_id}"]
    if excessive:
        return "ALTERNATE_BLOCK_REQUIRED", [f"excessive_alteration:{name}" for name in excessive]
    changed = any(abs(value) > 0.001 for value in residual_by_name.values())
    block_changed = selection.shape_block != "REGULAR" or selection.height_block != "REGULAR"
    return ("CUSTOM_ALTERATION", []) if changed or block_changed else ("NORMAL_GRADE", [])
