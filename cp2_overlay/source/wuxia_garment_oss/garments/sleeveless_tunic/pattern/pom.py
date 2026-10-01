"""Resolve finished tunic POM values from body measurements and design policy."""
from __future__ import annotations

from ....sizing.body_profile.model import BodyMeasurements
from .spec import TunicDesignSpec


def resolve_poms(
    body: BodyMeasurements,
    spec: TunicDesignSpec | None = None,
) -> dict[str, float]:
    spec = spec or TunicDesignSpec()
    body.validate()
    front_chest_arc = body.front_chest_arc + spec.front_chest_ease_m
    back_chest_arc = body.back_chest_arc + spec.back_chest_ease_m
    front_waist_arc = body.front_waist_arc + spec.front_waist_ease_m
    back_waist_arc = body.back_waist_arc + spec.back_waist_ease_m
    finished_hip = body.hip_circumference + spec.hip_ease_m
    shoulder_half = body.shoulder_width * 0.5
    neck_half = spec.reference_neck_half_m + spec.neck_shoulder_factor * (
        body.shoulder_width - spec.reference_shoulder_width_m
    )
    neck_half = spec.clamp_neck_half(neck_half)
    front_shoulder_height = body.front_torso_length
    back_shoulder_height = body.back_torso_length + spec.back_balance_m
    front_top_height = front_shoulder_height + spec.shoulder_drop_m
    back_top_height = back_shoulder_height + spec.shoulder_drop_m
    underarm_height = body.armscye_depth + spec.armscye_mobility_m
    front_neck_depth = spec.reference_front_neck_depth_m + spec.front_neck_length_factor * (
        body.front_torso_length - spec.reference_front_torso_m
    )
    back_neck_depth = spec.reference_back_neck_depth_m + spec.back_neck_length_factor * (
        body.back_torso_length - spec.reference_back_torso_m
    )
    skirt_length = spec.reference_skirt_length_m + spec.skirt_stature_factor * (
        body.stature - spec.reference_stature_m
    )
    hip_drop = spec.reference_hip_drop_m + spec.hip_drop_stature_factor * (
        body.stature - spec.reference_stature_m
    )
    hip_half = finished_hip * 0.25
    hem_half = max(front_waist_arc * 0.5, back_waist_arc * 0.5, hip_half)
    hem_half += spec.hem_flare_m
    return {
        "finished_chest_circumference_m": front_chest_arc + back_chest_arc,
        "finished_front_chest_arc_m": front_chest_arc,
        "finished_back_chest_arc_m": back_chest_arc,
        "finished_waist_circumference_m": front_waist_arc + back_waist_arc,
        "finished_front_waist_arc_m": front_waist_arc,
        "finished_back_waist_arc_m": back_waist_arc,
        "finished_hip_circumference_m": finished_hip,
        "front_chest_half_m": front_chest_arc * 0.5,
        "back_chest_half_m": back_chest_arc * 0.5,
        "front_waist_half_m": front_waist_arc * 0.5,
        "back_waist_half_m": back_waist_arc * 0.5,
        "shoulder_half_m": shoulder_half,
        "neck_half_m": neck_half,
        "front_shoulder_height_m": front_shoulder_height,
        "back_shoulder_height_m": back_shoulder_height,
        "front_top_height_m": front_top_height,
        "back_top_height_m": back_top_height,
        "underarm_height_m": underarm_height,
        "front_neck_depth_m": front_neck_depth,
        "back_neck_depth_m": back_neck_depth,
        "skirt_length_m": skirt_length,
        "hip_drop_m": hip_drop,
        "hip_half_m": hip_half,
        "hem_half_m": hem_half,
    }
