"""Explicit V1-to-V2 migration with a loss report; no silent direct-measurement claims."""
from __future__ import annotations

from typing import Mapping

from .model import BodyMeasurementProfileV2, MeasurementRecord, REQUIRED_TORSO


def _record(name: str, value: float | None, method: str, confidence: float) -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=name,
        value_m=value,
        method=method,
        confidence=confidence,
        tolerance_m=0.004 if value is not None else 0.0,
        source_revision="V1_TO_V2_RECOVERY_MIGRATION",
    )


def _derived_records(values: Mapping[str, float]) -> dict[str, MeasurementRecord]:
    stature = float(values["stature"])
    chest = float(values["chest_circumference"])
    waist = float(values["waist_circumference"])
    hip = float(values["hip_circumference"])
    shoulder = float(values["shoulder_width"])
    derived = {
        "left_arm_length": 0.335 * stature,
        "right_arm_length": 0.335 * stature,
        "left_upper_arm_circumference": 0.31 * chest,
        "right_upper_arm_circumference": 0.31 * chest,
        "left_elbow_circumference": 0.25 * chest,
        "right_elbow_circumference": 0.25 * chest,
        "left_wrist_circumference": 0.17 * chest,
        "right_wrist_circumference": 0.17 * chest,
        "crotch_depth": 0.145 * stature,
        "front_crotch_length": 0.31 * waist,
        "back_crotch_length": 0.37 * hip,
        "left_inseam": 0.455 * stature,
        "right_inseam": 0.455 * stature,
        "left_thigh_circumference": 0.58 * hip,
        "right_thigh_circumference": 0.58 * hip,
        "left_knee_circumference": 0.38 * hip,
        "right_knee_circumference": 0.38 * hip,
        "neck_base_circumference": 0.88 * shoulder,
    }
    return {name: _record(name, value, "FORMULA_DERIVED", 0.68) for name, value in derived.items()}


def _missing_records() -> dict[str, MeasurementRecord]:
    names = (
        "forward_head_offset",
        "left_shoulder_pitch",
        "right_shoulder_pitch",
        "thoracic_curve",
        "lumbar_curve",
        "pelvic_tilt",
        "left_shoulder_height",
        "right_shoulder_height",
        "left_hip_height",
        "right_hip_height",
    )
    return {name: _record(name, None, "MISSING", 0.0) for name in names}


def migrate_v1_profile(payload: Mapping[str, object]) -> tuple[BodyMeasurementProfileV2, dict]:
    measurements = payload.get("measurements", payload)
    if not isinstance(measurements, Mapping):
        raise ValueError("V1 measurements object is required")
    values = {name: float(measurements[name]) for name in REQUIRED_TORSO}
    records = {name: _record(name, value, "MIGRATED_V1", 0.95) for name, value in values.items()}
    records.update(_derived_records(values))
    records.update(_missing_records())
    profile = BodyMeasurementProfileV2(
        profile_id=str(payload.get("body_id", payload.get("profile_id", "REFERENCE_MIGRATED_V1"))),
        records=records,
        source_mesh_sha256=str(payload.get("source_mesh_sha256", "CP3_REFERENCE_BODY_UNAVAILABLE")),
        provenance={
            "migration": "BodyMeasurementProfile/1 -> /2",
            "byte_exact_cp0_predecessor": False,
            "reconstruction_basis": "CP3 sizing authority",
        },
    )
    profile.validate()
    report = {
        "contract": "AnthropometryMigrationReport/1",
        "source_contract": "BodyMeasurementProfile/1",
        "target_contract": "BodyMeasurementProfile/2",
        "mapped_fields": sorted(REQUIRED_TORSO),
        "formula_derived_fields": sorted(_derived_records(values)),
        "missing_fields": sorted(_missing_records()),
        "silent_defaults": [],
        "production_holds": [
            "direct_arm_measurements_required_for_fitted_sleeve_production",
            "direct_lower_body_measurements_required_for_trousers_production",
            "posture_and_asymmetry_measurements_required_for_asymmetric_fit",
        ],
    }
    return profile, report
