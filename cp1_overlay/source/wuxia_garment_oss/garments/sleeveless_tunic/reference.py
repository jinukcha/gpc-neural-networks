"""CP2B reference body, arbitrary-N table, and bounded CP1 fixtures."""
from __future__ import annotations

from ...sizing.body_profile.model import BodyMeasurementProfile, BodyMeasurements
from ...sizing.selection.resolver import SelectionRequest
from ...sizing.size_table.model import GarmentSizeTable, SizeEntry


def _measurements(**overrides: float) -> BodyMeasurements:
    values = {
        "stature": 1.740,
        "chest_circumference": 0.989,
        "front_chest_arc": 0.500,
        "back_chest_arc": 0.489,
        "waist_circumference": 0.856,
        "front_waist_arc": 0.428,
        "back_waist_arc": 0.428,
        "hip_circumference": 1.020,
        "shoulder_width": 0.410,
        "front_torso_length": 0.455,
        "back_torso_length": 0.435,
        "armscye_depth": 0.225,
    }
    values.update(overrides)
    return BodyMeasurements.from_dict(values)


def reference_size_table() -> GarmentSizeTable:
    entries = (
        SizeEntry("S", _measurements(
            stature=1.660, chest_circumference=0.917, front_chest_arc=0.464,
            back_chest_arc=0.453, waist_circumference=0.782,
            front_waist_arc=0.391, back_waist_arc=0.391,
            hip_circumference=0.940, shoulder_width=0.390,
            front_torso_length=0.430, back_torso_length=0.410, armscye_depth=0.210,
        )),
        SizeEntry("M", _measurements()),
        SizeEntry("L", _measurements(
            stature=1.820, chest_circumference=1.061, front_chest_arc=0.536,
            back_chest_arc=0.525, waist_circumference=0.930,
            front_waist_arc=0.465, back_waist_arc=0.465,
            hip_circumference=1.100, shoulder_width=0.430,
            front_torso_length=0.480, back_torso_length=0.460, armscye_depth=0.238,
        )),
        SizeEntry("XL", _measurements(
            stature=1.860, chest_circumference=1.133, front_chest_arc=0.572,
            back_chest_arc=0.561, waist_circumference=1.004,
            front_waist_arc=0.502, back_waist_arc=0.502,
            hip_circumference=1.180, shoulder_width=0.450,
            front_torso_length=0.495, back_torso_length=0.475, armscye_depth=0.250,
        )),
    )
    return GarmentSizeTable.from_entries("CP2B_TUNIC_STANDARD_V1", "M", entries)


def body_fixtures() -> dict[str, BodyMeasurementProfile]:
    bodies = {
        "REFERENCE": _measurements(),
        "MILD_CUSTOM": _measurements(
            chest_circumference=1.004, front_chest_arc=0.508,
            back_chest_arc=0.496, waist_circumference=0.870,
            front_waist_arc=0.435, back_waist_arc=0.435,
            shoulder_width=0.415, front_torso_length=0.462,
        ),
        "BROAD_SHOULDER": _measurements(shoulder_width=0.440),
        "FULL_CHEST": _measurements(
            chest_circumference=1.029, front_chest_arc=0.540, back_chest_arc=0.489,
        ),
        "FULL_ABDOMEN": _measurements(
            waist_circumference=0.906, front_waist_arc=0.478, back_waist_arc=0.428,
        ),
        "TALL": _measurements(
            stature=1.820, front_torso_length=0.480,
            back_torso_length=0.460, armscye_depth=0.235,
        ),
        "SHORT": _measurements(
            stature=1.660, front_torso_length=0.430,
            back_torso_length=0.410, armscye_depth=0.215,
        ),
        "NEAR_L_FORCED_M": _measurements(
            stature=1.815, chest_circumference=1.055, front_chest_arc=0.533,
            back_chest_arc=0.522, waist_circumference=0.924,
            front_waist_arc=0.462, back_waist_arc=0.462,
            hip_circumference=1.090, shoulder_width=0.428,
            front_torso_length=0.477, back_torso_length=0.457, armscye_depth=0.236,
        ),
        "COMBINED_TOPOLOGY": _measurements(
            chest_circumference=1.160, front_chest_arc=0.640, back_chest_arc=0.520,
            waist_circumference=1.050, front_waist_arc=0.580, back_waist_arc=0.470,
            shoulder_width=0.458,
        ),
        "OUT_OF_RANGE": _measurements(stature=2.350),
    }
    return {
        name: BodyMeasurementProfile(name, measurements)
        for name, measurements in bodies.items()
    }


def request_fixtures() -> tuple[tuple[str, SelectionRequest, str | None], ...]:
    return (
        ("STANDARD_M", SelectionRequest("STANDARD_M", "STANDARD_SIZE", "M"), None),
        ("AUTO_REFERENCE", SelectionRequest("AUTO_REFERENCE", "AUTO_BODY_FIT"), "REFERENCE"),
        ("AUTO_MILD_CUSTOM", SelectionRequest("AUTO_MILD_CUSTOM", "AUTO_BODY_FIT"), "MILD_CUSTOM"),
        ("AUTO_BROAD_SHOULDER", SelectionRequest("AUTO_BROAD_SHOULDER", "AUTO_BODY_FIT"), "BROAD_SHOULDER"),
        ("AUTO_FULL_CHEST", SelectionRequest("AUTO_FULL_CHEST", "AUTO_BODY_FIT"), "FULL_CHEST"),
        ("AUTO_FULL_ABDOMEN", SelectionRequest("AUTO_FULL_ABDOMEN", "AUTO_BODY_FIT"), "FULL_ABDOMEN"),
        ("AUTO_TALL", SelectionRequest("AUTO_TALL", "AUTO_BODY_FIT"), "TALL"),
        ("AUTO_SHORT", SelectionRequest("AUTO_SHORT", "AUTO_BODY_FIT"), "SHORT"),
        ("CUSTOM_FORCED_M", SelectionRequest("CUSTOM_FORCED_M", "CUSTOM_MEASUREMENTS", "M"), "NEAR_L_FORCED_M"),
        ("AUTO_TOPOLOGY", SelectionRequest("AUTO_TOPOLOGY", "AUTO_BODY_FIT"), "COMBINED_TOPOLOGY"),
        ("AUTO_HOLD", SelectionRequest("AUTO_HOLD", "AUTO_BODY_FIT"), "OUT_OF_RANGE"),
    )
