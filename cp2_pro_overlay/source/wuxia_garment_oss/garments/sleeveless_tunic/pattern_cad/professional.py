"""Professional grading, notch, and feature pilots for the tunic reference."""
from __future__ import annotations

from ....pattern_cad.document.model import PatternDocument
from ....pattern_cad.features.model import PatternFeatureGraph, PatternFeatureSpec
from ....pattern_cad.grading.model import (
    GradePointRule,
    GradeRuleSetV2,
    PatternNotch,
    PatternNotchSet,
    SizeGrade,
)


SIZE_IDS = ("XXS", "XS", "S", "M", "L", "XL", "XXL")


def _input_deltas(step: int) -> dict[str, float]:
    factors = {
        "body_stature": 0.040,
        "body_front_chest_arc": 0.018,
        "body_back_chest_arc": 0.018,
        "body_front_waist_arc": 0.0185,
        "body_back_waist_arc": 0.0185,
        "body_hip_circumference": 0.040,
        "body_shoulder_width": 0.010,
        "body_front_torso_length": 0.0125,
        "body_back_torso_length": 0.0125,
        "body_armscye_depth": 0.006,
    }
    return {name: value * step for name, value in factors.items()}


def _grade_points() -> tuple[GradePointRule, ...]:
    rows = []
    roles = {
        "shoulder_left": "SHOULDER_END_LEFT",
        "shoulder_right": "SHOULDER_END_RIGHT",
        "underarm_left": "CHEST_SIDE_LEFT",
        "underarm_right": "CHEST_SIDE_RIGHT",
        "waist_left": "WAIST_SIDE_LEFT",
        "waist_right": "WAIST_SIDE_RIGHT",
    }
    for panel in ("bodice_front", "bodice_back"):
        for suffix, role in roles.items():
            rows.append(GradePointRule(f"GP.{panel}.{suffix}", f"{panel}.{suffix}", role))
    for panel in ("skirt_front", "skirt_back"):
        for suffix, role in (
            ("waist_left", "SKIRT_WAIST_LEFT"),
            ("waist_right", "SKIRT_WAIST_RIGHT"),
            ("hip_left", "HIP_LEFT"),
            ("hip_right", "HIP_RIGHT"),
            ("hem_left", "HEM_LEFT"),
            ("hem_right", "HEM_RIGHT"),
        ):
            rows.append(GradePointRule(f"GP.{panel}.{suffix}", f"{panel}.{suffix}", role))
    return tuple(rows)


def tunic_grade_rule_set(document: PatternDocument) -> GradeRuleSetV2:
    sizes = tuple(
        SizeGrade(size_id, ordinal, _input_deltas(ordinal))
        for size_id, ordinal in zip(SIZE_IDS, range(-3, 4))
    )
    result = GradeRuleSetV2(
        "CP2B_TUNIC_INDUSTRIAL_GRADE_RULES_V2",
        "M",
        sizes,
        _grade_points(),
    )
    result.validate(document)
    return result


def tunic_notch_set(document: PatternDocument) -> PatternNotchSet:
    definitions = (
        ("N_SHOULDER_L_FRONT", "bodice_front.shoulder_left", 0.55, "SHOULDER_MATCH"),
        ("N_SHOULDER_L_BACK", "bodice_back.shoulder_left", 0.55, "SHOULDER_MATCH"),
        ("N_SHOULDER_R_FRONT", "bodice_front.shoulder_right", 0.45, "SHOULDER_MATCH"),
        ("N_SHOULDER_R_BACK", "bodice_back.shoulder_right", 0.45, "SHOULDER_MATCH"),
        ("N_SIDE_L_FRONT", "bodice_front.side_left", 0.40, "SIDE_MATCH"),
        ("N_SIDE_L_BACK", "bodice_back.side_left", 0.40, "SIDE_MATCH"),
        ("N_SIDE_R_FRONT", "bodice_front.side_right", 0.60, "SIDE_MATCH"),
        ("N_SIDE_R_BACK", "bodice_back.side_right", 0.60, "SIDE_MATCH"),
        ("N_WAIST_FRONT_BODICE", "bodice_front.waist", 0.50, "WAIST_CENTER"),
        ("N_WAIST_FRONT_SKIRT", "skirt_front.waist", 0.50, "WAIST_CENTER"),
        ("N_WAIST_BACK_BODICE", "bodice_back.waist", 0.50, "WAIST_CENTER"),
        ("N_WAIST_BACK_SKIRT", "skirt_back.waist", 0.50, "WAIST_CENTER"),
        ("N_HEM_FRONT_CENTER", "skirt_front.hem", 0.50, "HEM_REFERENCE"),
        ("N_HEM_BACK_CENTER", "skirt_back.hem", 0.50, "HEM_REFERENCE"),
    )
    result = PatternNotchSet(
        "CP2B_TUNIC_NOTCH_SET_V1",
        tuple(PatternNotch(*row) for row in definitions),
    )
    result.validate(document)
    return result


def tunic_feature_graph() -> PatternFeatureGraph:
    features = (
        PatternFeatureSpec(
            "DART_FRONT_WAIST", "DART", "bodice_front",
            {"center_x": 0.0, "waist_y": 0.0, "apex_x": 0.0, "apex_y": 0.18, "intake_m": 0.024},
        ),
        PatternFeatureSpec(
            "PLEAT_FRONT_CENTER", "PLEAT", "skirt_front",
            {"center_x": 0.0, "depth_m": 0.018, "bottom_margin_m": 0.04},
            ("DART_FRONT_WAIST",),
        ),
        PatternFeatureSpec(
            "GATHER_BACK_WAIST", "GATHER", "skirt_back.waist",
            {"ratio": 1.12, "start_fraction": 0.08, "end_fraction": 0.92},
        ),
        PatternFeatureSpec(
            "GUSSET_UNDERARM", "GUSSET", "gusset_underarm",
            {
                "width_m": 0.09,
                "height_m": 0.12,
                "center_x": 0.0,
                "center_y": 0.68,
                "connections": [
                    "bodice_front.side_left",
                    "bodice_back.side_left",
                    "bodice_front.side_right",
                    "bodice_back.side_right",
                ],
            },
            ("DART_FRONT_WAIST",),
        ),
    )
    graph = PatternFeatureGraph("CP2B_TUNIC_FEATURE_GRAPH_V1", features)
    graph.validate()
    return graph


def invalid_feature_specs() -> tuple[PatternFeatureSpec, ...]:
    return (
        PatternFeatureSpec(
            "BAD_DART_OWNER", "DART", "missing_panel",
            {"center_x": 0.0, "apex_y": 0.18, "intake_m": 0.024},
        ),
        PatternFeatureSpec(
            "BAD_PLEAT_DEPTH", "PLEAT", "skirt_front",
            {"center_x": 0.0, "depth_m": 0.0, "bottom_margin_m": 0.04},
        ),
        PatternFeatureSpec(
            "BAD_GATHER_RATIO", "GATHER", "skirt_back.waist",
            {"ratio": 0.80, "start_fraction": 0.0, "end_fraction": 1.0},
        ),
        PatternFeatureSpec(
            "BAD_GUSSET_DUPLICATE", "GUSSET", "bodice_front",
            {"width_m": 0.09, "height_m": 0.12, "center_x": 0.0, "center_y": 0.68},
        ),
    )
