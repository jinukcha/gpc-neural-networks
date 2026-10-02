"""Canonical CP0 visual views, gates, and clean/rejected observations."""
from __future__ import annotations

from wuxia_garment_oss.visual_acceptance.model import VisualAcceptanceProfile, VisualGateSpec, ViewRequirement


def _view(view_id, category, body="OPTIONAL", overlays=(), families=("ALL",), resolution=2048):
    return ViewRequirement(view_id, category, resolution, resolution, body, tuple(overlays), tuple(families))


def required_views() -> tuple[ViewRequirement, ...]:
    primary = tuple(_view(name, "PRIMARY", "VISIBLE") for name in (
        "front", "left", "right", "back", "front_three_quarter", "back_three_quarter"
    ))
    details = (
        _view("neck_facing_detail", "DETAIL", "VISIBLE"),
        _view("shoulder_sleeve_cap_detail", "DETAIL", "VISIBLE"),
        _view("underarm_detail", "DETAIL", "VISIBLE"),
        _view("elbow_cuff_detail", "DETAIL", "VISIBLE"),
        _view("waist_side_detail", "DETAIL", "VISIBLE"),
        _view("robe_opening_gore_hem_detail", "DETAIL", "VISIBLE", families=("STRAIGHT_SLEEVE_ROBE",)),
    )
    diagnostics = (
        _view("body_visible_fit", "DIAGNOSTIC", "VISIBLE"),
        _view("body_occlusion_runtime", "DIAGNOSTIC", "OCCLUDED"),
        _view("wireframe", "DIAGNOSTIC", "VISIBLE", ("WIREFRAME",)),
        _view("seam_notch_overlay", "DIAGNOSTIC", "VISIBLE", ("SEAM", "NOTCH")),
        _view("component_ownership", "DIAGNOSTIC", "VISIBLE", ("COMPONENT_ID",)),
        _view("layer_cutaway", "DIAGNOSTIC", "VISIBLE", ("LAYER",)),
        _view("normal_diagnostic", "DIAGNOSTIC", "VISIBLE", ("NORMAL",)),
    )
    motion = tuple(_view(name, "MOTION", "VISIBLE", resolution=1600) for name in (
        "motion_arms_forward", "motion_arms_overhead", "motion_cross_body_reach",
        "motion_deep_elbow_bend", "motion_torso_twist", "motion_walk_stride", "motion_seated"
    ))
    actual_distance = tuple(_view(name, "LOD_ACTUAL_DISTANCE", "VISIBLE", resolution=1600) for name in (
        "actual_distance_near", "actual_distance_mid", "actual_distance_far"
    ))
    equal_coverage = tuple(_view(name, "LOD_EQUAL_COVERAGE", "VISIBLE", resolution=1600) for name in (
        "equal_coverage_lod0", "equal_coverage_lod1", "equal_coverage_lod2"
    ))
    return primary + details + diagnostics + motion + actual_distance + equal_coverage


def _gate(gate_id, severity, comparator, threshold, scope, disposition, categories):
    return VisualGateSpec(gate_id, severity, gate_id, comparator, threshold, scope, disposition, tuple(categories))


def visual_gates() -> tuple[VisualGateSpec, ...]:
    zero = (
        _gate("floating_component_count", "ZERO_TOLERANCE", "EQ", 0, "GARMENT", "HOLD", ("PRIMARY", "DIAGNOSTIC")),
        _gate("unowned_loose_triangle_count", "ZERO_TOLERANCE", "EQ", 0, "TOPOLOGY", "HOLD", ("WIREFRAME", "COMPONENT_ID")),
        _gate("non_design_open_boundary_count", "ZERO_TOLERANCE", "EQ", 0, "BOUNDARY", "HOLD", ("WIREFRAME", "SEAM")),
        _gate("neck_facing_separation_count", "ZERO_TOLERANCE", "EQ", 0, "NECK", "GUIDED", ("DETAIL", "LAYER")),
        _gate("sleeve_cap_armhole_gap_count", "ZERO_TOLERANCE", "EQ", 0, "SHOULDER", "GUIDED", ("DETAIL", "SEAM")),
        _gate("underarm_seam_gap_count", "ZERO_TOLERANCE", "EQ", 0, "UNDERARM", "GUIDED", ("DETAIL", "SEAM")),
        _gate("normal_inversion_count", "ZERO_TOLERANCE", "EQ", 0, "SURFACE", "GUIDED", ("NORMAL",)),
        _gate("layer_order_reversal_count", "ZERO_TOLERANCE", "EQ", 0, "LAYER", "HOLD", ("LAYER",)),
        _gate("closure_separation_count", "ZERO_TOLERANCE", "EQ", 0, "CLOSURE", "GUIDED", ("DETAIL",)),
        _gate("non_finite_geometry_count", "ZERO_TOLERANCE", "EQ", 0, "GEOMETRY", "HOLD", ("DIAGNOSTIC",)),
    )
    bounded = (
        _gate("body_penetration_p99_m", "BOUNDED", "LE", 0.002, "FIT", "GUIDED", ("BODY_VISIBLE", "MOTION")),
        _gate("undesigned_silhouette_asymmetry_ratio", "BOUNDED", "LE", 0.01, "SILHOUETTE", "GUIDED", ("PRIMARY",)),
        _gate("lod1_silhouette_iou", "BOUNDED", "GE", 0.985, "LOD1", "HOLD", ("LOD_EQUAL_COVERAGE",)),
        _gate("lod2_silhouette_iou", "BOUNDED", "GE", 0.960, "LOD2", "HOLD", ("LOD_EQUAL_COVERAGE",)),
        _gate("lod_feature_boundary_displacement_px", "BOUNDED", "LE", 1.5, "LOD", "HOLD", ("LOD_EQUAL_COVERAGE",)),
        _gate("required_motion_view_pass", "BOUNDED", "TRUE", True, "MOTION", "HOLD", ("MOTION",)),
    )
    return zero + bounded


def visual_profile() -> VisualAcceptanceProfile:
    return VisualAcceptanceProfile(
        "R1C_VISUAL_PROFILE_V1",
        1,
        required_views(),
        visual_gates(),
        "TECHNICAL_PASS_AND_VISUAL_PASS",
    )


def clean_observations() -> dict[str, float | bool]:
    values: dict[str, float | bool] = {}
    for gate in visual_gates():
        if gate.comparator == "EQ":
            values[gate.gate_id] = gate.threshold
        elif gate.comparator == "LE":
            values[gate.gate_id] = 0.0 if float(gate.threshold) <= 0.01 else float(gate.threshold) * 0.5
        elif gate.comparator == "GE":
            values[gate.gate_id] = min(1.0, float(gate.threshold) + 0.01)
        elif gate.comparator == "TRUE":
            values[gate.gate_id] = True
    return values


def rejected_observations() -> dict[str, float | bool]:
    values = clean_observations()
    values.update({
        "floating_component_count": 1,
        "sleeve_cap_armhole_gap_count": 2,
        "lod2_silhouette_iou": 0.91,
        "required_motion_view_pass": False,
    })
    return values
