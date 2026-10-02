"""Exact full-front and full-back bodice pattern geometry."""
from __future__ import annotations

import math

from wuxia_garment_oss.pattern_geometry.curve import boundary_length, point_at_arc
from wuxia_garment_oss.pattern_geometry.inputs import GeometryInputs
from wuxia_garment_oss.pattern_geometry.model import (
    BoundaryGeometry,
    ComponentGeometryAuthority,
    CurveSegment,
    NotchPlacement,
    Point2,
)


def _notch(notch_id: str, role: str, boundary: BoundaryGeometry, segments, arc: float) -> NotchPlacement:
    point, normalized = point_at_arc(boundary, segments, arc)
    return NotchPlacement(notch_id, role, boundary.boundary_id, arc, normalized, point)


def _shoulder_x(half: float, front: bool) -> float:
    neck_half = half * 0.26
    front_drop = 0.018
    front_x = half * 0.66
    if front:
        return front_x
    front_length = math.hypot(front_x - neck_half, front_drop)
    back_drop = 0.010
    horizontal = math.sqrt(max(front_length * front_length - back_drop * back_drop, 0.0))
    return neck_half + horizontal


def _bodice_datums(inputs: GeometryInputs, front: bool) -> dict[str, Point2]:
    half = inputs.half_panel_width_m
    length = max(inputs.body_arm_length_m * 0.95, 0.56)
    underarm_y = length - inputs.armscye_depth_m
    shoulder_y = length - (0.018 if front else 0.010)
    neck_half = half * 0.26
    shoulder_x = _shoulder_x(half, front)
    neck_depth = inputs.neckline_depth_m if front else inputs.neckline_depth_m * 0.38
    return {
        "LEFT_HEM": Point2(-half, 0.0),
        "RIGHT_HEM": Point2(half, 0.0),
        "LEFT_UNDERARM": Point2(-half, underarm_y),
        "RIGHT_UNDERARM": Point2(half, underarm_y),
        "LEFT_SHOULDER": Point2(-shoulder_x, shoulder_y),
        "RIGHT_SHOULDER": Point2(shoulder_x, shoulder_y),
        "LEFT_NECK": Point2(-neck_half, length),
        "RIGHT_NECK": Point2(neck_half, length),
        "CENTER_NECK": Point2(0.0, length - neck_depth),
        "CENTER_HEM": Point2(0.0, 0.0),
        "LEFT_WAIST": Point2(-half * 0.94, length * 0.40),
        "RIGHT_WAIST": Point2(half * 0.94, length * 0.40),
    }


def _front_segments(points: dict[str, Point2]) -> tuple[CurveSegment, ...]:
    left_u, right_u = points["LEFT_UNDERARM"], points["RIGHT_UNDERARM"]
    left_s, right_s = points["LEFT_SHOULDER"], points["RIGHT_SHOULDER"]
    return (
        CurveSegment("hem_front_line", "LINE", (points["LEFT_HEM"], points["RIGHT_HEM"])),
        CurveSegment("side_left_line", "LINE", (points["LEFT_HEM"], left_u)),
        CurveSegment("side_right_line", "LINE", (points["RIGHT_HEM"], right_u)),
        CurveSegment("armhole_left_curve", "CUBIC_BEZIER", (
            left_u,
            Point2(left_u.x + 0.010, left_u.y + 0.080),
            Point2(left_s.x - 0.052, left_s.y - 0.070),
            left_s,
        )),
        CurveSegment("armhole_right_curve", "CUBIC_BEZIER", (
            right_u,
            Point2(right_u.x - 0.010, right_u.y + 0.080),
            Point2(right_s.x + 0.052, right_s.y - 0.070),
            right_s,
        )),
        CurveSegment("shoulder_left_line", "LINE", (left_s, points["LEFT_NECK"])),
        CurveSegment("shoulder_right_line", "LINE", (right_s, points["RIGHT_NECK"])),
        CurveSegment("neckline_left_curve", "CUBIC_BEZIER", (
            points["LEFT_NECK"],
            Point2(points["LEFT_NECK"].x * 0.70, points["LEFT_NECK"].y),
            Point2(-0.032, points["CENTER_NECK"].y),
            points["CENTER_NECK"],
        )),
        CurveSegment("neckline_right_curve", "CUBIC_BEZIER", (
            points["CENTER_NECK"],
            Point2(0.032, points["CENTER_NECK"].y),
            Point2(points["RIGHT_NECK"].x * 0.70, points["RIGHT_NECK"].y),
            points["RIGHT_NECK"],
        )),
        CurveSegment("center_front_fold_line", "LINE", (points["CENTER_HEM"], points["CENTER_NECK"])),
        CurveSegment("waist_reference_line", "LINE", (points["LEFT_WAIST"], points["RIGHT_WAIST"])),
    )


def _back_segments(points: dict[str, Point2]) -> tuple[CurveSegment, ...]:
    left_u, right_u = points["LEFT_UNDERARM"], points["RIGHT_UNDERARM"]
    left_s, right_s = points["LEFT_SHOULDER"], points["RIGHT_SHOULDER"]
    return (
        CurveSegment("hem_back_line", "LINE", (points["LEFT_HEM"], points["RIGHT_HEM"])),
        CurveSegment("side_left_line", "LINE", (points["LEFT_HEM"], left_u)),
        CurveSegment("side_right_line", "LINE", (points["RIGHT_HEM"], right_u)),
        CurveSegment("armhole_left_curve", "CUBIC_BEZIER", (
            left_u,
            Point2(left_u.x + 0.004, left_u.y + 0.092),
            Point2(left_s.x - 0.040, left_s.y - 0.058),
            left_s,
        )),
        CurveSegment("armhole_right_curve", "CUBIC_BEZIER", (
            right_u,
            Point2(right_u.x - 0.004, right_u.y + 0.092),
            Point2(right_s.x + 0.040, right_s.y - 0.058),
            right_s,
        )),
        CurveSegment("shoulder_left_line", "LINE", (left_s, points["LEFT_NECK"])),
        CurveSegment("shoulder_right_line", "LINE", (right_s, points["RIGHT_NECK"])),
        CurveSegment("neckline_left_curve", "CUBIC_BEZIER", (
            points["LEFT_NECK"],
            Point2(points["LEFT_NECK"].x * 0.68, points["LEFT_NECK"].y),
            Point2(-0.030, points["CENTER_NECK"].y),
            points["CENTER_NECK"],
        )),
        CurveSegment("neckline_right_curve", "CUBIC_BEZIER", (
            points["CENTER_NECK"],
            Point2(0.030, points["CENTER_NECK"].y),
            Point2(points["RIGHT_NECK"].x * 0.68, points["RIGHT_NECK"].y),
            points["RIGHT_NECK"],
        )),
        CurveSegment("center_back_fold_line", "LINE", (points["CENTER_HEM"], points["CENTER_NECK"])),
        CurveSegment("waist_reference_line", "LINE", (points["LEFT_WAIST"], points["RIGHT_WAIST"])),
    )


def _boundaries(front: bool) -> tuple[BoundaryGeometry, ...]:
    suffix = "front" if front else "back"
    return (
        BoundaryGeometry(f"hem_{suffix}", f"{suffix.upper()}_HEM", "FORWARD", "FINISHED", (f"hem_{suffix}_line",)),
        BoundaryGeometry("side_left", f"LEFT_SIDE_{suffix.upper()}", "FORWARD" if front else "REVERSE", "SEWN", ("side_left_line",)),
        BoundaryGeometry("side_right", f"RIGHT_SIDE_{suffix.upper()}", "FORWARD" if front else "REVERSE", "SEWN", ("side_right_line",)),
        BoundaryGeometry(f"armhole_left_{suffix}", f"LEFT_ARMHOLE_{suffix.upper()}", "FORWARD", "SEWN", ("armhole_left_curve",)),
        BoundaryGeometry(f"armhole_right_{suffix}", f"RIGHT_ARMHOLE_{suffix.upper()}", "FORWARD", "SEWN", ("armhole_right_curve",)),
        BoundaryGeometry("shoulder_left", f"LEFT_SHOULDER_{suffix.upper()}", "FORWARD" if front else "REVERSE", "SEWN", ("shoulder_left_line",)),
        BoundaryGeometry("shoulder_right", f"RIGHT_SHOULDER_{suffix.upper()}", "FORWARD" if front else "REVERSE", "SEWN", ("shoulder_right_line",)),
        BoundaryGeometry(f"neckline_{suffix}", f"{suffix.upper()}_NECKLINE", "FORWARD", "SEWN", ("neckline_left_curve", "neckline_right_curve")),
        BoundaryGeometry(f"center_{suffix}_fold", f"CENTER_{suffix.upper()}_FOLD", "FORWARD", "FOLD", (f"center_{suffix}_fold_line",)),
    )


def _notches(boundaries, segments, inputs: GeometryInputs, front: bool) -> tuple[NotchPlacement, ...]:
    mapping = {item.boundary_id: item for item in boundaries}
    result = []
    suffix = "front" if front else "back"
    for side in ("left", "right"):
        boundary = mapping[f"armhole_{side}_{suffix}"]
        length = boundary_length(boundary, segments)
        pitch_arc = min(inputs.front_pitch_notch_m if front else length * 0.42, length * 0.72)
        role = "FRONT_PITCH" if front else "BACK_PITCH"
        result.append(_notch(f"{side.upper()}_{role}", role, boundary, segments, pitch_arc))
        result.append(_notch(f"{side.upper()}_SHOULDER_POINT", "SHOULDER_POINT", boundary, segments, length))
    return tuple(result)


def build_bodice(inputs: GeometryInputs, front: bool) -> ComponentGeometryAuthority:
    points = _bodice_datums(inputs, front)
    segments = _front_segments(points) if front else _back_segments(points)
    boundaries = _boundaries(front)
    segment_map = {item.segment_id: item for item in segments}
    suffix = "front" if front else "back"
    return ComponentGeometryAuthority(
        instance_id=f"bodice_{suffix}",
        component_id=f"BODICE_{suffix.upper()}_BASIC",
        component_revision=2,
        parameter_set_sha256=inputs.resolved_set_sha256,
        mirror_source_instance_id=None,
        segments=segments,
        boundaries=boundaries,
        notches=_notches(boundaries, segment_map, inputs, front),
        landmarks=points,
        internal_segment_ids=(f"center_{suffix}_fold_line", "waist_reference_line"),
    )
