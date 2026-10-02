"""Exact collar, cuff, and side-gore component geometry."""
from __future__ import annotations

from wuxia_garment_oss.pattern_geometry.curve import mirror_segment, point_at_arc
from wuxia_garment_oss.pattern_geometry.inputs import GeometryInputs
from wuxia_garment_oss.pattern_geometry.model import (
    BoundaryGeometry,
    ComponentGeometryAuthority,
    CurveSegment,
    NotchPlacement,
    Point2,
)


def build_collar(inputs: GeometryInputs, front_neckline: float, back_neckline: float) -> ComponentGeometryAuthority:
    depth = inputs.collar_depth_m + inputs.turn_of_cloth_m
    split = front_neckline
    total = front_neckline + back_neckline
    points = {
        "FRONT_END_LOWER": Point2(0.0, 0.0),
        "FRONT_BACK_JUNCTION_LOWER": Point2(split, 0.0),
        "BACK_END_LOWER": Point2(total, 0.0),
        "FRONT_END_UPPER": Point2(0.0, depth),
        "FRONT_BACK_JUNCTION_UPPER": Point2(split, depth * 1.04),
        "BACK_END_UPPER": Point2(total, depth * 0.92),
    }
    segments = (
        CurveSegment("front_attach_line", "LINE", (points["FRONT_END_LOWER"], points["FRONT_BACK_JUNCTION_LOWER"])),
        CurveSegment("back_attach_line", "LINE", (points["FRONT_BACK_JUNCTION_LOWER"], points["BACK_END_LOWER"])),
        CurveSegment("back_end_line", "LINE", (points["BACK_END_LOWER"], points["BACK_END_UPPER"])),
        CurveSegment("outer_collar_curve", "CUBIC_BEZIER", (
            points["BACK_END_UPPER"],
            Point2(total * 0.67, depth * 1.12),
            Point2(total * 0.33, depth * 1.12),
            points["FRONT_END_UPPER"],
        )),
        CurveSegment("front_end_line", "LINE", (points["FRONT_END_UPPER"], points["FRONT_END_LOWER"])),
        CurveSegment("roll_line", "LINE", (points["FRONT_END_UPPER"], points["BACK_END_UPPER"])),
    )
    boundaries = (
        BoundaryGeometry("front_attach", "COLLAR_FRONT_NECK_ATTACH", "REVERSE", "SEWN", ("front_attach_line",)),
        BoundaryGeometry("back_attach", "COLLAR_BACK_NECK_ATTACH", "REVERSE", "SEWN", ("back_attach_line",)),
        BoundaryGeometry("outer", "COLLAR_OUTER_EDGE", "FORWARD", "FINISHED", ("outer_collar_curve",)),
        BoundaryGeometry("front_end", "COLLAR_FRONT_END", "FORWARD", "FINISHED", ("front_end_line",)),
        BoundaryGeometry("back_end", "COLLAR_BACK_END", "FORWARD", "FINISHED", ("back_end_line",)),
    )
    segment_map = {item.segment_id: item for item in segments}
    front_boundary = boundaries[0]
    back_boundary = boundaries[1]
    front_point, front_norm = point_at_arc(front_boundary, segment_map, front_neckline)
    back_point, back_norm = point_at_arc(back_boundary, segment_map, 0.0)
    notches = (
        NotchPlacement("COLLAR_FRONT_BACK_JUNCTION_FRONT", "FRONT_BACK_JUNCTION", "front_attach", front_neckline, front_norm, front_point),
        NotchPlacement("COLLAR_FRONT_BACK_JUNCTION_BACK", "FRONT_BACK_JUNCTION", "back_attach", 0.0, back_norm, back_point),
    )
    return ComponentGeometryAuthority(
        instance_id="collar",
        component_id="COLLAR_STAND_BASIC",
        component_revision=1,
        parameter_set_sha256=inputs.resolved_set_sha256,
        mirror_source_instance_id=None,
        segments=segments,
        boundaries=boundaries,
        notches=notches,
        landmarks=points,
        internal_segment_ids=("roll_line",),
    )


def build_cuff(inputs: GeometryInputs, instance_id: str, wrist_length: float, mirror_source: str | None) -> ComponentGeometryAuthority:
    depth = max(0.060, inputs.collar_depth_m * 2.8)
    points = {
        "ATTACH_LEFT": Point2(0.0, 0.0),
        "ATTACH_RIGHT": Point2(wrist_length, 0.0),
        "OUTER_LEFT": Point2(0.0, depth),
        "OUTER_RIGHT": Point2(wrist_length, depth),
    }
    segments = (
        CurveSegment("sleeve_attach_line", "LINE", (points["ATTACH_LEFT"], points["ATTACH_RIGHT"])),
        CurveSegment("right_end_line", "LINE", (points["ATTACH_RIGHT"], points["OUTER_RIGHT"])),
        CurveSegment("outer_line", "LINE", (points["OUTER_RIGHT"], points["OUTER_LEFT"])),
        CurveSegment("left_end_line", "LINE", (points["OUTER_LEFT"], points["ATTACH_LEFT"])),
        CurveSegment("fold_line", "LINE", (Point2(0.0, depth * 0.5), Point2(wrist_length, depth * 0.5))),
    )
    boundaries = (
        BoundaryGeometry("sleeve_attach", "CUFF_SLEEVE_ATTACH", "REVERSE", "SEWN", ("sleeve_attach_line",)),
        BoundaryGeometry("end_left", "CUFF_END_LEFT", "FORWARD", "SEWN", ("left_end_line",)),
        BoundaryGeometry("end_right", "CUFF_END_RIGHT", "REVERSE", "SEWN", ("right_end_line",)),
        BoundaryGeometry("outer", "CUFF_OUTER_EDGE", "FORWARD", "FINISHED", ("outer_line",)),
    )
    return ComponentGeometryAuthority(
        instance_id=instance_id,
        component_id="CUFF_STRAIGHT_BASIC",
        component_revision=1,
        parameter_set_sha256=inputs.resolved_set_sha256,
        mirror_source_instance_id=mirror_source,
        segments=segments,
        boundaries=boundaries,
        notches=(),
        landmarks=points,
        internal_segment_ids=("fold_line",),
    )


def build_gore(inputs: GeometryInputs, instance_id: str, mirror_source: str | None) -> ComponentGeometryAuthority:
    half_width = max(0.085, inputs.half_panel_width_m * 0.28)
    height = max(0.50, inputs.body_arm_length_m * 0.88)
    points = {
        "APEX": Point2(0.0, height),
        "FRONT_HEM": Point2(-half_width, 0.0),
        "BACK_HEM": Point2(half_width, 0.0),
    }
    segments = (
        CurveSegment("front_attach_line", "LINE", (points["FRONT_HEM"], points["APEX"])),
        CurveSegment("back_attach_line", "LINE", (points["APEX"], points["BACK_HEM"])),
        CurveSegment("hem_line", "LINE", (points["BACK_HEM"], points["FRONT_HEM"])),
        CurveSegment("grainline", "LINE", (Point2(0.0, 0.04), Point2(0.0, height - 0.03))),
    )
    if mirror_source:
        segments = tuple(mirror_segment(item, item.segment_id) for item in segments)
        points = {key: Point2(-value.x, value.y) for key, value in points.items()}
    boundaries = (
        BoundaryGeometry("front_attach", "GORE_FRONT_ATTACH", "FORWARD", "SEWN", ("front_attach_line",)),
        BoundaryGeometry("back_attach", "GORE_BACK_ATTACH", "REVERSE", "SEWN", ("back_attach_line",)),
        BoundaryGeometry("hem", "GORE_HEM", "FORWARD", "FINISHED", ("hem_line",)),
    )
    return ComponentGeometryAuthority(
        instance_id=instance_id,
        component_id="SIDE_GORE_BASIC",
        component_revision=1,
        parameter_set_sha256=inputs.resolved_set_sha256,
        mirror_source_instance_id=mirror_source,
        segments=segments,
        boundaries=boundaries,
        notches=(),
        landmarks=points,
        internal_segment_ids=("grainline",),
    )
