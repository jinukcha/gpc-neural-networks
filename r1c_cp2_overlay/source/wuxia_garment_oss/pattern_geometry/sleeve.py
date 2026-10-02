"""True asymmetric one-piece set-in sleeve pattern geometry."""
from __future__ import annotations

from wuxia_garment_oss.pattern_geometry.curve import boundary_length, mirror_segment, point_at_arc, segment_length
from wuxia_garment_oss.pattern_geometry.inputs import GeometryInputs
from wuxia_garment_oss.pattern_geometry.model import (
    BoundaryGeometry,
    ComponentGeometryAuthority,
    CurveSegment,
    NotchPlacement,
    Point2,
)


def _cap_segment(segment_id: str, half_width: float, cap_height: float, length: float, front: bool) -> CurveSegment:
    sign = -1.0 if front else 1.0
    apex = Point2(0.0, length)
    underarm = Point2(sign * half_width, length - cap_height)
    first_scale = 0.34 if front else 0.30
    second_scale = 0.94 if front else 0.97
    second_height = 0.45 if front else 0.58
    return CurveSegment(
        segment_id,
        "CUBIC_BEZIER",
        (
            apex,
            Point2(sign * half_width * first_scale, length - cap_height * 0.03),
            Point2(sign * half_width * second_scale, underarm.y + cap_height * second_height),
            underarm,
        ),
    )


def _solve_half_width(target_length: float, cap_height: float, sleeve_length: float, front: bool) -> float:
    lower, upper = 0.05, 0.48
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        length = segment_length(_cap_segment("probe", middle, cap_height, sleeve_length, front))
        if length < target_length:
            lower = middle
        else:
            upper = middle
    result = 0.5 * (lower + upper)
    achieved = segment_length(_cap_segment("probe", result, cap_height, sleeve_length, front))
    if abs(achieved - target_length) > 1.0e-8:
        raise ValueError("sleeve-cap root solve did not converge")
    return result


def _build_segments(inputs: GeometryInputs, front_target: float, back_target: float):
    front_half = _solve_half_width(front_target, inputs.cap_height_m, inputs.sleeve_length_m, True)
    back_half = _solve_half_width(back_target, inputs.cap_height_m, inputs.sleeve_length_m, False)
    front_cap = _cap_segment("cap_front_curve", front_half, inputs.cap_height_m, inputs.sleeve_length_m, True)
    back_cap = _cap_segment("cap_back_curve", back_half, inputs.cap_height_m, inputs.sleeve_length_m, False)
    front_underarm, back_underarm = front_cap.points[-1], back_cap.points[-1]
    front_wrist = Point2(-max(0.078, front_half * 0.54), 0.0)
    back_wrist = Point2(max(0.082, back_half * 0.56), 0.0)
    elbow_y = inputs.sleeve_length_m * 0.47
    segments = (
        front_cap,
        back_cap,
        CurveSegment("underarm_front_line", "LINE", (front_underarm, front_wrist)),
        CurveSegment("underarm_back_line", "LINE", (back_wrist, back_underarm)),
        CurveSegment("wrist_line", "LINE", (front_wrist, back_wrist)),
        CurveSegment("biceps_line", "LINE", (front_underarm, back_underarm)),
        CurveSegment("elbow_line", "LINE", (
            Point2(front_wrist.x * 0.58 + front_underarm.x * 0.42, elbow_y),
            Point2(back_wrist.x * 0.58 + back_underarm.x * 0.42, elbow_y),
        )),
        CurveSegment("grainline", "LINE", (Point2(0.0, 0.04), Point2(0.0, inputs.sleeve_length_m - 0.02))),
    )
    landmarks = {
        "SHOULDER_POINT": Point2(0.0, inputs.sleeve_length_m),
        "FRONT_UNDERARM": front_underarm,
        "BACK_UNDERARM": back_underarm,
        "FRONT_WRIST": front_wrist,
        "BACK_WRIST": back_wrist,
        "FRONT_ELBOW": segments[6].points[0],
        "BACK_ELBOW": segments[6].points[1],
    }
    return segments, landmarks


def _boundaries() -> tuple[BoundaryGeometry, ...]:
    return (
        BoundaryGeometry("cap_front", "SLEEVE_CAP_FRONT", "REVERSE", "SEWN", ("cap_front_curve",)),
        BoundaryGeometry("cap_back", "SLEEVE_CAP_BACK", "REVERSE", "SEWN", ("cap_back_curve",)),
        BoundaryGeometry("underarm_front", "SLEEVE_UNDERARM_FRONT", "FORWARD", "SEWN", ("underarm_front_line",)),
        BoundaryGeometry("underarm_back", "SLEEVE_UNDERARM_BACK", "REVERSE", "SEWN", ("underarm_back_line",)),
        BoundaryGeometry("wrist", "SLEEVE_WRIST", "FORWARD", "SEWN", ("wrist_line",)),
    )


def _notch(notch_id: str, role: str, boundary, segments, normalized: float) -> NotchPlacement:
    length = boundary_length(boundary, segments)
    arc = length * normalized
    point, actual = point_at_arc(boundary, segments, arc)
    return NotchPlacement(notch_id, role, boundary.boundary_id, arc, actual, point)


def _notches(boundaries, segments, front_armhole: float, back_armhole: float, inputs: GeometryInputs):
    mapping = {item.boundary_id: item for item in boundaries}
    front_pitch_norm = min(inputs.front_pitch_notch_m / front_armhole, 0.72)
    back_pitch_norm = 0.42
    return (
        _notch("FRONT_PITCH", "FRONT_PITCH", mapping["cap_front"], segments, 1.0 - front_pitch_norm),
        _notch("FRONT_SHOULDER_POINT", "SHOULDER_POINT", mapping["cap_front"], segments, 0.0),
        _notch("BACK_PITCH", "BACK_PITCH", mapping["cap_back"], segments, 1.0 - back_pitch_norm),
        _notch("BACK_SHOULDER_POINT", "SHOULDER_POINT", mapping["cap_back"], segments, 0.0),
    )


def build_left_sleeve(inputs: GeometryInputs, front_armhole: float, back_armhole: float) -> ComponentGeometryAuthority:
    front_target = front_armhole + inputs.cap_ease_m * 0.45
    back_target = back_armhole + inputs.cap_ease_m * 0.55
    segments, landmarks = _build_segments(inputs, front_target, back_target)
    boundaries = _boundaries()
    segment_map = {item.segment_id: item for item in segments}
    return ComponentGeometryAuthority(
        instance_id="sleeve_left",
        component_id="SET_IN_SLEEVE_BASIC",
        component_revision=2,
        parameter_set_sha256=inputs.resolved_set_sha256,
        mirror_source_instance_id=None,
        segments=segments,
        boundaries=boundaries,
        notches=_notches(boundaries, segment_map, front_armhole, back_armhole, inputs),
        landmarks=landmarks,
        internal_segment_ids=("biceps_line", "elbow_line", "grainline"),
    )


def mirror_sleeve(left: ComponentGeometryAuthority) -> ComponentGeometryAuthority:
    mirrored_segments = tuple(mirror_segment(item, item.segment_id) for item in left.segments)
    mirrored_landmarks = {key: Point2(-value.x, value.y) for key, value in left.landmarks.items()}
    segment_map = {item.segment_id: item for item in mirrored_segments}
    notches = []
    for source in left.notches:
        boundary = left.boundary(source.boundary_id)
        point, normalized = point_at_arc(boundary, segment_map, source.arc_length_m)
        notches.append(NotchPlacement(source.notch_id, source.semantic_role, source.boundary_id, source.arc_length_m, normalized, point))
    return ComponentGeometryAuthority(
        instance_id="sleeve_right",
        component_id=left.component_id,
        component_revision=left.component_revision,
        parameter_set_sha256=left.parameter_set_sha256,
        mirror_source_instance_id=left.instance_id,
        segments=mirrored_segments,
        boundaries=left.boundaries,
        notches=tuple(notches),
        landmarks=mirrored_landmarks,
        internal_segment_ids=left.internal_segment_ids,
    )
