"""Deterministic exact-curve evaluation and physical arc-length utilities."""
from __future__ import annotations

from bisect import bisect_left
import math

from .model import BoundaryGeometry, CurveSegment, Point2


def _lerp(left: Point2, right: Point2, t: float) -> Point2:
    return Point2(left.x * (1.0 - t) + right.x * t, left.y * (1.0 - t) + right.y * t)


def evaluate(segment: CurveSegment, t: float) -> Point2:
    segment.validate()
    if not 0.0 <= t <= 1.0:
        raise ValueError("curve parameter must be in [0, 1]")
    points = segment.points
    if segment.kind == "LINE":
        return _lerp(points[0], points[1], t)
    first = tuple(_lerp(points[index], points[index + 1], t) for index in range(len(points) - 1))
    if segment.kind == "QUADRATIC_BEZIER":
        return _lerp(first[0], first[1], t)
    second = (_lerp(first[0], first[1], t), _lerp(first[1], first[2], t))
    return _lerp(second[0], second[1], t)


def derivative(segment: CurveSegment, t: float) -> Point2:
    points = segment.points
    if segment.kind == "LINE":
        return Point2(points[1].x - points[0].x, points[1].y - points[0].y)
    if segment.kind == "QUADRATIC_BEZIER":
        left = Point2(points[1].x - points[0].x, points[1].y - points[0].y)
        right = Point2(points[2].x - points[1].x, points[2].y - points[1].y)
        value = _lerp(left, right, t)
        return Point2(2.0 * value.x, 2.0 * value.y)
    a = Point2(points[1].x - points[0].x, points[1].y - points[0].y)
    b = Point2(points[2].x - points[1].x, points[2].y - points[1].y)
    c = Point2(points[3].x - points[2].x, points[3].y - points[2].y)
    first = _lerp(a, b, t)
    second = _lerp(b, c, t)
    value = _lerp(first, second, t)
    return Point2(3.0 * value.x, 3.0 * value.y)


def _speed(segment: CurveSegment, t: float) -> float:
    value = derivative(segment, t)
    return math.hypot(value.x, value.y)


def _simpson(segment: CurveSegment, left: float, right: float) -> float:
    middle = 0.5 * (left + right)
    return (right - left) * (_speed(segment, left) + 4.0 * _speed(segment, middle) + _speed(segment, right)) / 6.0


def _adaptive_length(segment: CurveSegment, left: float, right: float, whole: float, tolerance: float, depth: int) -> float:
    middle = 0.5 * (left + right)
    first = _simpson(segment, left, middle)
    second = _simpson(segment, middle, right)
    if depth <= 0 or abs(first + second - whole) <= 15.0 * tolerance:
        return first + second + (first + second - whole) / 15.0
    return _adaptive_length(segment, left, middle, first, tolerance * 0.5, depth - 1) + _adaptive_length(segment, middle, right, second, tolerance * 0.5, depth - 1)


def segment_length(segment: CurveSegment, tolerance: float = 1.0e-10) -> float:
    segment.validate()
    if segment.kind == "LINE":
        return math.hypot(segment.points[1].x - segment.points[0].x, segment.points[1].y - segment.points[0].y)
    whole = _simpson(segment, 0.0, 1.0)
    return _adaptive_length(segment, 0.0, 1.0, whole, tolerance, 16)


def boundary_length(boundary: BoundaryGeometry, segment_map: dict[str, CurveSegment]) -> float:
    boundary.validate()
    return sum(segment_length(segment_map[segment_id]) for segment_id in boundary.segment_ids)


def _boundary_samples(boundary: BoundaryGeometry, segment_map: dict[str, CurveSegment], steps: int = 256):
    points: list[Point2] = []
    cumulative: list[float] = []
    total = 0.0
    for segment_id in boundary.segment_ids:
        segment = segment_map[segment_id]
        for index in range(steps + 1):
            if points and index == 0:
                continue
            point = evaluate(segment, index / steps)
            if points:
                total += math.hypot(point.x - points[-1].x, point.y - points[-1].y)
            points.append(point)
            cumulative.append(total)
    return points, cumulative


def point_at_arc(boundary: BoundaryGeometry, segment_map: dict[str, CurveSegment], arc_length_m: float) -> tuple[Point2, float]:
    points, cumulative = _boundary_samples(boundary, segment_map)
    total = cumulative[-1]
    tolerance = max(1.0e-7, total * 1.0e-5)
    if arc_length_m < -tolerance or arc_length_m > total + tolerance:
        raise ValueError(f"arc length outside boundary: {boundary.boundary_id}")
    target = min(max(arc_length_m, 0.0), total)
    index = bisect_left(cumulative, target)
    if index <= 0:
        return points[0], 0.0
    if index >= len(points):
        return points[-1], 1.0
    before, after = cumulative[index - 1], cumulative[index]
    ratio = 0.0 if after == before else (target - before) / (after - before)
    point = _lerp(points[index - 1], points[index], ratio)
    return point, 0.0 if total == 0.0 else target / total


def mirror_segment(segment: CurveSegment, new_id: str) -> CurveSegment:
    return CurveSegment(new_id, segment.kind, tuple(Point2(-point.x, point.y) for point in segment.points))
