"""Deterministic 2D construction-line derivation from exact PatternDocument curves."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..pattern_cad.document.model import PatternDocument
from .model import BoundaryInterval, ClosureSpec, FacingSpec


Point2 = tuple[float, float]


def _lerp(a: Point2, b: Point2, t: float) -> Point2:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def curve_point(curve_type: str, points: Sequence[Sequence[float]], t: float) -> Point2:
    rows = [(float(item[0]), float(item[1])) for item in points]
    if curve_type == "LINE":
        return _lerp(rows[0], rows[1], t)
    if curve_type == "QUADRATIC_BEZIER":
        a = _lerp(rows[0], rows[1], t)
        b = _lerp(rows[1], rows[2], t)
        return _lerp(a, b, t)
    if curve_type == "CUBIC_BEZIER":
        a = _lerp(rows[0], rows[1], t)
        b = _lerp(rows[1], rows[2], t)
        c = _lerp(rows[2], rows[3], t)
        return _lerp(_lerp(a, b, t), _lerp(b, c, t), t)
    raise ValueError(f"unsupported construction curve type: {curve_type}")


def sample_curve(curve: Mapping[str, object], interval: BoundaryInterval, count: int = 64) -> list[Point2]:
    if count < 2:
        raise ValueError("curve sample count must be at least two")
    start, end = interval.start_fraction, interval.end_fraction
    return [
        curve_point(str(curve["curve_type"]), curve["points"], start + (end - start) * index / count)
        for index in range(count + 1)
    ]


def panel_centroid(document: PatternDocument, resolved: Mapping[str, object], panel_id: str) -> Point2:
    rows = [
        resolved["points"][point_id]
        for point_id, point in document.points.items()
        if point.panel_id == panel_id
    ]
    if not rows:
        raise ValueError(f"panel has no resolved points: {panel_id}")
    return sum(float(row[0]) for row in rows) / len(rows), sum(float(row[1]) for row in rows) / len(rows)


def _unit_outward_normal(points: Sequence[Point2], index: int, centroid: Point2) -> Point2:
    left = points[max(0, index - 1)]
    right = points[min(len(points) - 1, index + 1)]
    dx, dy = right[0] - left[0], right[1] - left[1]
    length = math.hypot(dx, dy)
    if length <= 1.0e-12:
        return 0.0, 0.0
    nx, ny = -dy / length, dx / length
    rx, ry = points[index][0] - centroid[0], points[index][1] - centroid[1]
    if nx * rx + ny * ry < 0.0:
        nx, ny = -nx, -ny
    return nx, ny


def offset_polyline(points: Sequence[Point2], centroid: Point2, offset_m: float) -> list[Point2]:
    result = []
    for index, point in enumerate(points):
        nx, ny = _unit_outward_normal(points, index, centroid)
        result.append((point[0] + nx * offset_m, point[1] + ny * offset_m))
    return result


def polyline_length(points: Sequence[Point2]) -> float:
    return sum(math.dist(points[index - 1], points[index]) for index in range(1, len(points)))


def line_record(
    document: PatternDocument,
    resolved: Mapping[str, object],
    boundary: BoundaryInterval,
    allowance_m: float,
) -> dict:
    boundary.validate(document)
    curve = resolved["curves"][boundary.curve_id]
    stitch = sample_curve(curve, boundary)
    centroid = panel_centroid(document, resolved, boundary.panel_id)
    cut = offset_polyline(stitch, centroid, allowance_m)
    return {
        "panel_id": boundary.panel_id,
        "curve_id": boundary.curve_id,
        "interval": [boundary.start_fraction, boundary.end_fraction],
        "allowance_m": allowance_m,
        "stitch_line": [list(item) for item in stitch],
        "cut_line": [list(item) for item in cut],
        "stitch_length_m": polyline_length(stitch),
        "cut_length_m": polyline_length(cut),
        "sample_count": len(stitch),
    }


def facing_record(
    document: PatternDocument,
    resolved: Mapping[str, object],
    spec: FacingSpec,
) -> dict:
    spec.validate(document)
    outer_rows = []
    inner_rows = []
    centroid = panel_centroid(document, resolved, spec.owner_panel_id)
    for curve_id in spec.source_curve_ids:
        boundary = BoundaryInterval(spec.owner_panel_id, curve_id)
        curve = resolved["curves"][curve_id]
        outer = sample_curve(curve, boundary)
        inner = offset_polyline(outer, centroid, -spec.depth_m)
        outer_rows.extend(outer)
        inner_rows.extend(inner)
    return {
        **spec.to_dict(),
        "outer_stitch_line": [list(item) for item in outer_rows],
        "inner_cut_line": [list(item) for item in inner_rows],
        "outer_length_m": polyline_length(outer_rows),
        "inner_length_m": polyline_length(inner_rows),
    }


def closure_record(
    document: PatternDocument,
    resolved: Mapping[str, object],
    spec: ClosureSpec,
) -> dict:
    spec.validate(document)
    start = resolved["points"][spec.start_point_id]
    dx, dy = spec.direction_xy
    norm = math.hypot(dx, dy)
    direction = dx / norm, dy / norm
    end = [float(start[0]) + direction[0] * spec.length_m, float(start[1]) + direction[1] * spec.length_m]
    return {
        **spec.to_dict(),
        "internal_cut_line": [[float(start[0]), float(start[1])], end],
        "stitch_guard_offset_m": 0.006,
    }
