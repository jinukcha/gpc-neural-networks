"""Simple-polygon validation helpers for garment panels."""
from __future__ import annotations

from math import hypot

Point = tuple[float, float]
Triangle = tuple[int, int, int]


def cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def signed_area(points: list[Point]) -> float:
    return 0.5 * sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(points, points[1:] + points[:1])
    )


def remove_adjacent_duplicates(points: list[Point], tolerance: float = 1.0e-10) -> list[Point]:
    result: list[Point] = []
    for point in points:
        if not result or hypot(point[0] - result[-1][0], point[1] - result[-1][1]) > tolerance:
            result.append(point)
    if len(result) > 1 and hypot(
        result[0][0] - result[-1][0], result[0][1] - result[-1][1]
    ) <= tolerance:
        result.pop()
    return result


def remove_collinear(points: list[Point], tolerance: float = 1.0e-10) -> list[Point]:
    current = remove_adjacent_duplicates(points, tolerance)
    changed = True
    while changed and len(current) > 3:
        changed = False
        kept: list[Point] = []
        for index, point in enumerate(current):
            previous = current[index - 1]
            following = current[(index + 1) % len(current)]
            if abs(cross(previous, point, following)) <= tolerance:
                changed = True
                continue
            kept.append(point)
        current = kept
    return current


def ensure_ccw(points: list[Point]) -> list[Point]:
    cleaned = remove_collinear(points)
    return cleaned if signed_area(cleaned) > 0.0 else list(reversed(cleaned))


def _orientation(a: Point, b: Point, c: Point, tolerance: float) -> int:
    value = cross(a, b, c)
    if abs(value) <= tolerance:
        return 0
    return 1 if value > 0.0 else -1


def _on_segment(a: Point, b: Point, p: Point, tolerance: float) -> bool:
    return (
        min(a[0], b[0]) - tolerance <= p[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= p[1] <= max(a[1], b[1]) + tolerance
        and abs(cross(a, b, p)) <= tolerance
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point, tolerance: float = 1.0e-10) -> bool:
    o1 = _orientation(a, b, c, tolerance)
    o2 = _orientation(a, b, d, tolerance)
    o3 = _orientation(c, d, a, tolerance)
    o4 = _orientation(c, d, b, tolerance)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return any((
        o1 == 0 and _on_segment(a, b, c, tolerance),
        o2 == 0 and _on_segment(a, b, d, tolerance),
        o3 == 0 and _on_segment(c, d, a, tolerance),
        o4 == 0 and _on_segment(c, d, b, tolerance),
    ))


def self_intersections(points: list[Point]) -> list[tuple[int, int]]:
    intersections: list[tuple[int, int]] = []
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count}:
                continue
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            if segments_intersect(a, b, c, d):
                intersections.append((first, second))
    return intersections


def point_in_triangle(point: Point, a: Point, b: Point, c: Point, tolerance: float = 1.0e-12) -> bool:
    c1 = cross(a, b, point)
    c2 = cross(b, c, point)
    c3 = cross(c, a, point)
    return c1 >= -tolerance and c2 >= -tolerance and c3 >= -tolerance


def triangle_area(points: list[Point], triangle: Triangle) -> float:
    a, b, c = (points[index] for index in triangle)
    return abs(cross(a, b, c)) * 0.5
