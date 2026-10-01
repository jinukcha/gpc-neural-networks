"""Reusable 2D curve evaluation and physical-length resampling."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot

Point = tuple[float, float]


def _point(value: list[float] | tuple[float, float]) -> Point:
    return float(value[0]), float(value[1])


def _mix(a: Point, b: Point, t: float) -> Point:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


@dataclass(frozen=True)
class Curve2D:
    kind: str
    points: tuple[Point, ...]

    def evaluate(self, t: float) -> Point:
        t = min(1.0, max(0.0, float(t)))
        if self.kind == "LINE":
            return _mix(self.points[0], self.points[1], t)
        if self.kind != "QUADRATIC_BEZIER":
            raise ValueError(f"unsupported curve kind: {self.kind}")
        p0, p1, p2 = self.points
        u = 1.0 - t
        return (
            u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
        )

    def dense_samples(self, count: int = 257) -> list[Point]:
        if count < 2:
            raise ValueError("curve sample count must be at least two")
        return [self.evaluate(index / (count - 1)) for index in range(count)]

    def length(self, dense_count: int = 257) -> float:
        points = self.dense_samples(dense_count)
        return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))

    def sample_by_arclength(self, count: int, dense_count: int = 513) -> list[Point]:
        dense = self.dense_samples(dense_count)
        cumulative = [0.0]
        for a, b in zip(dense, dense[1:]):
            cumulative.append(cumulative[-1] + hypot(b[0] - a[0], b[1] - a[1]))
        total = cumulative[-1]
        if total <= 1.0e-12:
            raise ValueError("zero-length boundary curve")
        return [_sample_dense(dense, cumulative, total * i / (count - 1)) for i in range(count)]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "control_points": [[float(x), float(y)] for x, y in self.points],
        }


def _sample_dense(points: list[Point], cumulative: list[float], distance: float) -> Point:
    if distance <= 0.0:
        return points[0]
    if distance >= cumulative[-1]:
        return points[-1]
    low, high = 0, len(cumulative) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if cumulative[middle] < distance:
            low = middle
        else:
            high = middle
    span = cumulative[high] - cumulative[low]
    local = 0.0 if span <= 1.0e-15 else (distance - cumulative[low]) / span
    return _mix(points[low], points[high], local)


def line(start: list[float], end: list[float]) -> Curve2D:
    return Curve2D("LINE", (_point(start), _point(end)))


def quadratic_through(
    start: list[float], through: list[float], end: list[float], through_t: float = 0.5
) -> Curve2D:
    p0, pm, p2 = _point(start), _point(through), _point(end)
    t = min(0.95, max(0.05, float(through_t)))
    u = 1.0 - t
    denominator = 2.0 * u * t
    control = (
        (pm[0] - u * u * p0[0] - t * t * p2[0]) / denominator,
        (pm[1] - u * u * p0[1] - t * t * p2[1]) / denominator,
    )
    return Curve2D("QUADRATIC_BEZIER", (p0, control, p2))


def count_for_length(length_m: float, target_edge_m: float, minimum: int = 2) -> int:
    if target_edge_m <= 0.0:
        raise ValueError("target edge length must be positive")
    return max(int(minimum), int(ceil(length_m / target_edge_m)) + 1)
