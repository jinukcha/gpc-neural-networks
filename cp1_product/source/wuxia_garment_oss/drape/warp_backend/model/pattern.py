"""Human-pattern outlines and avatar-aware shell placement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

from ..authority import INITIAL_SEWING_GAP_M

Vec2 = Tuple[float, float]
Curve = Callable[[float], Vec2]


@dataclass(frozen=True)
class PatternBoundary:
    points: np.ndarray
    roles: Dict[str, np.ndarray]
    center: np.ndarray
    front: bool
    kind: str


def _line(a: Vec2, b: Vec2) -> Curve:
    return lambda t: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _quadratic(a: Vec2, control: Vec2, b: Vec2) -> Curve:
    def sample(t: float) -> Vec2:
        u = 1.0 - t
        return (
            u * u * a[0] + 2.0 * u * t * control[0] + t * t * b[0],
            u * u * a[1] + 2.0 * u * t * control[1] + t * t * b[1],
        )

    return sample


def _append_segment(
    points: list[Vec2],
    roles: Dict[str, list[int]],
    role: str,
    count: int,
    curve: Curve,
) -> None:
    if count <= 0:
        raise ValueError(f"{role}: non-positive boundary count")
    indices = roles.setdefault(role, [])
    for offset in range(count):
        indices.append(len(points))
        points.append(curve(offset / count))


def _freeze(points: Iterable[Vec2], roles: Dict[str, list[int]], **kwargs) -> PatternBoundary:
    array = np.asarray(list(points), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("pattern boundary must be [N,2]")
    return PatternBoundary(
        points=array,
        roles={name: np.asarray(values, dtype=np.int32) for name, values in roles.items()},
        center=np.asarray(kwargs.pop("center"), dtype=np.float64),
        **kwargs,
    )


def bodice_boundary(front: bool) -> PatternBoundary:
    waist_half = 0.244
    chest_half = 0.275 if front else 0.269
    shoulder_half = 0.205
    neck_half = 0.083
    waist_z, underarm_z = 1.060, 1.345
    shoulder_z, top_z = 1.515, 1.585
    neck_depth = 0.160 if front else 0.072
    points: list[Vec2] = []
    roles: Dict[str, list[int]] = {}

    _append_segment(points, roles, "waist", 35, _line((-waist_half, waist_z), (waist_half, waist_z)))
    _append_segment(points, roles, "side_right", 22, _line((waist_half, waist_z), (chest_half, underarm_z)))
    _append_segment(
        points,
        roles,
        "armhole_right",
        15,
        _quadratic((chest_half, underarm_z), (chest_half + 0.030, 1.430), (shoulder_half, shoulder_z)),
    )
    _append_segment(points, roles, "shoulder_right", 13, _line((shoulder_half, shoulder_z), (neck_half, top_z)))

    def neckline(t: float) -> Vec2:
        x = neck_half * (1.0 - 2.0 * t)
        normalized = x / neck_half
        z = top_z - neck_depth * (1.0 - normalized * normalized)
        return x, z

    _append_segment(points, roles, "neckline", 15, neckline)
    _append_segment(points, roles, "shoulder_left", 13, _line((-neck_half, top_z), (-shoulder_half, shoulder_z)))
    _append_segment(
        points,
        roles,
        "armhole_left",
        15,
        _quadratic((-shoulder_half, shoulder_z), (-chest_half - 0.030, 1.430), (-chest_half, underarm_z)),
    )
    _append_segment(points, roles, "side_left", 22, _line((-chest_half, underarm_z), (-waist_half, waist_z)))
    return _freeze(points, roles, center=(0.0, 1.300), front=front, kind="bodice")


def skirt_boundary(front: bool) -> PatternBoundary:
    waist_half, hem_half = 0.244, 0.310
    waist_z, hem_z = 1.060, 0.240
    hem_count = 66 if front else 65
    points: list[Vec2] = []
    roles: Dict[str, list[int]] = {}
    _append_segment(points, roles, "hem", hem_count, _line((-hem_half, hem_z), (hem_half, hem_z)))
    _append_segment(points, roles, "side_right", 67, _line((hem_half, hem_z), (waist_half, waist_z)))
    _append_segment(points, roles, "waist", 35, _line((waist_half, waist_z), (-waist_half, waist_z)))
    _append_segment(points, roles, "side_left", 66, _line((-waist_half, waist_z), (-hem_half, hem_z)))
    return _freeze(points, roles, center=(0.0, 0.650), front=front, kind="skirt")


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def garment_radii(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    levels = np.asarray((0.24, 0.72, 1.06, 1.30, 1.46, 1.585), dtype=np.float64)
    rx_values = np.asarray((0.300, 0.230, 0.166, 0.184, 0.197, 0.218), dtype=np.float64)
    ry_values = np.asarray((0.238, 0.180, 0.126, 0.143, 0.157, 0.154), dtype=np.float64)
    return np.interp(z, levels, rx_values), np.interp(z, levels, ry_values)


def _half_width(uv: np.ndarray, kind: str, front: bool) -> np.ndarray:
    z = uv[:, 1]
    if kind == "skirt":
        return np.interp(z, (0.24, 1.06), (0.310, 0.244))
    chest = 0.275 if front else 0.269
    return np.interp(z, (1.06, 1.345, 1.585), (0.244, chest, 0.205))


def arrange_on_avatar(uv: np.ndarray, kind: str, front: bool) -> np.ndarray:
    half_width = np.maximum(_half_width(uv, kind, front), 1.0e-6)
    lateral = np.clip(uv[:, 0] / half_width, -1.0, 1.0)
    z = uv[:, 1]
    rx, ry = garment_radii(z)
    theta = lateral * np.pi * 0.5
    x = rx * np.sin(theta)
    y_abs = ry * np.cos(theta) + INITIAL_SEWING_GAP_M * 0.5
    if kind == "bodice":
        shoulder = _smoothstep((z - 1.455) / 0.100) * _smoothstep((np.abs(lateral) - 0.25) / 0.65)
        y_abs = y_abs * (1.0 - shoulder) + INITIAL_SEWING_GAP_M * 0.5 * shoulder
    y = y_abs if front else -y_abs
    return np.column_stack((x, y, z)).astype(np.float64)


def force_shoulder_gap(
    positions: np.ndarray,
    boundary_indices: np.ndarray,
    role_indices: Dict[str, np.ndarray],
    front: bool,
) -> None:
    sign = 1.0 if front else -1.0
    for role in ("shoulder_left", "shoulder_right"):
        local = role_indices[role]
        global_indices = boundary_indices[local]
        positions[global_indices, 1] = sign * INITIAL_SEWING_GAP_M * 0.5
