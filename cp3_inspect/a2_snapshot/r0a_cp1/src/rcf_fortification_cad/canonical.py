from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable

from .contract import PROJECT_FRAME_ID, PROVIDER_FRAME_ID, Vec3

PROJECT_TO_PROVIDER_MATRIX = (
    (1000.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1000.0, 0.0),
    (0.0, 1000.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
PROVIDER_TO_PROJECT_MATRIX = (
    (0.001, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.001, 0.0),
    (0.0, -0.001, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_ref(value: bytes) -> str:
    return "sha256:" + sha256_bytes(value)


def project_to_provider(v: Vec3 | tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = (v.x, v.y, v.z) if isinstance(v, Vec3) else v
    return (float(x) * 1000.0, float(-z) * 1000.0, float(y) * 1000.0)


def provider_to_project(v: Any) -> tuple[float, float, float]:
    if hasattr(v, "X"):
        x, y, z = float(v.X), float(v.Y), float(v.Z)
    else:
        x, y, z = map(float, v)
    return (x / 1000.0, z / 1000.0, -y / 1000.0)


def normalize(v: Vec3) -> Vec3:
    length = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if length == 0:
        raise ValueError("zero vector")
    return Vec3(v.x / length, v.y / length, v.z / length)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def add_scaled(origin: Vec3, x_axis: Vec3, y_axis: Vec3, point: tuple[float, float]) -> Vec3:
    return Vec3(origin.x + point[0] * x_axis.x + point[1] * y_axis.x, origin.y + point[0] * x_axis.y + point[1] * y_axis.y, origin.z + point[0] * x_axis.z + point[1] * y_axis.z)


def _round(value: float, digits: int = 9) -> float:
    result = round(float(value), digits)
    return 0.0 if result == -0.0 else result


def canonicalize_mesh(provider_vertices: Iterable[Any], provider_triangles: Iterable[tuple[int, int, int]]) -> dict[str, Any]:
    converted = [tuple(_round(x) for x in provider_to_project(v)) for v in provider_vertices]
    unique = sorted(set(converted))
    index_of = {point: index for index, point in enumerate(unique)}
    source_to_unique = [index_of[point] for point in converted]
    triangles = []
    for raw in provider_triangles:
        tri = tuple(source_to_unique[i] for i in raw)
        rotations = (tri, (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1]))
        triangles.append(min(rotations))
    triangles.sort()
    if not unique or not triangles:
        raise ValueError("neutral mesh is empty")
    bounds_min = [min(point[i] for point in unique) for i in range(3)]
    bounds_max = [max(point[i] for point in unique) for i in range(3)]
    return {
        "schema": "royal-capital.fortification.neutral-indexed-mesh/1",
        "units": "METER",
        "frame": PROJECT_FRAME_ID,
        "vertices_m": [list(point) for point in unique],
        "triangles": [list(tri) for tri in triangles],
        "bounds_m": {"min": bounds_min, "max": bounds_max},
    }


def normalized_step_sha256(data: bytes) -> str:
    text = data.decode("ascii")
    normalized, count = re.subn(r"FILE_NAME\(\s*'[^']*'\s*,\s*'[^']*'", "FILE_NAME('<NORMALIZED_NAME>','1970-01-01T00:00:00'", text, count=1)
    if count != 1:
        raise ValueError("STEP FILE_NAME header not found exactly once")
    return sha256_bytes(normalized.replace("\r\n", "\n").encode("ascii"))


def unit_frame_contract() -> dict[str, Any]:
    return {
        "units": "METER",
        "project_frame": PROJECT_FRAME_ID,
        "provider_frame": PROVIDER_FRAME_ID,
        "project_to_provider_matrix": [list(row) for row in PROJECT_TO_PROVIDER_MATRIX],
        "provider_to_project_matrix": [list(row) for row in PROVIDER_TO_PROJECT_MATRIX],
        "orientation_determinant": 1.0,
        "project_unit": "METER",
        "provider_numeric_unit": "MILLIMETER",
        "project_to_provider_length_scale": 1000.0,
    }
