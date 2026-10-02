"""Curve evaluation and ordered component-outline sampling."""
from __future__ import annotations

import math

import numpy as np


_OUTLINE = {
    "BODICE_FRONT_BASIC": (
        ("side_left", False),
        ("armhole_left_front", False),
        ("shoulder_left", False),
        ("neckline_front", False),
        ("shoulder_right", True),
        ("armhole_right_front", True),
        ("side_right", True),
        ("hem_front", True),
    ),
    "BODICE_BACK_BASIC": (
        ("side_left", False),
        ("armhole_left_back", False),
        ("shoulder_left", False),
        ("neckline_back", False),
        ("shoulder_right", True),
        ("armhole_right_back", True),
        ("side_right", True),
        ("hem_back", True),
    ),
    "SET_IN_SLEEVE_BASIC": (
        ("cap_front", False),
        ("underarm_front", False),
        ("wrist", False),
        ("underarm_back", False),
        ("cap_back", True),
    ),
    "COLLAR_STAND_BASIC": (
        ("front_attach", False),
        ("back_attach", False),
        ("back_end", False),
        ("outer", False),
        ("front_end", False),
    ),
    "CUFF_STRAIGHT_BASIC": (
        ("sleeve_attach", False),
        ("end_right", False),
        ("outer", False),
        ("end_left", False),
    ),
}


def _point(payload) -> np.ndarray:
    return np.asarray(payload, dtype=np.float64)


def evaluate(segment: dict, t: float) -> np.ndarray:
    points = np.asarray(segment["points_m"], dtype=np.float64)
    kind = segment["kind"]
    if kind == "LINE":
        return points[0] * (1.0 - t) + points[1] * t
    first = points[:-1] * (1.0 - t) + points[1:] * t
    if kind == "QUADRATIC_BEZIER":
        return first[0] * (1.0 - t) + first[1] * t
    second = first[:-1] * (1.0 - t) + first[1:] * t
    return second[0] * (1.0 - t) + second[1] * t


def derivative(segment: dict, t: float) -> np.ndarray:
    points = np.asarray(segment["points_m"], dtype=np.float64)
    kind = segment["kind"]
    if kind == "LINE":
        return points[1] - points[0]
    delta = points[1:] - points[:-1]
    if kind == "QUADRATIC_BEZIER":
        return 2.0 * (delta[0] * (1.0 - t) + delta[1] * t)
    first = delta[:-1] * (1.0 - t) + delta[1:] * t
    return 3.0 * (first[0] * (1.0 - t) + first[1] * t)


def segment_length(segment: dict, steps: int = 160) -> float:
    points = np.asarray([evaluate(segment, index / steps) for index in range(steps + 1)])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _segment_map(geometry: dict) -> dict[str, dict]:
    return {item["segment_id"]: item for item in geometry["segments"]}


def _boundary(geometry: dict, boundary_id: str) -> dict:
    for item in geometry["boundaries"]:
        if item["boundary_id"] == boundary_id:
            return item
    raise KeyError(f"unknown boundary {geometry['instance_id']}.{boundary_id}")


def boundary_length(geometry: dict, boundary_id: str) -> float:
    segments = _segment_map(geometry)
    boundary = _boundary(geometry, boundary_id)
    return sum(segment_length(segments[item]) for item in boundary["segment_ids"])


def sample_boundary(geometry: dict, boundary_id: str, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("boundary sample count must be >= 2")
    segments = _segment_map(geometry)
    boundary = _boundary(geometry, boundary_id)
    lengths = [segment_length(segments[item]) for item in boundary["segment_ids"]]
    total = sum(lengths)
    targets = np.linspace(0.0, total, count)
    points = []
    for target in targets:
        accumulated = 0.0
        for index, segment_id in enumerate(boundary["segment_ids"]):
            length = lengths[index]
            if target <= accumulated + length + 1.0e-12:
                local = 0.0 if length == 0.0 else (target - accumulated) / length
                points.append(evaluate(segments[segment_id], min(max(local, 0.0), 1.0)))
                break
            accumulated += length
        else:
            points.append(evaluate(segments[boundary["segment_ids"][-1]], 1.0))
    result = np.asarray(points, dtype=np.float64)
    if boundary.get("orientation") == "REVERSE":
        result = result[::-1].copy()
    return result


def outline_spec(component_id: str):
    if component_id not in _OUTLINE:
        raise KeyError(f"no CP4-R1 outline mapping for {component_id}")
    return _OUTLINE[component_id]


def sample_outline(
    geometry: dict,
    boundary_counts: dict[str, int],
    default_spacing_m: float = 0.028,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    vertices: list[np.ndarray] = []
    boundary_indices: dict[str, np.ndarray] = {}
    first_index: int | None = None
    for boundary_id, reverse in outline_spec(geometry["component_id"]):
        length = boundary_length(geometry, boundary_id)
        count = boundary_counts.get(boundary_id, max(3, int(math.ceil(length / default_spacing_m)) + 1))
        points = sample_boundary(geometry, boundary_id, count)
        if reverse:
            points = points[::-1].copy()
        indices = []
        for point_index, point in enumerate(points):
            if vertices and np.linalg.norm(point - vertices[-1]) <= 1.0e-9:
                indices.append(len(vertices) - 1)
                continue
            if first_index is not None and point_index == len(points) - 1 and np.linalg.norm(point - vertices[first_index]) <= 1.0e-9:
                indices.append(first_index)
                continue
            if first_index is None:
                first_index = 0
            vertices.append(point)
            indices.append(len(vertices) - 1)
        boundary_indices[boundary_id] = np.asarray(indices, dtype=np.int32)
    result = np.asarray(vertices, dtype=np.float64)
    if len(result) < 3:
        raise ValueError(f"component outline too small: {geometry['instance_id']}")
    return result, boundary_indices
