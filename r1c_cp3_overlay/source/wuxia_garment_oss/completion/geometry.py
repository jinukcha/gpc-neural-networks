"""Raw JSON curve measurements used by CP3 diagnosis and evidence."""
from __future__ import annotations

import math


def _lerp(left, right, t: float):
    return (left[0] * (1.0 - t) + right[0] * t, left[1] * (1.0 - t) + right[1] * t)


def evaluate_segment(segment: dict, t: float):
    points = [tuple(item) for item in segment["points_m"]]
    kind = segment["kind"]
    if kind == "LINE":
        return _lerp(points[0], points[1], t)
    first = [_lerp(points[index], points[index + 1], t) for index in range(len(points) - 1)]
    if kind == "QUADRATIC_BEZIER":
        return _lerp(first[0], first[1], t)
    second = [_lerp(first[index], first[index + 1], t) for index in range(2)]
    return _lerp(second[0], second[1], t)


def sample_segment(segment: dict, steps: int = 256):
    return [evaluate_segment(segment, index / steps) for index in range(steps + 1)]


def segment_length(segment: dict, steps: int = 512) -> float:
    points = sample_segment(segment, steps)
    total = 0.0
    for index in range(len(points) - 1):
        left = points[index]
        right = points[index + 1]
        total += math.hypot(right[0] - left[0], right[1] - left[1])
    return total


def segment_map(geometry: dict) -> dict[str, dict]:
    return {item["segment_id"]: item for item in geometry["segments"]}


def boundary_map(geometry: dict) -> dict[str, dict]:
    return {item["boundary_id"]: item for item in geometry["boundaries"]}


def boundary_length(geometry: dict, boundary_id: str) -> float:
    boundary = boundary_map(geometry)[boundary_id]
    segments = segment_map(geometry)
    return sum(segment_length(segments[segment_id]) for segment_id in boundary["segment_ids"])


def max_point_delta(canonical_segment: dict, candidate_segment: dict):
    canonical_points = canonical_segment["points_m"]
    candidate_points = candidate_segment["points_m"]
    if len(canonical_points) != len(candidate_points):
        return math.inf, True
    maximum = 0.0
    endpoint_changed = False
    last = len(canonical_points) - 1
    for index, pair in enumerate(zip(canonical_points, candidate_points)):
        expected, observed = pair
        delta = math.hypot(observed[0] - expected[0], observed[1] - expected[1])
        maximum = max(maximum, delta)
        endpoint_changed = endpoint_changed or (delta > 1.0e-12 and index in {0, last})
    return maximum, endpoint_changed


def changed_segments(canonical_geometry: dict, candidate_geometry: dict) -> list[dict]:
    expected = segment_map(canonical_geometry)
    observed = segment_map(candidate_geometry)
    result = []
    for segment_id, canonical_segment in expected.items():
        candidate_segment = observed.get(segment_id)
        if candidate_segment is None:
            result.append({"segment_id": segment_id, "delta_m": math.inf, "endpoint_changed": True})
            continue
        delta, endpoint_changed = max_point_delta(canonical_segment, candidate_segment)
        if delta > 1.0e-12:
            result.append({"segment_id": segment_id, "delta_m": delta, "endpoint_changed": endpoint_changed})
    return result


def seam_lengths(snapshot: dict, seam: dict):
    geometry = snapshot["geometry_by_instance"]
    endpoint_a = seam["endpoint_a"]
    endpoint_b = seam["endpoint_b"]
    first = geometry.get(endpoint_a["component_instance_id"])
    second = geometry.get(endpoint_b["component_instance_id"])
    if first is None or second is None:
        return None
    return (
        boundary_length(first, endpoint_a["boundary_id"]),
        boundary_length(second, endpoint_b["boundary_id"]),
    )
