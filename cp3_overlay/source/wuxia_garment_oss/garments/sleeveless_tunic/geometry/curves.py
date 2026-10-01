"""Compile tunic-specific boundary curves from CP2 named landmarks."""
from __future__ import annotations

from math import hypot

from ....sizing.geometry.curve import Curve2D, count_for_length, line, quadratic_through

TARGET_BOUNDARY_EDGE_M = 0.012


_OUTLINE_PLANS = {
    "bodice_front": (
        ("shoulder_left", True), ("armhole_left", True), ("side_left", True),
        ("waist", True), ("side_right", True), ("armhole_right", True),
        ("shoulder_right", True), ("neckline", False),
    ),
    "bodice_back": (
        ("shoulder_left", True), ("armhole_left", True), ("side_left", True),
        ("waist", True), ("side_right", True), ("armhole_right", True),
        ("shoulder_right", True), ("neckline", False),
    ),
    "skirt_front": (
        ("waist", True), ("side_right", False), ("hem", False), ("side_left", False),
    ),
    "skirt_back": (
        ("waist", True), ("side_right", False), ("hem", False), ("side_left", False),
    ),
}


def _through_t(points: list[list[float]]) -> float:
    start, middle, end = points
    denominator = start[1] - end[1]
    if abs(denominator) <= 1.0e-12:
        return 0.5
    return min(0.9, max(0.1, (start[1] - middle[1]) / denominator))


def boundary_curve(panel_id: str, boundary: dict, landmarks: dict[str, list[float]]) -> Curve2D:
    points = [landmarks[name] for name in boundary["landmark_order"]]
    if len(points) == 2:
        return line(points[0], points[1])
    if len(points) != 3:
        raise ValueError(f"unsupported landmark count for {panel_id}:{boundary['boundary_id']}")
    through_t = _through_t(points) if panel_id.startswith("skirt_") else 0.5
    return quadratic_through(points[0], points[1], points[2], through_t)


def _maximum_spacing(samples: list[tuple[float, float]]) -> float:
    return max(
        hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(samples, samples[1:])
    )


def compile_panel_curves(
    panel: dict, target_edge_m: float = TARGET_BOUNDARY_EDGE_M
) -> tuple[dict, dict[str, Curve2D]]:
    curves: dict[str, Curve2D] = {}
    boundaries = []
    for source in panel["boundaries"]:
        curve = boundary_curve(panel["panel_id"], source, panel["landmarks"])
        length = curve.length()
        minimum = 4 if curve.kind == "QUADRATIC_BEZIER" else 2
        count = count_for_length(length, target_edge_m, minimum)
        samples = curve.sample_by_arclength(count)
        curves[source["boundary_id"]] = curve
        boundaries.append({
            **source,
            "curve": curve.to_dict(),
            "length_m": length,
            "target_edge_m": target_edge_m,
            "sample_count": count,
            "maximum_sample_spacing_m": _maximum_spacing(samples),
            "samples": [[float(x), float(y)] for x, y in samples],
        })
    payload = {
        "panel_id": panel["panel_id"],
        "landmarks": panel["landmarks"],
        "boundaries": boundaries,
        "outline_plan": [
            {"boundary_id": boundary_id, "forward": forward}
            for boundary_id, forward in _OUTLINE_PLANS[panel["panel_id"]]
        ],
    }
    return payload, curves


def assemble_outline(panel_payload: dict) -> list[tuple[float, float]]:
    boundaries = {row["boundary_id"]: row for row in panel_payload["boundaries"]}
    outline: list[tuple[float, float]] = []
    for item in panel_payload["outline_plan"]:
        samples = [tuple(point) for point in boundaries[item["boundary_id"]]["samples"]]
        if not item["forward"]:
            samples.reverse()
        outline.extend(samples if not outline else samples[1:])
    if outline and hypot(outline[-1][0] - outline[0][0], outline[-1][1] - outline[0][1]) <= 1.0e-9:
        outline.pop()
    return outline
