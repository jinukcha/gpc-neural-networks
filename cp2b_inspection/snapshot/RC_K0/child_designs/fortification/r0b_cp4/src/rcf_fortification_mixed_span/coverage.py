from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .model import COVERAGE_SCHEMA, MixedSpanContractError, cross, dot, length, normalize, sub, _round, _rv

SURFACE_ROLE_ORDER = ("TOP", "BOTTOM", "END_CAP", "START_CAP", "INSIDE", "OUTSIDE")


def _triangle_normal(vertices: Sequence[Sequence[float]], triangle: Sequence[int]) -> tuple[float, float, float]:
    a, b, c = (vertices[int(index)] for index in triangle)
    normal = cross(sub(b, a), sub(c, a))
    if length(normal) <= 1e-15:
        raise MixedSpanContractError(f"degenerate triangle {list(triangle)}")
    return normalize(normal)


def _centroid(vertices: Sequence[Sequence[float]], triangle: Sequence[int]) -> tuple[float, float, float]:
    points = [vertices[int(index)] for index in triangle]
    return tuple(sum(float(point[axis]) for point in points) / 3.0 for axis in range(3))


def _nearest_frame(frames: Sequence[Mapping[str, Any]], point: Sequence[float]) -> Mapping[str, Any]:
    if not frames:
        raise MixedSpanContractError("surface classifier requires at least one local frame")
    return min(
        frames,
        key=lambda frame: sum((float(frame["origin_m"][axis]) - float(point[axis])) ** 2 for axis in range(3)),
    )


def classify_surface(
    vertices: Sequence[Sequence[float]],
    triangle: Sequence[int],
    frames: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, float]]:
    normal = _triangle_normal(vertices, triangle)
    frame = _nearest_frame(frames, _centroid(vertices, triangle))
    up_score = dot(normal, frame["up"])
    tangent_score = dot(normal, frame["tangent"])
    inside_score = dot(normal, frame["inside"])
    # Deterministic axis preference resolves exact diagonal ties without provider face identity.
    candidates = [
        (abs(up_score), 3, "UP", up_score),
        (abs(tangent_score), 2, "TANGENT", tangent_score),
        (abs(inside_score), 1, "INSIDE", inside_score),
    ]
    _, _, axis, signed = max(candidates)
    if axis == "UP":
        role = "TOP" if signed >= 0.0 else "BOTTOM"
    elif axis == "TANGENT":
        role = "END_CAP" if signed >= 0.0 else "START_CAP"
    else:
        role = "INSIDE" if signed >= 0.0 else "OUTSIDE"
    scores = {
        "up": _round(up_score, 12),
        "tangent": _round(tangent_score, 12),
        "inside": _round(inside_score, 12),
        "selected_abs_score": _round(abs(signed), 12),
    }
    return role, scores


def _ranges(indices: Sequence[int]) -> list[dict[str, int]]:
    values = sorted(set(int(value) for value in indices))
    if not values:
        return []
    rows: list[dict[str, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            rows.append({"offset": start, "count": previous - start + 1})
            start = value
        previous = value
    rows.append({"offset": start, "count": previous - start + 1})
    return rows


def build_module_coverage(
    *,
    module_id: str,
    module_kind: str,
    family: str,
    source_mesh: Mapping[str, Any],
    source_ranges: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    global_triangle_offset: int,
    source_refs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vertices = source_mesh["vertices_m"]
    triangles = source_mesh["triangles"]
    rows: list[dict[str, Any]] = []
    source_owners: dict[int, str] = {}
    global_owners: dict[int, str] = {}
    score_floor = 1.0
    for source_range in source_ranges:
        start = int(source_range["triangle_offset"])
        end = start + int(source_range["triangle_count"])
        if not 0 <= start < end <= len(triangles):
            raise MixedSpanContractError(f"invalid source range for {module_id}: {source_range}")
        grouped: dict[str, list[tuple[int, dict[str, float]]]] = {role: [] for role in SURFACE_ROLE_ORDER}
        for source_triangle_index in range(start, end):
            if source_triangle_index in source_owners:
                raise MixedSpanContractError(f"source triangle overlap in {module_id}: {source_triangle_index}")
            role, scores = classify_surface(vertices, triangles[source_triangle_index], frames)
            grouped[role].append((source_triangle_index, scores))
            score_floor = min(score_floor, float(scores["selected_abs_score"]))
        range_id = str(source_range["range_id"])
        for role in SURFACE_ROLE_ORDER:
            items = grouped[role]
            if not items:
                continue
            source_indices = [item[0] for item in items]
            global_indices = [global_triangle_offset + index for index in source_indices]
            surface_id = f"surface/{module_id}/{range_id}/{role.lower()}"
            for source_index, global_index in zip(source_indices, global_indices):
                source_owners[source_index] = surface_id
                if global_index in global_owners:
                    raise MixedSpanContractError(f"global triangle overlap in {module_id}: {global_index}")
                global_owners[global_index] = surface_id
            row = {
                "surface_id": surface_id,
                "module_id": module_id,
                "module_kind": module_kind,
                "family": family,
                "range_id": range_id,
                "part_id": source_range.get("part_id"),
                "unit_id": source_range.get("unit_id"),
                "segment_id": source_range.get("segment_id"),
                "semantic_role": source_range.get("semantic_role"),
                "surface_role": role,
                "source_triangle_indices": source_indices,
                "source_triangle_ranges": _ranges(source_indices),
                "global_triangle_indices": global_indices,
                "global_triangle_ranges": _ranges(global_indices),
                "triangle_count": len(source_indices),
                "classification_policy": "DOMINANT_NORMAL_IN_NEAREST_PROJECT_LOCAL_FRAME",
                "minimum_selected_abs_score": _round(min(float(item[1]["selected_abs_score"]) for item in items), 12),
                "source_refs": dict(source_refs),
            }
            rows.append(row)
    expected = set(range(len(triangles)))
    observed = set(source_owners)
    gaps = sorted(expected - observed)
    overlaps = []
    if gaps:
        raise MixedSpanContractError(f"module source coverage gap {module_id}: {gaps}")
    summary = {
        "module_id": module_id,
        "module_kind": module_kind,
        "family": family,
        "source_triangle_count": len(triangles),
        "covered_triangle_count": len(observed),
        "surface_count": len(rows),
        "gap_count": len(gaps),
        "overlap_count": len(overlaps),
        "minimum_selected_abs_score": _round(score_floor, 12),
    }
    return rows, summary


def build_assembly_coverage(
    assembly_id: str,
    module_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    total_triangles: int,
) -> dict[str, Any]:
    owners: dict[int, list[str]] = {}
    for row in coverage_rows:
        for triangle_index in row["global_triangle_indices"]:
            owners.setdefault(int(triangle_index), []).append(str(row["surface_id"]))
    expected = set(range(int(total_triangles)))
    gaps = sorted(expected - set(owners))
    overlaps = sorted(index for index, refs in owners.items() if len(refs) != 1)
    if gaps or overlaps:
        raise MixedSpanContractError(f"assembly source coverage incomplete gaps={gaps} overlaps={overlaps}")
    document = {
        "schema": COVERAGE_SCHEMA,
        "assembly_id": assembly_id,
        "coverage_policy": "EXCLUSIVE_COMPLETE_MODULE_RANGE_AND_PROJECT_NORMAL_CLASSIFICATION",
        "provider_face_identity_used": False,
        "module_summaries": [dict(row) for row in module_rows],
        "surfaces": [dict(row) for row in coverage_rows],
        "summary": {
            "module_count": len(module_rows),
            "mesh_triangle_count": int(total_triangles),
            "covered_triangle_count": len(owners),
            "unique_triangle_count": len(owners) - len(overlaps),
            "surface_count": len(coverage_rows),
            "gap_count": len(gaps),
            "overlap_count": len(overlaps),
            "gaps": gaps,
            "overlaps": overlaps,
        },
    }
    return document
