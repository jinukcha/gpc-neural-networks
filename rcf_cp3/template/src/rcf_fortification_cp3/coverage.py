from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

COVERAGE_SCHEMA = "royal-capital.fortification.source-coverage-map/1"
RESULT_SCHEMA = "royal-capital.fortification.cp3-result/1"
RECEIPT_SCHEMA = "royal-capital.fortification.cp3-qualification-receipt/1"
FAILURE_SCHEMA = "royal-capital.fortification.cp3-failure/1"
PART_ORDER = ("foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet")
SURFACE_ORDER = ("span_start", "span_end", "bottom", "top", "outside", "inside")
NORMALS = {
    "span_start": [-1.0, 0.0, 0.0], "span_end": [1.0, 0.0, 0.0],
    "bottom": [0.0, -1.0, 0.0], "top": [0.0, 1.0, 0.0],
    "outside": [0.0, 0.0, -1.0], "inside": [0.0, 0.0, 1.0],
}


class Cp3FailureCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    STALE_INPUT = "STALE_INPUT"
    RUNTIME_MISMATCH = "RUNTIME_MISMATCH"
    GEOMETRY_BUDGET_EXCEEDED = "GEOMETRY_BUDGET_EXCEEDED"
    REQUIRED_PART_FAILED = "REQUIRED_PART_FAILED"
    SOURCE_COVERAGE_FAILED = "SOURCE_COVERAGE_FAILED"
    PUBLISH_ABORTED = "PUBLISH_ABORTED"
    PROVIDER_EXECUTION_FAILED = "PROVIDER_EXECUTION_FAILED"


class Cp3Error(RuntimeError):
    def __init__(self, code: Cp3FailureCode | str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = message


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    if not root.exists():
        return "sha256:" + h.hexdigest()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode()
        h.update(rel + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + h.hexdigest()


def _ranges(indices: Sequence[int]) -> list[dict[str, int]]:
    values = sorted(set(int(x) for x in indices))
    out: list[dict[str, int]] = []
    if not values:
        return out
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            out.append({"offset": start, "count": previous - start + 1})
            start = value
        previous = value
    out.append({"offset": start, "count": previous - start + 1})
    return out


def _surface_for_triangle(vertices: list[list[float]], triangle: Sequence[int], bounds: Mapping[str, list[float]], epsilon: float) -> tuple[str, str, float]:
    points = [vertices[int(i)] for i in triangle]
    minimum, maximum = bounds["min"], bounds["max"]
    candidates: list[tuple[str, str, float]] = []
    planes = (
        ("span_start", "X", minimum[0], 0), ("span_end", "X", maximum[0], 0),
        ("bottom", "Y", minimum[1], 1), ("top", "Y", maximum[1], 1),
        ("outside", "Z", minimum[2], 2), ("inside", "Z", maximum[2], 2),
    )
    for role, axis, coordinate, component in planes:
        if all(abs(float(point[component]) - float(coordinate)) <= epsilon for point in points):
            candidates.append((role, axis, float(coordinate)))
    if len(candidates) != 1:
        raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"triangle {list(triangle)} maps to {len(candidates)} source planes: {candidates}")
    return candidates[0]


def build_source_coverage(mesh: Mapping[str, Any], parts: Mapping[str, Any], stored: Mapping[str, Any], *, linear_tolerance_m: float) -> dict[str, Any]:
    vertices = mesh["vertices_m"]
    triangles = mesh["triangles"]
    part_ranges = {row["part_id"]: row for row in mesh["part_ranges"]}
    part_docs = {row["part_id"]: row for row in parts["parts"]}
    if tuple(mesh["part_order"]) != PART_ORDER or tuple(part_docs) != PART_ORDER:
        raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, "part order mismatch")
    epsilon = max(1e-9, float(linear_tolerance_m) * 2.0)
    surface_rows: list[dict[str, Any]] = []
    assigned: dict[int, list[str]] = {}
    for part_id in PART_ORDER:
        part_range = part_ranges[part_id]
        part_doc = part_docs[part_id]
        start = int(part_range["triangle_offset"])
        end = start + int(part_range["triangle_count"])
        grouped = {role: [] for role in SURFACE_ORDER}
        for triangle_index in range(start, end):
            role, axis, coordinate = _surface_for_triangle(vertices, triangles[triangle_index], part_doc["bounds_m"], epsilon)
            grouped[role].append(triangle_index)
            assigned.setdefault(triangle_index, []).append(f"{part_id}:{role}")
        stored_refs = [
            {"format": row["format"], "path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in stored["copies"] if row.get("part_id") == part_id
        ]
        if {row["format"] for row in stored_refs} != {"STEP", "BREP"}:
            raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"stored-copy refs incomplete for {part_id}")
        for role in SURFACE_ORDER:
            indices = grouped[role]
            if not indices:
                raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"surface {part_id}:{role} has no triangles")
            axis = "X" if role.startswith("span_") else ("Y" if role in {"bottom", "top"} else "Z")
            coordinate = {
                "span_start": part_doc["bounds_m"]["min"][0], "span_end": part_doc["bounds_m"]["max"][0],
                "bottom": part_doc["bounds_m"]["min"][1], "top": part_doc["bounds_m"]["max"][1],
                "outside": part_doc["bounds_m"]["min"][2], "inside": part_doc["bounds_m"]["max"][2],
            }[role]
            surface_rows.append({
                "surface_id": f"surface/{part_id}/{role}", "part_id": part_id,
                "semantic_role": role.upper(), "normal_project": NORMALS[role],
                "plane": {"axis": axis, "coordinate_m": float(coordinate)},
                "triangle_indices": indices, "triangle_ranges": _ranges(indices), "triangle_count": len(indices),
                "part_triangle_range": {"offset": start, "count": end - start},
                "provider_result_ref": part_doc["provider_result_ref"],
                "provider_receipt_ref": part_doc["provider_receipt_ref"],
                "stored_copy_refs": stored_refs,
                "derivation": "PROJECT_ANALYTIC_BOUNDS_AND_VERTEX_PLANE_CLASSIFICATION",
            })
    all_indices = set(range(len(triangles)))
    covered = set(assigned)
    overlaps = sorted(index for index, owners in assigned.items() if len(owners) != 1)
    gaps = sorted(all_indices - covered)
    document = {
        "schema": COVERAGE_SCHEMA, "span_id": parts["span_id"], "units": "METER",
        "frame": mesh["frame"], "surface_order": list(SURFACE_ORDER), "part_order": list(PART_ORDER),
        "classification_epsilon_m": epsilon, "provider_face_order_trusted": False,
        "surfaces": surface_rows,
        "summary": {"mesh_triangle_count": len(triangles), "covered_triangle_count": len(covered), "unique_triangle_count": len(covered) - len(overlaps), "surface_count": len(surface_rows), "gap_count": len(gaps), "overlap_count": len(overlaps), "gaps": gaps, "overlaps": overlaps},
    }
    validate_source_coverage(document, mesh)
    return document


def validate_source_coverage(document: Mapping[str, Any], mesh: Mapping[str, Any]) -> None:
    if document.get("schema") != COVERAGE_SCHEMA:
        raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, "coverage schema mismatch")
    triangle_count = len(mesh["triangles"])
    owners: dict[int, list[str]] = {}
    expected_surface_ids = {f"surface/{part}/{role}" for part in PART_ORDER for role in SURFACE_ORDER}
    observed_ids: set[str] = set()
    part_ranges = {row["part_id"]: row for row in mesh["part_ranges"]}
    for row in document.get("surfaces", []):
        sid = row.get("surface_id")
        if sid in observed_ids:
            raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"duplicate surface id {sid}")
        observed_ids.add(sid)
        part_id = row.get("part_id")
        if part_id not in part_ranges:
            raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"unknown part {part_id}")
        pr = part_ranges[part_id]
        start, end = int(pr["triangle_offset"]), int(pr["triangle_offset"]) + int(pr["triangle_count"])
        indices = [int(x) for x in row.get("triangle_indices", [])]
        if len(indices) != len(set(indices)) or not indices:
            raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"invalid triangle list for {sid}")
        if any(index < start or index >= end or index < 0 or index >= triangle_count for index in indices):
            raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"triangle outside part range for {sid}")
        for index in indices:
            owners.setdefault(index, []).append(str(sid))
    if observed_ids != expected_surface_ids:
        raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"surface set mismatch missing={sorted(expected_surface_ids-observed_ids)} extra={sorted(observed_ids-expected_surface_ids)}")
    gaps = sorted(set(range(triangle_count)) - set(owners))
    overlaps = sorted(index for index, refs in owners.items() if len(refs) != 1)
    if gaps or overlaps:
        raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, f"coverage gaps={gaps} overlaps={overlaps}")
    summary = document.get("summary", {})
    if summary.get("gap_count") != 0 or summary.get("overlap_count") != 0 or summary.get("covered_triangle_count") != triangle_count or summary.get("surface_count") != 30:
        raise Cp3Error(Cp3FailureCode.SOURCE_COVERAGE_FAILED, "coverage summary mismatch")
