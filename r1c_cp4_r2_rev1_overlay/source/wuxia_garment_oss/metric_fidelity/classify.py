"""Classify CP4-R1 structural edges by component, boundary, and feature."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


TARGET_COMPONENTS = {"sleeve_left", "sleeve_right", "collar", "cuff_left", "cuff_right"}


@dataclass(frozen=True)
class FidelityInputs:
    arrays: dict[str, np.ndarray]
    component_order: tuple[str, ...]
    offsets: dict[str, int]
    components: dict[str, dict]
    assembled_package: dict


def load_fidelity_inputs(root: Path) -> FidelityInputs:
    build = root / "build/r1c_cp4_r1"
    with np.load(build / "compiled_materialization.npz", allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    offsets_payload = _read_json(build / "component_offsets.json")
    product = _read_json(build / "product/materialized_product.json")
    components = {item["instance_id"]: item for item in product["components"]}
    package_path = root / "build/r1c_cp3/repaired/repaired_assembled_pattern_package.json"
    return FidelityInputs(
        arrays=arrays,
        component_order=tuple(offsets_payload["component_order"]),
        offsets={key: int(value) for key, value in offsets_payload["offsets"].items()},
        components=components,
        assembled_package=_read_json(package_path),
    )


def classify_edges(inputs: FidelityInputs, positions: np.ndarray) -> tuple[list[dict], dict]:
    edges = inputs.arrays["edges"].astype(np.int32)
    pattern = inputs.arrays["pattern_rest_lengths"].astype(np.float64)
    expected = expected_pattern_lengths(inputs)
    arrangement = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    records = []
    for index, edge in enumerate(edges):
        component = inputs.component_order[int(inputs.arrays["component_ids"][int(edge[0])])]
        local = edge - inputs.offsets[component]
        region, feature, boundary = classify_local_edge(inputs.components[component], local)
        ratio = float(arrangement[index] / max(expected[index], 1.0e-12))
        symmetric = max(ratio, 1.0 / max(ratio, 1.0e-12))
        records.append({
            "edge_index": index,
            "vertices": [int(edge[0]), int(edge[1])],
            "component": component,
            "region": region,
            "feature": feature,
            "boundary": boundary,
            "target_component": component in TARGET_COMPONENTS,
            "pattern_length_m": float(pattern[index]),
            "expected_length_m": float(expected[index]),
            "arrangement_length_m": float(arrangement[index]),
            "raw_ratio": ratio,
            "symmetric_ratio": symmetric,
            "log_isometry_error": abs(float(np.log(max(ratio, 1.0e-12)))),
        })
    return records, summarize_records(records)


def expected_pattern_lengths(inputs: FidelityInputs) -> np.ndarray:
    edges = inputs.arrays["edges"].astype(np.int32)
    expected = inputs.arrays["pattern_rest_lengths"].astype(np.float64).copy()
    ease = _ease_by_component_boundary(inputs.assembled_package)
    for index, edge in enumerate(edges):
        component = inputs.component_order[int(inputs.arrays["component_ids"][int(edge[0])])]
        local = edge - inputs.offsets[component]
        _, _, boundary = classify_local_edge(inputs.components[component], local)
        expected[index] /= ease.get((component, boundary), 1.0)
    return expected


def summarize_records(records: list[dict]) -> dict:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        key = (record["component"], record["region"], record["feature"])
        groups.setdefault(key, []).append(record)
    summaries = [_group_summary(key, items) for key, items in sorted(groups.items())]
    target = [item for item in records if item["target_component"]]
    top = sorted(records, key=lambda item: item["log_isometry_error"], reverse=True)[:100]
    return {
        "contract": "ComponentBoundaryMetricDecomposition/1",
        "edge_count": len(records),
        "target_edge_count": len(target),
        "global": _metric_summary(records),
        "target_components": _metric_summary(target),
        "groups": summaries,
        "top_distorted_edges": top,
    }


def classify_local_edge(component: dict, local_edge: np.ndarray) -> tuple[str, str, str]:
    first, second = map(int, local_edge)
    shared, touched = [], []
    for boundary_id, values in component["boundaries"].items():
        indices = set(map(int, values))
        if first in indices or second in indices:
            touched.append(boundary_id)
        if first in indices and second in indices:
            shared.append(boundary_id)
    if shared:
        boundary = sorted(shared)[0]
        return "BOUNDARY", _feature(component["instance_id"], boundary), boundary
    if touched:
        boundary = sorted(touched)[0]
        return "SEAM_ADJACENT_INTERIOR", _feature(component["instance_id"], boundary), boundary
    return "INTERIOR", _interior_feature(component, first, second), ""


def _group_summary(key: tuple[str, str, str], items: list[dict]) -> dict:
    payload = _metric_summary(items)
    payload.update({"component": key[0], "region": key[1], "feature": key[2]})
    return payload


def _metric_summary(items: list[dict]) -> dict:
    if not items:
        return {"edge_count": 0}
    raw = np.asarray([item["raw_ratio"] for item in items], dtype=np.float64)
    symmetric = np.asarray([item["symmetric_ratio"] for item in items], dtype=np.float64)
    log_error = np.asarray([item["log_isometry_error"] for item in items], dtype=np.float64)
    return {
        "edge_count": len(items),
        "raw_ratio_p50": float(np.quantile(raw, 0.50)),
        "raw_ratio_p95": float(np.quantile(raw, 0.95)),
        "raw_ratio_p99": float(np.quantile(raw, 0.99)),
        "symmetric_ratio_p50": float(np.quantile(symmetric, 0.50)),
        "symmetric_ratio_p95": float(np.quantile(symmetric, 0.95)),
        "symmetric_ratio_p99": float(np.quantile(symmetric, 0.99)),
        "symmetric_ratio_max": float(np.max(symmetric)),
        "log_error_p50": float(np.quantile(log_error, 0.50)),
        "log_error_p95": float(np.quantile(log_error, 0.95)),
        "log_error_p99": float(np.quantile(log_error, 0.99)),
    }


def _ease_by_component_boundary(package: dict) -> dict[tuple[str, str], float]:
    result = {}
    for seam in package["seams"]:
        endpoint = seam["endpoint_a"]
        component = endpoint["component_instance_id"]
        boundary = endpoint["boundary_id"]
        if component.startswith("sleeve") and boundary in {"cap_front", "cap_back"}:
            result[(component, boundary)] = max(float(seam["length_ratio"]), 1.0)
    return result


def _feature(instance_id: str, boundary: str) -> str:
    if instance_id.startswith("sleeve") and boundary in {"cap_front", "cap_back"}:
        return f"SLEEVE_{boundary.upper()}"
    if instance_id.startswith("sleeve") and boundary.startswith("underarm"):
        return "SLEEVE_UNDERARM"
    if instance_id.startswith("sleeve") and boundary == "wrist":
        return "SLEEVE_WRIST"
    if instance_id == "collar":
        return "COLLAR_ATTACH" if boundary.endswith("attach") else "COLLAR_OUTER"
    if instance_id.startswith("cuff"):
        if boundary == "sleeve_attach":
            return "CUFF_ATTACH"
        if boundary.startswith("end_"):
            return "CUFF_END"
        return "CUFF_OUTER"
    return "ORDINARY_BOUNDARY"


def _interior_feature(component: dict, first: int, second: int) -> str:
    if component["instance_id"].startswith("cuff"):
        positions = np.asarray(component["positions_m"], dtype=np.float64)
        y = 0.5 * (positions[first, 1] + positions[second, 1])
        middle = 0.5 * (positions[:, 1].min() + positions[:, 1].max())
        if abs(y - middle) <= 0.012:
            return "CUFF_FOLD_NEIGHBORHOOD"
    return "ORDINARY_INTERIOR"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
