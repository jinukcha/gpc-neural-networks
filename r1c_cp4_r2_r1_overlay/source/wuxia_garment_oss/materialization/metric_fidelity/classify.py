"""Classify structural edges by component and boundary ownership."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from ..model import canonical_sha256


def _boundary_sets(product: dict, offsets: dict[str, int]) -> dict[int, tuple[str, ...]]:
    ownership: dict[int, set[str]] = defaultdict(set)
    for component in product["components"]:
        offset = int(offsets[component["instance_id"]])
        for boundary_id, indices in component["boundaries"].items():
            for local_index in indices:
                ownership[offset + int(local_index)].add(boundary_id)
    return {index: tuple(sorted(values)) for index, values in ownership.items()}


def _edge_region(first: int, second: int, boundaries: dict[int, tuple[str, ...]]) -> str:
    left = set(boundaries.get(first, ()))
    right = set(boundaries.get(second, ()))
    shared = sorted(left & right)
    if shared:
        return "BOUNDARY:" + "+".join(shared)
    adjacent = sorted(left | right)
    if adjacent:
        return "BOUNDARY_ADJACENT:" + "+".join(adjacent)
    return "INTERIOR"


def _summary(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "count": int(len(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def _record(index: int, edge: np.ndarray, arrays: dict, order: list[str], boundaries: dict) -> dict:
    first, second = map(int, edge)
    component_index = int(arrays["component_ids"][first])
    component_id = order[component_index]
    pattern = float(arrays["pattern_rest_lengths"][index])
    arrangement = float(arrays["arrangement_rest_lengths"][index])
    ratio = arrangement / max(pattern, 1.0e-12)
    symmetric = max(ratio, 1.0 / max(ratio, 1.0e-12))
    return {
        "edge_index": index,
        "vertices": [first, second],
        "component_id": component_id,
        "region": _edge_region(first, second, boundaries),
        "pattern_length_m": pattern,
        "arrangement_length_m": arrangement,
        "arrangement_over_pattern": ratio,
        "symmetric_distortion": symmetric,
        "absolute_delta_m": abs(arrangement - pattern),
        "absolute_log_ratio": abs(math.log(max(ratio, 1.0e-12))),
    }


def _group_records(records: list[dict], key) -> dict[str, dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    raw: dict[str, list[float]] = defaultdict(list)
    for item in records:
        name = key(item)
        grouped[name].append(item["symmetric_distortion"])
        raw[name].append(item["arrangement_over_pattern"])
    return {
        name: {
            "symmetric_distortion": _summary(np.asarray(grouped[name], dtype=np.float64)),
            "arrangement_over_pattern": _summary(np.asarray(raw[name], dtype=np.float64)),
        }
        for name in sorted(grouped)
    }


def decompose_metric(arrays: dict, product: dict, offsets: dict[str, int], stage: str) -> dict:
    order = list(arrays["component_order"])
    boundaries = _boundary_sets(product, offsets)
    records = [
        _record(index, edge, arrays, order, boundaries)
        for index, edge in enumerate(arrays["edges"])
        if arrays["pattern_rest_lengths"][index] > 1.0e-8
    ]
    raw = np.asarray([item["arrangement_over_pattern"] for item in records], dtype=np.float64)
    symmetric = np.asarray([item["symmetric_distortion"] for item in records], dtype=np.float64)
    top = sorted(records, key=lambda item: (-item["symmetric_distortion"], item["edge_index"]))[:96]
    payload = {
        "contract": "PatternArrangementMetricDecomposition/1",
        "stage": stage,
        "edge_count": len(records),
        "global": {
            "arrangement_over_pattern": _summary(raw),
            "symmetric_distortion": _summary(symmetric),
        },
        "by_component": _group_records(records, lambda item: item["component_id"]),
        "by_component_region": _group_records(
            records, lambda item: f"{item['component_id']}::{item['region']}"
        ),
        "top_distortion_edges": top,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
