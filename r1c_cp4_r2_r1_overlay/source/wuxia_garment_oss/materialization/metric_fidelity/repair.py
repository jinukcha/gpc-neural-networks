"""Bounded component-local metric repair with hard seam preservation."""
from __future__ import annotations

import numpy as np

from ..body import BodyProfile
from ..model import canonical_sha256
from ..settle import _project_body_clearance


_TARGET_PREFIXES = ("sleeve_", "cuff_")


def _selected_components(order: list[str]) -> set[int]:
    return {
        index for index, name in enumerate(order)
        if name.startswith(_TARGET_PREFIXES) or name == "collar"
    }


def _protected_vertices(arrays: dict) -> np.ndarray:
    protected = np.zeros(len(arrays["positions_rest"]), dtype=np.bool_)
    if len(arrays["seam_pairs"]):
        protected[np.unique(arrays["seam_pairs"].reshape(-1))] = True
    return protected


def _edge_mask(arrays: dict, selected: set[int]) -> np.ndarray:
    first = arrays["edges"][:, 0]
    second = arrays["edges"][:, 1]
    owners = arrays["component_ids"]
    return np.isin(owners[first], list(selected)) & (owners[first] == owners[second])


def _iteration(positions, baseline, arrays, edge_mask, movable, relaxation, anchor):
    edges = arrays["edges"][edge_mask]
    target = arrays["pattern_rest_lengths"][edge_mask]
    first, second = edges[:, 0], edges[:, 1]
    delta = positions[second] - positions[first]
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1.0e-10
    correction = np.zeros_like(delta)
    correction[valid] = relaxation * ((length[valid] - target[valid]) / length[valid])[:, None] * delta[valid]
    accumulation = np.zeros_like(positions)
    counts = np.zeros(len(positions), dtype=np.float64)
    first_move, second_move = movable[first], movable[second]
    both = first_move & second_move
    np.add.at(accumulation, first[both], 0.5 * correction[both])
    np.add.at(accumulation, second[both], -0.5 * correction[both])
    np.add.at(counts, first[both], 1.0)
    np.add.at(counts, second[both], 1.0)
    only_first = first_move & ~second_move
    only_second = second_move & ~first_move
    np.add.at(accumulation, first[only_first], correction[only_first])
    np.add.at(accumulation, second[only_second], -correction[only_second])
    np.add.at(counts, first[only_first], 1.0)
    np.add.at(counts, second[only_second], 1.0)
    active = movable & (counts > 0.0)
    positions[active] -= accumulation[active] / counts[active, None]
    positions[movable] += anchor * (baseline[movable] - positions[movable])


def _clamp_displacement(positions, baseline, movable, maximum):
    displacement = positions - baseline
    length = np.linalg.norm(displacement, axis=1)
    over = movable & (length > maximum)
    positions[over] = baseline[over] + displacement[over] / length[over, None] * maximum


def _candidate(arrays: dict, profile: BodyProfile, iterations: int, relaxation: float, anchor: float):
    positions = arrays["positions_rest"].astype(np.float64).copy()
    baseline = positions.copy()
    selected = _selected_components(list(arrays["component_order"]))
    protected = _protected_vertices(arrays)
    movable = np.isin(arrays["component_ids"], list(selected)) & ~protected
    edge_mask = _edge_mask(arrays, selected)
    for _ in range(iterations):
        _iteration(positions, baseline, arrays, edge_mask, movable, relaxation, anchor)
        _clamp_displacement(positions, baseline, movable, 0.018)
    contact_arrays = dict(arrays)
    contact_arrays["fixed_mask"] = protected
    positions, contact_count = _project_body_clearance(positions, contact_arrays, profile, 0.0025)
    positions[protected] = baseline[protected]
    return positions, movable, edge_mask, contact_count


def _metrics(arrays: dict, positions: np.ndarray, edge_mask: np.ndarray) -> dict:
    edges = arrays["edges"]
    length = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    pattern = arrays["pattern_rest_lengths"]
    valid = pattern > 1.0e-8
    ratio = length[valid] / pattern[valid]
    selected_ratio = length[edge_mask] / np.maximum(pattern[edge_mask], 1.0e-12)
    return {
        "lengths": length,
        "global_p95": float(np.quantile(ratio, 0.95)),
        "global_p99": float(np.quantile(ratio, 0.99)),
        "selected_p95": float(np.quantile(selected_ratio, 0.95)),
        "selected_p99": float(np.quantile(selected_ratio, 0.99)),
    }


def _score(metrics: dict, baseline: dict) -> float:
    return sum(metrics[key] / baseline[key] for key in ("global_p95", "global_p99", "selected_p95", "selected_p99"))


def repair_arrangement(arrays: dict, profile: BodyProfile) -> tuple[np.ndarray, dict, np.ndarray]:
    selected = _selected_components(list(arrays["component_order"]))
    edge_mask = _edge_mask(arrays, selected)
    baseline = _metrics(arrays, arrays["positions_rest"], edge_mask)
    candidates = []
    for iterations, relaxation, anchor in ((240, 0.16, 0.018), (360, 0.14, 0.012), (480, 0.11, 0.008)):
        result = _candidate(arrays, profile, iterations, relaxation, anchor)
        positions, movable, mask, contact = result
        metrics = _metrics(arrays, positions, mask)
        candidates.append((_score(metrics, baseline), positions, movable, metrics, contact, iterations, relaxation, anchor))
    _, positions, movable, after, contact, iterations, relaxation, anchor = min(candidates, key=lambda item: item[0])
    displacement = np.linalg.norm(positions - arrays["positions_rest"], axis=1)
    improved = after["global_p95"] < baseline["global_p95"] and after["global_p99"] < baseline["global_p99"]
    receipt = {
        "contract": "IsometricArrangementRepairReceipt/1",
        "target_components": [arrays["component_order"][index] for index in sorted(selected)],
        "movable_vertex_count": int(np.count_nonzero(movable)),
        "protected_seam_vertex_count": int(np.count_nonzero(_protected_vertices(arrays))),
        "iterations": iterations,
        "relaxation": relaxation,
        "anchor": anchor,
        "maximum_displacement_m": float(displacement.max()),
        "p95_displacement_m": float(np.quantile(displacement[movable], 0.95)),
        "body_clearance_projection_count": int(contact),
        "before": {key: value for key, value in baseline.items() if key != "lengths"},
        "after": {key: value for key, value in after.items() if key != "lengths"},
        "global_metric_improved": improved,
        "source_pattern_changed": False,
        "bodice_arrangement_changed": False,
        "post_repair_vertex_edit_count": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return positions, receipt, after["lengths"]
