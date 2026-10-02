"""Bounded isometric repair over sleeve, collar, and cuff arrangement owners."""
from __future__ import annotations

import numpy as np

from .classify import FidelityInputs, TARGET_COMPONENTS, expected_pattern_lengths


Node = int


def repair_arrangement(inputs: FidelityInputs, iterations: int = 240) -> tuple[np.ndarray, dict]:
    baseline = inputs.arrays["positions_rest"].astype(np.float64)
    positions = baseline.copy()
    edges = inputs.arrays["edges"].astype(np.int32)
    target_lengths = expected_pattern_lengths(inputs)
    target_edges = _target_edge_indices(inputs, edges)
    mobility = _mobility(inputs)
    seam_classes = _target_seam_classes(inputs)
    before = _edge_error(positions, edges[target_edges], target_lengths[target_edges])
    for _ in range(iterations):
        _project_edge_lengths(positions, edges, target_lengths, target_edges, mobility)
        _project_seam_classes(positions, seam_classes, mobility)
        _anchor_positions(positions, baseline, mobility)
    _project_seam_classes(positions, seam_classes, mobility)
    after = _edge_error(positions, edges[target_edges], target_lengths[target_edges])
    displacement = np.linalg.norm(positions - baseline, axis=1)
    receipt = {
        "contract": "IsometricArrangementRepairReceipt/1",
        "owner_components": sorted(TARGET_COMPONENTS),
        "iterations": iterations,
        "target_edge_count": int(len(target_edges)),
        "movable_vertex_count": int(np.count_nonzero(mobility > 0.0)),
        "seam_class_count": len(seam_classes),
        "before": before,
        "after": after,
        "maximum_displacement_m": float(displacement.max()),
        "p95_displacement_m": float(np.quantile(displacement, 0.95)),
        "mean_displacement_m": float(displacement.mean()),
        "source_pattern_changed": False,
        "post_settle_vertex_repair_count": 0,
    }
    return positions, receipt


def _target_edge_indices(inputs: FidelityInputs, edges: np.ndarray) -> np.ndarray:
    component_ids = inputs.arrays["component_ids"]
    indices = []
    for edge_index, edge in enumerate(edges):
        component = inputs.component_order[int(component_ids[int(edge[0])])]
        if component in TARGET_COMPONENTS:
            indices.append(edge_index)
    return np.asarray(indices, dtype=np.int32)


def _mobility(inputs: FidelityInputs) -> np.ndarray:
    component_ids = inputs.arrays["component_ids"]
    mobility = np.zeros(len(component_ids), dtype=np.float64)
    for component_index, component in enumerate(inputs.component_order):
        if component in TARGET_COMPONENTS:
            mobility[component_ids == component_index] = 1.0
    for first, second in inputs.arrays["seam_pairs"].astype(np.int32):
        if mobility[first] > 0.0 and mobility[second] == 0.0:
            mobility[second] = 0.28
        elif mobility[second] > 0.0 and mobility[first] == 0.0:
            mobility[first] = 0.28
    return mobility


def _target_seam_classes(inputs: FidelityInputs) -> list[np.ndarray]:
    parent: dict[Node, Node] = {}
    component_ids = inputs.arrays["component_ids"]
    target_ids = {
        index for index, name in enumerate(inputs.component_order) if name in TARGET_COMPONENTS
    }
    for first, second in inputs.arrays["seam_pairs"].astype(np.int32):
        if int(component_ids[first]) not in target_ids and int(component_ids[second]) not in target_ids:
            continue
        _union(parent, int(first), int(second))
    groups: dict[int, list[int]] = {}
    for node in parent:
        groups.setdefault(_find(parent, node), []).append(node)
    return [np.asarray(sorted(values), dtype=np.int32) for _, values in sorted(groups.items())]


def _project_edge_lengths(
    positions: np.ndarray,
    edges: np.ndarray,
    targets: np.ndarray,
    edge_indices: np.ndarray,
    mobility: np.ndarray,
) -> None:
    for edge_index in edge_indices:
        first, second = map(int, edges[int(edge_index)])
        vector = positions[second] - positions[first]
        length = float(np.linalg.norm(vector))
        if length <= 1.0e-12:
            continue
        first_weight = mobility[first]
        second_weight = mobility[second]
        total = first_weight + second_weight
        if total <= 0.0:
            continue
        relative_error = (length - targets[int(edge_index)]) / length
        strength = min(0.42, 0.20 + 0.28 * abs(relative_error))
        correction = vector * relative_error * strength
        positions[first] += correction * (first_weight / total)
        positions[second] -= correction * (second_weight / total)


def _project_seam_classes(positions: np.ndarray, classes: list[np.ndarray], mobility: np.ndarray) -> None:
    for indices in classes:
        weights = np.maximum(mobility[indices], 0.05)
        target = np.average(positions[indices], axis=0, weights=weights)
        positions[indices] = target


def _anchor_positions(positions: np.ndarray, baseline: np.ndarray, mobility: np.ndarray) -> None:
    active = mobility > 0.0
    target_strength = np.where(mobility >= 0.9, 0.018, 0.055)
    positions[active] += (baseline[active] - positions[active]) * target_strength[active, None]


def _edge_error(positions: np.ndarray, edges: np.ndarray, target: np.ndarray) -> dict:
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    ratio = lengths / np.maximum(target, 1.0e-12)
    symmetric = np.maximum(ratio, 1.0 / np.maximum(ratio, 1.0e-12))
    log_error = np.abs(np.log(np.maximum(ratio, 1.0e-12)))
    return {
        "symmetric_ratio_p50": float(np.quantile(symmetric, 0.50)),
        "symmetric_ratio_p95": float(np.quantile(symmetric, 0.95)),
        "symmetric_ratio_p99": float(np.quantile(symmetric, 0.99)),
        "log_error_p95": float(np.quantile(log_error, 0.95)),
        "log_error_p99": float(np.quantile(log_error, 0.99)),
    }


def _find(parent: dict[Node, Node], node: Node) -> Node:
    parent.setdefault(node, node)
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent: dict[Node, Node], first: Node, second: Node) -> None:
    root_first = _find(parent, first)
    root_second = _find(parent, second)
    if root_first == root_second:
        return
    lower, upper = sorted((root_first, root_second))
    parent[upper] = lower
