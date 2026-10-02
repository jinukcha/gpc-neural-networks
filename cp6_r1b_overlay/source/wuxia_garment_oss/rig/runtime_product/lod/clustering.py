"""Deterministic vertex clustering that preserves skin and morph attributes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wuxia_garment_oss.rig.game_product.glb_skin import _accessor_array, _append_accessor


@dataclass(frozen=True)
class ClusterResult:
    representatives: np.ndarray
    inverse: np.ndarray
    resolution: int


def _normal_bits(normals: np.ndarray | None, count: int) -> np.ndarray:
    if normals is None:
        return np.zeros((count, 3), dtype=np.int16)
    return np.signbit(normals.astype(np.float64)).astype(np.int16)


def _keys(positions: np.ndarray, normals: np.ndarray | None, resolution: int) -> np.ndarray:
    values = positions.astype(np.float64)
    low = np.min(values, axis=0)
    extent = np.max(values, axis=0) - low
    safe = np.where(extent > 1.0e-9, extent, 1.0)
    quantized = np.floor(np.clip((values - low) / safe, 0.0, 1.0) * resolution).astype(np.int32)
    return np.concatenate((quantized, _normal_bits(normals, len(values))), axis=1)


def _cluster_for_resolution(
    positions: np.ndarray,
    normals: np.ndarray | None,
    resolution: int,
) -> ClusterResult:
    keys = _keys(positions, normals, resolution)
    _, representatives, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(representatives)
    representatives = representatives[order]
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order), dtype=np.int64)
    return ClusterResult(representatives.astype(np.int64), remap[inverse], resolution)


def choose_clusters(
    positions: np.ndarray,
    normals: np.ndarray | None,
    target_vertex_count: int,
) -> ClusterResult:
    count = len(positions)
    if target_vertex_count >= count:
        identity = np.arange(count, dtype=np.int64)
        return ClusterResult(identity, identity, 0)
    low, high = 1, 512
    best: ClusterResult | None = None
    best_error = count
    for _ in range(12):
        resolution = (low + high) // 2
        candidate = _cluster_for_resolution(positions, normals, resolution)
        error = abs(len(candidate.representatives) - target_vertex_count)
        if error < best_error:
            best, best_error = candidate, error
        if len(candidate.representatives) < target_vertex_count:
            low = resolution + 1
        else:
            high = resolution - 1
        if low > high:
            break
    if best is None:
        raise RuntimeError("unable to choose a clustering resolution")
    return best


def _source_indices(glb, primitive: dict, vertex_count: int) -> np.ndarray:
    accessor_index = primitive.get("indices")
    if accessor_index is None:
        return np.arange(vertex_count, dtype=np.int64)
    return _accessor_array(glb, accessor_index).reshape(-1).astype(np.int64)


def _valid_triangles(indices: np.ndarray, inverse: np.ndarray, positions: np.ndarray) -> np.ndarray:
    triangles = indices[: len(indices) // 3 * 3].reshape(-1, 3)
    mapped = inverse[triangles]
    distinct = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 0] != mapped[:, 2])
    mapped = mapped[distinct]
    if not len(mapped):
        return mapped.astype(np.uint32)
    first = positions[mapped[:, 1]] - positions[mapped[:, 0]]
    second = positions[mapped[:, 2]] - positions[mapped[:, 0]]
    area = np.linalg.norm(np.cross(first, second), axis=1)
    mapped = mapped[area > 1.0e-12]
    if not len(mapped):
        return mapped.astype(np.uint32)
    canonical = np.sort(mapped, axis=1)
    _, unique_indices = np.unique(canonical, axis=0, return_index=True)
    return mapped[np.sort(unique_indices)].astype(np.uint32)


def _append_like(glb, source_accessor: int, values: np.ndarray, target: int | None) -> int:
    source = glb.document["accessors"][source_accessor]
    result = _append_accessor(
        glb,
        values,
        int(source["componentType"]),
        str(source["type"]),
        target,
        str(source["type"]) == "VEC3" and target == 34962,
    )
    if source.get("normalized"):
        glb.document["accessors"][result]["normalized"] = True
    return result


def _cluster_attributes(glb, primitive: dict, representatives: np.ndarray) -> dict:
    attributes = {}
    for semantic, accessor_index in primitive.get("attributes", {}).items():
        values = _accessor_array(glb, accessor_index)[representatives]
        attributes[semantic] = _append_like(glb, accessor_index, values, 34962)
    return attributes


def _cluster_targets(glb, primitive: dict, representatives: np.ndarray) -> list[dict]:
    targets = []
    for source_target in primitive.get("targets", []):
        target = {}
        for semantic, accessor_index in source_target.items():
            values = _accessor_array(glb, accessor_index)[representatives]
            target[semantic] = _append_like(glb, accessor_index, values, 34962)
        targets.append(target)
    return targets


def cluster_primitive(glb, primitive: dict, ratio: float) -> tuple[dict, dict]:
    position_accessor = primitive.get("attributes", {}).get("POSITION")
    if position_accessor is None or int(primitive.get("mode", 4)) != 4:
        return dict(primitive), {"changed": False, "reason": "NON_TRIANGLE_OR_NO_POSITION"}
    positions = _accessor_array(glb, position_accessor).astype(np.float64)
    if len(positions) < 64 or ratio >= 0.999:
        return dict(primitive), {"changed": False, "reason": "BELOW_CLUSTER_THRESHOLD"}
    normal_accessor = primitive.get("attributes", {}).get("NORMAL")
    normals = _accessor_array(glb, normal_accessor) if normal_accessor is not None else None
    target_count = max(32, int(round(len(positions) * ratio)))
    clusters = choose_clusters(positions, normals, target_count)
    reduced_positions = positions[clusters.representatives]
    indices = _source_indices(glb, primitive, len(positions))
    triangles = _valid_triangles(indices, clusters.inverse, reduced_positions)
    if len(triangles) < 4:
        return dict(primitive), {"changed": False, "reason": "CLUSTER_COLLAPSE_GUARD"}
    result = {key: value for key, value in primitive.items() if key not in {"attributes", "indices", "targets"}}
    result["attributes"] = _cluster_attributes(glb, primitive, clusters.representatives)
    result["indices"] = _append_accessor(glb, triangles.reshape(-1), 5125, "SCALAR", 34963)
    if primitive.get("targets"):
        result["targets"] = _cluster_targets(glb, primitive, clusters.representatives)
    receipt = {
        "changed": True,
        "source_vertices": int(len(positions)),
        "target_vertices": int(len(reduced_positions)),
        "source_triangles": int(len(indices) // 3),
        "target_triangles": int(len(triangles)),
        "resolution": int(clusters.resolution),
        "vertex_ratio": float(len(reduced_positions) / len(positions)),
        "triangle_ratio": float(len(triangles) / max(len(indices) // 3, 1)),
    }
    return result, receipt
