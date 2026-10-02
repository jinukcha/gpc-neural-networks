"""Orientation-aware seam mapping and cap-patch arrangement relaxation."""
from __future__ import annotations

import numpy as np

from .model import ComponentMesh, SeamMap


Node = tuple[str, int]


def compile_seam_maps(meshes: dict[str, ComponentMesh], package: dict) -> tuple[list[SeamMap], dict]:
    maps: list[SeamMap] = []
    for seam in package["seams"]:
        endpoint_a = seam["endpoint_a"]
        endpoint_b = seam["endpoint_b"]
        mesh_a = meshes[endpoint_a["component_instance_id"]]
        mesh_b = meshes[endpoint_b["component_instance_id"]]
        indices_a = mesh_a.boundary_indices[endpoint_a["boundary_id"]]
        indices_b = mesh_b.boundary_indices[endpoint_b["boundary_id"]]
        if len(indices_a) != len(indices_b):
            raise ValueError(f"seam sample-count mismatch: {seam['interface_id']}")
        direct = _distances(mesh_a.vertices_3d[indices_a], mesh_b.vertices_3d[indices_b])
        reverse = _distances(mesh_a.vertices_3d[indices_a], mesh_b.vertices_3d[indices_b[::-1]])
        orientation = "DIRECT" if float(direct.mean()) <= float(reverse.mean()) else "REVERSED"
        paired_b = indices_b if orientation == "DIRECT" else indices_b[::-1]
        chosen = direct if orientation == "DIRECT" else reverse
        maps.append(_make_seam_map(seam, mesh_a, mesh_b, indices_a, paired_b, direct, reverse, chosen, orientation))
    receipt = {
        "contract": "SeamCorrespondenceReceipt/1",
        "interface_count": len(maps),
        "direct_count": sum(item.orientation == "DIRECT" for item in maps),
        "reversed_count": sum(item.orientation == "REVERSED" for item in maps),
        "maximum_initial_p95_m": max(item.initial_p95_m for item in maps),
        "interfaces": [item.to_dict() for item in maps],
    }
    return maps, receipt


def _make_seam_map(seam, mesh_a, mesh_b, indices_a, paired_b, direct, reverse, chosen, orientation):
    return SeamMap(
        interface_id=seam["interface_id"],
        component_a=mesh_a.instance_id,
        boundary_a=seam["endpoint_a"]["boundary_id"],
        component_b=mesh_b.instance_id,
        boundary_b=seam["endpoint_b"]["boundary_id"],
        orientation=orientation,
        vertex_pairs=np.column_stack((indices_a, paired_b)).astype(np.int32),
        direct_mean_m=float(direct.mean()),
        reversed_mean_m=float(reverse.mean()),
        initial_mean_m=float(chosen.mean()),
        initial_p95_m=float(np.quantile(chosen, 0.95)),
    )


def align_and_relax(meshes: dict[str, ComponentMesh], seam_maps: list[SeamMap]) -> dict:
    classes = _seam_equivalence_classes(seam_maps)
    for members in classes:
        target = np.mean([meshes[owner].vertices_3d[index] for owner, index in members], axis=0)
        for owner, index in members:
            meshes[owner].vertices_3d[index] = target
            meshes[owner].fixed_mask[index] = True
    for mesh in meshes.values():
        _laplacian_cap_relax(mesh)
    post = seam_gap_metrics(meshes, seam_maps)
    return {
        "contract": "CapPatchArrangementReceipt/1",
        "source_pattern_changed": False,
        "compiled_rest_metric_pending": True,
        "seam_equivalence_class_count": len(classes),
        "maximum_seam_junction_size": max(len(item) for item in classes),
        "post_alignment_seam_mean_m": post["mean_m"],
        "post_alignment_seam_p95_m": post["p95_m"],
        "post_alignment_seam_max_m": post["max_m"],
        "component_relaxation": "BOUNDARY_FIXED_LAPLACIAN_WITH_INITIAL_SHAPE_REGULARIZATION",
    }


def _seam_equivalence_classes(seam_maps: list[SeamMap]) -> list[list[Node]]:
    parent: dict[Node, Node] = {}
    for seam in seam_maps:
        for index_a, index_b in seam.vertex_pairs:
            first = (seam.component_a, int(index_a))
            second = (seam.component_b, int(index_b))
            _union(parent, first, second)
    grouped: dict[Node, list[Node]] = {}
    for node in parent:
        grouped.setdefault(_find(parent, node), []).append(node)
    return [sorted(members) for _, members in sorted(grouped.items())]


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


def seam_gap_metrics(meshes: dict[str, ComponentMesh], seam_maps: list[SeamMap]) -> dict:
    values = []
    by_interface = []
    for seam in seam_maps:
        mesh_a = meshes[seam.component_a]
        mesh_b = meshes[seam.component_b]
        points_a = mesh_a.vertices_3d[seam.vertex_pairs[:, 0]]
        points_b = mesh_b.vertices_3d[seam.vertex_pairs[:, 1]]
        distances = _distances(points_a, points_b)
        values.extend(distances.tolist())
        by_interface.append({
            "interface_id": seam.interface_id,
            "mean_m": float(distances.mean()),
            "p95_m": float(np.quantile(distances, 0.95)),
            "max_m": float(distances.max()),
        })
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_m": float(array.mean()),
        "p95_m": float(np.quantile(array, 0.95)),
        "max_m": float(array.max()),
        "interfaces": by_interface,
    }


def _laplacian_cap_relax(mesh: ComponentMesh, iterations: int = 70) -> None:
    adjacency = _adjacency(len(mesh.vertices_3d), mesh.triangles)
    initial = mesh.vertices_3d.copy()
    current = mesh.vertices_3d.copy()
    fixed = mesh.fixed_mask
    for _ in range(iterations):
        updated = current.copy()
        for index, neighbours in enumerate(adjacency):
            if fixed[index] or not neighbours:
                continue
            average = current[np.fromiter(neighbours, dtype=np.int32)].mean(axis=0)
            updated[index] = 0.62 * average + 0.38 * initial[index]
        current = updated
    mesh.vertices_3d = current
    mesh.metadata["cap_patch_relax_iterations"] = iterations


def _adjacency(vertex_count: int, triangles: np.ndarray) -> list[set[int]]:
    result = [set() for _ in range(vertex_count)]
    for triangle in triangles:
        a, b, c = map(int, triangle)
        result[a].update((b, c))
        result[b].update((a, c))
        result[c].update((a, b))
    return result


def _distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(left - right, axis=1)
