"""Deterministic equivalence classes for shared sewn vertices."""
from __future__ import annotations

from .model import SeamMap


Node = tuple[str, int]


def seam_groups(seam_maps: list[SeamMap]) -> list[list[Node]]:
    parent: dict[Node, Node] = {}
    for seam in seam_maps:
        for index_a, index_b in seam.vertex_pairs:
            union(parent, (seam.component_a, int(index_a)), (seam.component_b, int(index_b)))
    grouped: dict[Node, list[Node]] = {}
    for node in parent:
        grouped.setdefault(find(parent, node), []).append(node)
    return [sorted(members) for _, members in sorted(grouped.items())]


def find(parent: dict[Node, Node], node: Node) -> Node:
    parent.setdefault(node, node)
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def union(parent: dict[Node, Node], first: Node, second: Node) -> None:
    root_first = find(parent, first)
    root_second = find(parent, second)
    if root_first == root_second:
        return
    lower, upper = sorted((root_first, root_second))
    parent[upper] = lower
