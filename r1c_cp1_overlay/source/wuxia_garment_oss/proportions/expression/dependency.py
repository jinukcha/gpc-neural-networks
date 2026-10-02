"""Deterministic dependency graph and cycle rejection."""
from __future__ import annotations

from collections import defaultdict

from .parser import parse_expression, referenced_paths
from ..model.parameter import ParameterDefinition


def parameter_dependencies(definition: ParameterDefinition) -> tuple[str, ...]:
    if definition.mode != "AUTO_DERIVED":
        return ()
    paths = referenced_paths(parse_expression(definition.expression or ""))
    return tuple(sorted(path.split(".", 1)[1] for path in paths if path.startswith("param.")))


def build_dependency_graph(
    definitions: tuple[ParameterDefinition, ...],
) -> dict[str, tuple[str, ...]]:
    ids = {item.parameter_id for item in definitions}
    graph = {}
    for item in definitions:
        dependencies = parameter_dependencies(item)
        missing = sorted(set(dependencies) - ids)
        if missing:
            raise ValueError(f"missing parameter dependencies for {item.parameter_id}: {missing}")
        graph[item.parameter_id] = dependencies
    return graph


def _cycle_path(graph: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> tuple[str, ...] | None:
        marker = state.get(node, 0)
        if marker == 2:
            return None
        if marker == 1:
            index = stack.index(node)
            return tuple(stack[index:] + [node])
        state[node] = 1
        stack.append(node)
        for dependency in graph[node]:
            cycle = visit(dependency)
            if cycle:
                return cycle
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def topological_order(graph: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    cycle = _cycle_path(graph)
    if cycle:
        raise ValueError("CYCLE_DETECTED:" + " -> ".join(cycle))
    reverse: dict[str, list[str]] = defaultdict(list)
    indegree = {node: len(dependencies) for node, dependencies in graph.items()}
    for node, dependencies in graph.items():
        for dependency in dependencies:
            reverse[dependency].append(node)
    ready = sorted(node for node, count in indegree.items() if count == 0)
    result = []
    while ready:
        node = ready.pop(0)
        result.append(node)
        for dependent in sorted(reverse[node]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    if len(result) != len(graph):
        raise ValueError("dependency ordering incomplete")
    return tuple(result)
