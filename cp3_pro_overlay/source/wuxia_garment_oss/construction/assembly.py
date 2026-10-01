"""Assembly DAG validation and deterministic topological planning."""
from __future__ import annotations

from .model import ConstructionGraph


def topological_order(graph: ConstructionGraph) -> list[str]:
    graph.validate()
    dependencies = {
        item.operation_id: set(item.depends_on)
        for item in graph.operations
    }
    remaining = set(dependencies)
    result: list[str] = []
    while remaining:
        ready = sorted(item for item in remaining if not (dependencies[item] & remaining))
        if not ready:
            cycle_nodes = sorted(remaining)
            raise ValueError(f"construction graph contains a cycle: {cycle_nodes}")
        result.extend(ready)
        remaining.difference_update(ready)
    return result


def compile_assembly_plan(graph: ConstructionGraph) -> dict:
    order = topological_order(graph)
    by_id = {item.operation_id: item for item in graph.operations}
    sequence = []
    completed: set[str] = set()
    for index, operation_id in enumerate(order, start=1):
        operation = by_id[operation_id]
        if not set(operation.depends_on).issubset(completed):
            raise AssertionError(f"topological dependency violation: {operation_id}")
        sequence.append({
            "sequence": index,
            **operation.to_dict(),
        })
        completed.add(operation_id)
    return {
        "contract": "AssemblyPlan/1",
        "graph": graph.to_dict(),
        "operation_count": len(sequence),
        "cycle_free": True,
        "ordered_operation_ids": order,
        "sequence": sequence,
    }


def cycle_failure_probe(graph: ConstructionGraph) -> dict:
    rows = list(graph.operations)
    if len(rows) < 2:
        raise ValueError("cycle probe requires at least two operations")
    first, second = rows[0], rows[1]
    from .model import AssemblyOperation

    invalid = ConstructionGraph(
        f"{graph.graph_id}_CYCLE_PROBE",
        (
            AssemblyOperation(first.operation_id, first.operation_type, first.owner_ids, (second.operation_id,)),
            AssemblyOperation(second.operation_id, second.operation_type, second.owner_ids, (first.operation_id,)),
            *rows[2:],
        ),
    )
    try:
        compile_assembly_plan(invalid)
    except ValueError as exc:
        return {
            "probe": "ASSEMBLY_CYCLE",
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
            "failure_atomicity": True,
        }
    raise AssertionError("cyclic assembly graph unexpectedly accepted")
