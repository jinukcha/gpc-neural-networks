"""Resolve PatternDocument expressions, points, curves, and constraints."""
from __future__ import annotations

import math
from typing import Mapping

from ..expressions.evaluator import evaluate_expression_dag
from .model import PatternConstraint, PatternDocument


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _constraint_residual(
    constraint: PatternConstraint,
    points: Mapping[str, tuple[float, float]],
    values: Mapping[str, float],
) -> float:
    selected = [points[name] for name in constraint.point_ids]
    if constraint.kind == "COINCIDENT":
        return _point_distance(selected[0], selected[1])
    if constraint.kind == "HORIZONTAL":
        return abs(selected[0][1] - selected[1][1])
    if constraint.kind == "VERTICAL":
        return abs(selected[0][0] - selected[1][0])
    if constraint.kind == "SYMMETRY_X":
        return max(abs(selected[0][0] + selected[1][0]), abs(selected[0][1] - selected[1][1]))
    if constraint.target_expression is None:
        raise ValueError(f"constraint target missing: {constraint.constraint_id}")
    target = _target_value(constraint.target_expression, values)
    distance = _point_distance(selected[0], selected[1])
    if constraint.kind == "FIXED_DISTANCE":
        return abs(distance - target)
    if constraint.kind == "MIN_DISTANCE":
        return max(0.0, target - distance)
    raise ValueError(f"unsupported constraint: {constraint.kind}")


def _target_value(expression: str, values: Mapping[str, float]) -> float:
    result = evaluate_expression_dag(values, {"__target__": expression})
    return float(result["__target__"])


def _resolved_points(document: PatternDocument, values: Mapping[str, float]) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for point_id, point in sorted(document.points.items()):
        x = _target_value(point.x_expression, values)
        y = _target_value(point.y_expression, values)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"non-finite resolved point: {point_id}")
        points[point_id] = (x, y)
    return points


def _curve_payload(document: PatternDocument, points: Mapping[str, tuple[float, float]]) -> dict:
    payload = {}
    for curve_id, curve in sorted(document.curves.items()):
        payload[curve_id] = {
            "curve_id": curve_id,
            "panel_id": curve.panel_id,
            "curve_type": curve.curve_type,
            "point_ids": list(curve.point_ids),
            "points": [list(points[point_id]) for point_id in curve.point_ids],
            "boundary_role": curve.boundary_role,
            "disposition": curve.disposition,
        }
    return payload


def resolve_document(document: PatternDocument) -> dict:
    document.validate_structure()
    expression_values = evaluate_expression_dag(document.inputs, document.expressions)
    values = {**document.inputs, **expression_values}
    points = _resolved_points(document, values)
    constraint_rows = []
    hard_failures = []
    for constraint_id, constraint in sorted(document.constraints.items()):
        residual = _constraint_residual(constraint, points, values)
        passed = residual <= constraint.tolerance
        row = {
            "constraint_id": constraint_id,
            "kind": constraint.kind,
            "strength": constraint.strength,
            "residual": residual,
            "tolerance": constraint.tolerance,
            "passed": passed,
        }
        constraint_rows.append(row)
        if constraint.strength == "HARD" and not passed:
            hard_failures.append(row)
    return {
        "contract": "ResolvedPatternDocument/1",
        "document_id": document.document_id,
        "revision": document.revision,
        "document_sha256": document.to_dict()["document_sha256"],
        "values": {name: values[name] for name in sorted(values)},
        "points": {name: list(points[name]) for name in sorted(points)},
        "curves": _curve_payload(document, points),
        "constraints": constraint_rows,
        "hard_constraint_failures": hard_failures,
        "hard_constraints_pass": not hard_failures,
    }


def require_resolved(document: PatternDocument) -> dict:
    resolved = resolve_document(document)
    if not resolved["hard_constraints_pass"]:
        failed = [item["constraint_id"] for item in resolved["hard_constraint_failures"]]
        raise ValueError(f"hard constraints failed: {failed}")
    return resolved
