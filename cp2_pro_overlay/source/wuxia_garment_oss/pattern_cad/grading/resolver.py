"""Resolve arbitrary-N grade variants and preserve notch arc position."""
from __future__ import annotations

import math

from ..document.model import PatternDocument
from ..document.resolver import require_resolved
from .model import GradeRuleSetV2, PatternNotchSet


def _curve_point(curve_type: str, points: list[list[float]], t: float) -> tuple[float, float]:
    if curve_type == "LINE":
        weights = (1.0 - t, t)
    elif curve_type == "QUADRATIC_BEZIER":
        weights = ((1.0 - t) ** 2, 2.0 * (1.0 - t) * t, t * t)
    elif curve_type == "CUBIC_BEZIER":
        weights = (
            (1.0 - t) ** 3,
            3.0 * (1.0 - t) ** 2 * t,
            3.0 * (1.0 - t) * t * t,
            t ** 3,
        )
    else:
        raise ValueError(f"unsupported curve type: {curve_type}")
    x = sum(weight * float(point[0]) for weight, point in zip(weights, points))
    y = sum(weight * float(point[1]) for weight, point in zip(weights, points))
    return x, y


def _arc_samples(curve: dict, count: int = 512) -> tuple[list[tuple[float, float]], list[float]]:
    points = [_curve_point(curve["curve_type"], curve["points"], i / count) for i in range(count + 1)]
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))
    return points, cumulative


def point_at_arc_fraction(curve: dict, fraction: float) -> tuple[float, float]:
    points, cumulative = _arc_samples(curve)
    total = cumulative[-1]
    if total <= 0.0:
        raise ValueError(f"zero-length curve: {curve['curve_id']}")
    target = min(1.0, max(0.0, fraction)) * total
    for index in range(1, len(cumulative)):
        if cumulative[index] < target:
            continue
        span = cumulative[index] - cumulative[index - 1]
        alpha = 0.0 if span <= 0.0 else (target - cumulative[index - 1]) / span
        first, second = points[index - 1], points[index]
        return (
            first[0] + alpha * (second[0] - first[0]),
            first[1] + alpha * (second[1] - first[1]),
        )
    return points[-1]


def grade_document(base: PatternDocument, rule_set: GradeRuleSetV2, size_id: str) -> PatternDocument:
    rule_set.validate(base)
    grade = rule_set.size(size_id)
    candidate = base.clone()
    for input_id, delta in grade.input_deltas.items():
        candidate.inputs[input_id] = float(candidate.inputs[input_id] + delta)
    candidate.document_id = f"{base.document_id}__GRADE_{size_id}"
    candidate.parent_revision = base.revision
    candidate.revision = base.revision + 1
    candidate.metadata = {
        **candidate.metadata,
        "grade_rule_set_id": rule_set.grade_rule_set_id,
        "grade_size_id": size_id,
        "grade_ordinal": grade.ordinal,
        "grade_source_document_sha256": base.to_dict()["document_sha256"],
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
    }
    require_resolved(candidate)
    return candidate


def _grade_point_movements(
    base_resolved: dict,
    variant_resolved: dict,
    rule_set: GradeRuleSetV2,
) -> list[dict]:
    rows = []
    for grade_point in rule_set.grade_points:
        before = base_resolved["points"][grade_point.point_id]
        after = variant_resolved["points"][grade_point.point_id]
        rows.append({
            **grade_point.to_dict(),
            "before": before,
            "after": after,
            "delta": [after[0] - before[0], after[1] - before[1]],
        })
    return rows


def _curve_propagation(variant_resolved: dict, rule_set: GradeRuleSetV2) -> dict:
    checked = 0
    maximum_error = 0.0
    grade_point_ids = {item.point_id for item in rule_set.grade_points}
    for curve in variant_resolved["curves"].values():
        for point_id, curve_point in zip(curve["point_ids"], curve["points"]):
            if point_id not in grade_point_ids:
                continue
            owner_point = variant_resolved["points"][point_id]
            maximum_error = max(maximum_error, math.dist(curve_point, owner_point))
            checked += 1
    return {
        "checked_curve_point_references": checked,
        "maximum_curve_propagation_error": maximum_error,
        "passed": checked > 0 and maximum_error <= 1.0e-12,
    }


def resolve_notches(notch_set: PatternNotchSet, resolved: dict) -> list[dict]:
    rows = []
    for notch in notch_set.notches:
        curve = resolved["curves"][notch.curve_id]
        position = point_at_arc_fraction(curve, notch.arc_fraction)
        rows.append({
            **notch.to_dict(),
            "position": [position[0], position[1]],
            "curve_disposition": curve["disposition"],
        })
    return rows


def compile_grade_variants(
    base: PatternDocument,
    rule_set: GradeRuleSetV2,
    notch_set: PatternNotchSet,
) -> dict[str, dict]:
    rule_set.validate(base)
    notch_set.validate(base)
    before_sha = base.to_dict()["document_sha256"]
    base_resolved = require_resolved(base)
    variants = {}
    for size in rule_set.ordered_sizes:
        document = grade_document(base, rule_set, size.size_id)
        resolved = require_resolved(document)
        variants[size.size_id] = {
            "contract": "GradedPatternVariant/1",
            "size_id": size.size_id,
            "ordinal": size.ordinal,
            "document": document.to_dict(),
            "resolved": resolved,
            "grade_point_movements": _grade_point_movements(base_resolved, resolved, rule_set),
            "curve_propagation": _curve_propagation(resolved, rule_set),
            "notches": resolve_notches(notch_set, resolved),
        }
    if base.to_dict()["document_sha256"] != before_sha:
        raise AssertionError("grade compilation mutated the base document")
    return variants
