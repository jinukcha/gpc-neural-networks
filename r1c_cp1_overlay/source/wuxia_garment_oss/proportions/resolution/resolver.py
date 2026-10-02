"""Atomic deterministic resolution of absolute, relative, and derived parameters."""
from __future__ import annotations

from dataclasses import dataclass

from ..expression.dependency import build_dependency_graph, topological_order
from ..expression.evaluator import evaluate_expression
from ..expression.parser import parse_expression, referenced_paths
from ..model.parameter import ParameterDefinition
from ..model.quantity import TypedValue, canonical_unit, convert_to_si, quantity_dimension
from .bounds import BoundFailure, apply_bounds
from .context import ParameterResolutionContext
from .receipt import canonical_sha256


@dataclass(frozen=True)
class ResolutionFailure(Exception):
    code: str
    message: str
    parameter_id: str | None = None
    details: dict | None = None

    def __str__(self) -> str:
        return self.message


def _definition_set_payload(definitions: tuple[ParameterDefinition, ...]) -> dict:
    payload = {
        "contract": "ParameterDefinitionSet/1",
        "definitions": [item.to_dict() for item in sorted(definitions, key=lambda value: value.parameter_id)],
    }
    payload["definition_set_sha256"] = canonical_sha256(payload)
    return payload


def _definition_error_code(message: str) -> str:
    if "unit" in message:
        return "UNIT_MISMATCH"
    if "finite ratio" in message:
        return "NON_FINITE_RATIO"
    return "INVALID_PARAMETER_DEFINITION"


def _validate_definitions(definitions: tuple[ParameterDefinition, ...]) -> None:
    ids = [item.parameter_id for item in definitions]
    if not definitions or len(ids) != len(set(ids)):
        raise ResolutionFailure("INVALID_DEFINITION_SET", "parameter definitions must be non-empty and unique")
    for item in definitions:
        try:
            item.validate()
        except ValueError as error:
            message = str(error)
            raise ResolutionFailure(_definition_error_code(message), message, item.parameter_id) from error


def _relative_value(
    definition: ParameterDefinition,
    context: ParameterResolutionContext,
) -> tuple[float, tuple[str, ...]]:
    try:
        reference = context.reference(definition.reference_scope or "", definition.reference_path or "")
    except KeyError as error:
        raise ResolutionFailure("MISSING_REFERENCE", str(error), definition.parameter_id) from error
    if reference.quantity != definition.quantity:
        raise ResolutionFailure(
            "QUANTITY_MISMATCH",
            f"reference quantity {reference.quantity} does not match {definition.quantity}",
            definition.parameter_id,
        )
    return reference.value_si * float(definition.ratio), (reference.qualified_path,)


def _auto_error_code(message: str) -> str:
    return "QUANTITY_MISMATCH" if "quantity mismatch" in message or "dimension" in message else "EXPRESSION_ERROR"


def _auto_value(
    definition: ParameterDefinition,
    environment: dict[str, TypedValue],
) -> tuple[float, tuple[str, ...]]:
    try:
        expression = definition.expression or ""
        result = evaluate_expression(expression, environment)
        paths = referenced_paths(parse_expression(expression))
    except ValueError as error:
        message = str(error)
        raise ResolutionFailure(_auto_error_code(message), message, definition.parameter_id) from error
    expected = quantity_dimension(definition.quantity)
    if result.dimension != expected:
        raise ResolutionFailure(
            "QUANTITY_MISMATCH",
            f"expression dimension {result.dimension} does not match {expected}",
            definition.parameter_id,
        )
    return result.value_si, paths


def _raw_value(
    definition: ParameterDefinition,
    context: ParameterResolutionContext,
    environment: dict[str, TypedValue],
) -> tuple[float, tuple[str, ...]]:
    if definition.mode == "ABSOLUTE":
        try:
            value = convert_to_si(definition.absolute_value or 0.0, definition.unit, definition.quantity)
        except ValueError as error:
            raise ResolutionFailure("UNIT_MISMATCH", str(error), definition.parameter_id) from error
        return value, ()
    if definition.mode == "RELATIVE":
        return _relative_value(definition, context)
    return _auto_value(definition, environment)


def _entry(
    definition: ParameterDefinition,
    requested: float,
    resolved: float,
    paths: tuple[str, ...],
    dependencies: tuple[str, ...],
    bound: dict,
    index: int,
) -> dict:
    return {
        "parameter_id": definition.parameter_id,
        "owner_component_id": definition.owner_component_id,
        "mode": definition.mode,
        "quantity": definition.quantity,
        "unit": canonical_unit(definition.quantity),
        "requested_value_si": requested,
        "resolved_value_si": resolved,
        "reference_paths": list(paths),
        "parameter_dependencies": list(dependencies),
        "expression": definition.expression,
        "ratio": definition.ratio,
        "bounds": bound,
        "resolution_index": index,
    }


def _resolve(
    definitions: tuple[ParameterDefinition, ...],
    context: ParameterResolutionContext,
    resolution_id: str,
) -> tuple[dict, dict]:
    _validate_definitions(definitions)
    try:
        context.validate()
        graph = build_dependency_graph(definitions)
        order = topological_order(graph)
    except ValueError as error:
        code = "CYCLE_DETECTED" if str(error).startswith("CYCLE_DETECTED") else "DEPENDENCY_ERROR"
        raise ResolutionFailure(code, str(error)) from error
    by_id = {item.parameter_id: item for item in definitions}
    environment = context.typed_environment()
    entries = []
    for index, parameter_id in enumerate(order):
        definition = by_id[parameter_id]
        requested, paths = _raw_value(definition, context, environment)
        try:
            outcome = apply_bounds(requested, definition.bounds)
        except BoundFailure as error:
            raise ResolutionFailure(
                error.code,
                str(error),
                parameter_id,
                {"value_si": error.value_si, "bounds": error.spec.to_dict()},
            ) from error
        environment[f"param.{parameter_id}"] = TypedValue(
            outcome.resolved_value_si,
            quantity_dimension(definition.quantity),
        )
        entries.append(
            _entry(
                definition,
                requested,
                outcome.resolved_value_si,
                paths,
                graph[parameter_id],
                outcome.to_dict(),
                index,
            )
        )
    definition_set = _definition_set_payload(definitions)
    context_payload = context.to_dict()
    resolved_set = {
        "contract": "ResolvedParameterSet/1",
        "resolution_id": resolution_id,
        "context_sha256": context_payload["context_sha256"],
        "definition_set_sha256": definition_set["definition_set_sha256"],
        "resolution_order": list(order),
        "parameters": entries,
        "immutable": True,
    }
    resolved_set["resolved_set_sha256"] = canonical_sha256(resolved_set)
    receipt = {
        "contract": "ParameterResolutionReceipt/1",
        "resolution_id": resolution_id,
        "status": "RESOLVED",
        "accepted": True,
        "resolved_set_sha256": resolved_set["resolved_set_sha256"],
        "context_sha256": context_payload["context_sha256"],
        "definition_set_sha256": definition_set["definition_set_sha256"],
        "parameter_count": len(entries),
        "clamp_count": sum(item["bounds"]["action"] == "CLAMPED" for item in entries),
        "partial_publication_count": 0,
        "error": None,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return resolved_set, receipt


def resolve_parameter_set(
    definitions: tuple[ParameterDefinition, ...],
    context: ParameterResolutionContext,
    resolution_id: str,
) -> tuple[dict | None, dict]:
    try:
        return _resolve(definitions, context, resolution_id)
    except ResolutionFailure as error:
        receipt = {
            "contract": "ParameterResolutionReceipt/1",
            "resolution_id": resolution_id,
            "status": error.code,
            "accepted": False,
            "resolved_set_sha256": None,
            "parameter_count": 0,
            "clamp_count": 0,
            "partial_publication_count": 0,
            "error": {
                "code": error.code,
                "message": error.message,
                "parameter_id": error.parameter_id,
                "details": error.details or {},
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        return None, receipt
