"""JSON schemas for the R1C CP1 ratio parameter engine."""
from __future__ import annotations


def _schema(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def parameter_schemas() -> dict[str, dict]:
    string = {"type": "string", "minLength": 1}
    number = {"type": "number"}
    return {
        "parameter_definition.schema.json": _schema(
            "ParameterDefinition/1",
            ["parameter_id", "mode", "quantity", "unit", "bounds"],
            {
                "parameter_id": string,
                "mode": {"enum": ["ABSOLUTE", "RELATIVE", "AUTO_DERIVED"]},
                "quantity": {"enum": ["LENGTH", "AREA", "ANGLE", "MASS_PER_AREA", "DIMENSIONLESS"]},
                "unit": string,
                "absolute_value": {"type": ["number", "null"]},
                "ratio": {"type": ["number", "null"]},
                "expression": {"type": ["string", "null"]},
                "bounds": {"type": "object"},
            },
        ),
        "parameter_reference.schema.json": _schema(
            "ParameterReference/1",
            ["scope", "path", "quantity", "unit", "value_si", "source_package_sha256"],
            {
                "scope": {"enum": ["BODY_RELATIVE", "BLOCK_RELATIVE", "COMPONENT_RELATIVE", "BOUNDARY_RELATIVE", "MATERIAL_RELATIVE"]},
                "path": string,
                "quantity": string,
                "unit": string,
                "value_si": number,
                "source_package_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
            },
        ),
        "parameter_resolution_context.schema.json": _schema(
            "ParameterResolutionContext/1",
            ["context_id", "references", "context_sha256"],
            {
                "context_id": string,
                "references": {"type": "array", "minItems": 1},
                "context_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
            },
        ),
        "resolved_parameter_set.schema.json": _schema(
            "ResolvedParameterSet/1",
            ["resolution_id", "resolution_order", "parameters", "resolved_set_sha256"],
            {
                "resolution_id": string,
                "resolution_order": {"type": "array", "items": string},
                "parameters": {"type": "array", "minItems": 1},
                "immutable": {"const": True},
                "resolved_set_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
            },
        ),
        "parameter_resolution_receipt.schema.json": _schema(
            "ParameterResolutionReceipt/1",
            ["resolution_id", "status", "accepted", "partial_publication_count", "receipt_sha256"],
            {
                "resolution_id": string,
                "status": string,
                "accepted": {"type": "boolean"},
                "partial_publication_count": {"const": 0},
                "receipt_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
            },
        ),
    }
