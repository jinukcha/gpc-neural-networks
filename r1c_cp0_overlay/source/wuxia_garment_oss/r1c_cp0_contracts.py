"""JSON schemas for R1C CP0 contracts."""
from __future__ import annotations


def _schema(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def r1c_cp0_schemas() -> dict[str, dict]:
    string = {"type": "string", "minLength": 1}
    return {
        "pattern_component_definition.schema.json": _schema(
            "PatternComponentDefinition/1",
            ["component_id", "geometry_authority", "boundaries", "component_sha256"],
            {
                "component_id": string,
                "geometry_authority": {"const": "EXACT_2D_PATTERN"},
                "boundaries": {"type": "array", "minItems": 1},
                "three_dimensional_primitive_authority": {"const": False},
                "component_sha256": string,
            },
        ),
        "component_interface_spec.schema.json": _schema(
            "ComponentInterfaceSpec/1",
            ["interface_id", "endpoint_a", "endpoint_b", "length_policy", "interface_sha256"],
            {
                "interface_id": string,
                "endpoint_a": {"type": "object"},
                "endpoint_b": {"type": "object"},
                "length_policy": {"enum": ["EQUAL", "BOUNDED_EASE", "GATHERED"]},
                "interface_sha256": string,
            },
        ),
        "garment_assembly_recipe.schema.json": _schema(
            "GarmentAssemblyRecipe/1",
            ["recipe_id", "component_instances", "interface_ids", "recipe_sha256"],
            {
                "recipe_id": string,
                "component_instances": {"type": "array", "minItems": 1},
                "interface_ids": {"type": "array"},
                "three_dimensional_primitive_fallback": {"const": False},
                "recipe_sha256": string,
            },
        ),
        "assembly_admission_receipt.schema.json": _schema(
            "AssemblyAdmissionReceipt/1",
            ["recipe_id", "accepted", "errors", "receipt_sha256"],
            {
                "recipe_id": string,
                "accepted": {"type": "boolean"},
                "errors": {"type": "array"},
                "geometry_executed": {"const": False},
                "simulation_executed": {"const": False},
                "receipt_sha256": string,
            },
        ),
        "visual_acceptance_profile.schema.json": _schema(
            "VisualAcceptanceProfile/1",
            ["profile_id", "required_views", "gates", "profile_sha256"],
            {
                "profile_id": string,
                "required_views": {"type": "array", "minItems": 20},
                "gates": {"type": "array", "minItems": 10},
                "acceptance_expression": {"const": "TECHNICAL_PASS_AND_VISUAL_PASS"},
                "profile_sha256": string,
            },
        ),
        "visual_review_receipt.schema.json": _schema(
            "VisualReviewReceipt/1",
            ["profile_id", "visual_review", "product_acceptance", "receipt_sha256"],
            {
                "profile_id": string,
                "visual_review": {"enum": ["PASS", "FAIL"]},
                "product_acceptance": {"type": "boolean"},
                "pattern_mutated": {"const": False},
                "geometry_mutated": {"const": False},
                "receipt_sha256": string,
            },
        ),
    }
