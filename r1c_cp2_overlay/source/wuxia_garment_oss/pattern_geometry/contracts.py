"""JSON schemas for R1C CP2 exact component and assembly products."""
from __future__ import annotations


def _schema(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def cp2_schemas() -> dict[str, dict]:
    string = {"type": "string", "minLength": 1}
    return {
        "component_geometry_authority.schema.json": _schema(
            "ComponentGeometryAuthority/1",
            ["instance_id", "component_id", "segments", "boundaries", "geometry_sha256"],
            {
                "instance_id": string,
                "component_id": string,
                "segments": {"type": "array", "minItems": 1},
                "boundaries": {"type": "array", "minItems": 1},
                "notches": {"type": "array"},
                "geometry_sha256": string,
            },
        ),
        "pattern_geometry_library.schema.json": _schema(
            "PatternGeometryLibrary/1",
            ["library_id", "authorities", "library_sha256"],
            {
                "library_id": string,
                "authority_count": {"type": "integer", "minimum": 1},
                "authorities": {"type": "array", "minItems": 1},
                "library_sha256": string,
            },
        ),
        "interface_compatibility_receipt.schema.json": _schema(
            "InterfaceCompatibilityReceipt/1",
            ["interface_id", "accepted", "receipt_sha256"],
            {
                "interface_id": string,
                "accepted": {"type": "boolean"},
                "length_a_m": {"type": "number", "exclusiveMinimum": 0},
                "length_b_m": {"type": "number", "exclusiveMinimum": 0},
                "notch_correspondence": {"type": "array"},
                "receipt_sha256": string,
            },
        ),
        "assembled_pattern_package.schema.json": _schema(
            "AssembledPatternPackage/1",
            ["assembly_id", "component_instances", "seams", "assembled_package_sha256"],
            {
                "assembly_id": string,
                "component_instances": {"type": "array", "minItems": 1},
                "seams": {"type": "array", "minItems": 1},
                "assembled_package_sha256": string,
                "triangulation_executed": {"const": False},
                "simulation_executed": {"const": False},
            },
        ),
        "assembly_compilation_receipt.schema.json": _schema(
            "AssemblyCompilationReceipt/1",
            ["recipe_id", "accepted", "status", "receipt_sha256"],
            {
                "recipe_id": string,
                "accepted": {"type": "boolean"},
                "status": {"enum": ["COMPILED", "REJECTED_ATOMIC"]},
                "partial_publication_count": {"const": 0},
                "receipt_sha256": string,
            },
        ),
    }
