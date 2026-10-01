"""JSON Schema for the CP3 garment geometry package."""
from __future__ import annotations


def geometry_package_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "GarmentGeometryPackage/1",
        "type": "object",
        "required": [
            "contract", "geometry_package_id", "source_parameter_package_id",
            "panels", "seam_correspondence", "qualification",
            "triangulation_admission",
        ],
        "properties": {
            "contract": {"const": "GarmentGeometryPackage/1"},
            "geometry_package_id": {"type": "string", "minLength": 64},
            "source_parameter_package_id": {"type": "string", "minLength": 64},
            "panels": {"type": "array", "minItems": 4, "maxItems": 4},
            "seam_correspondence": {"type": "array", "minItems": 8, "maxItems": 8},
            "qualification": {"type": "object"},
            "triangulation_admission": {"enum": ["PASS", "FAIL"]},
            "warp_simulation_executed": {"const": False},
            "product_simulation_executed": {"const": False},
            "mesh_scaling": {"const": "FORBIDDEN"},
        },
        "additionalProperties": True,
    }
