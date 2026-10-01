"""JSON Schema contracts for professional construction outputs."""
from __future__ import annotations


def _boundary_schema() -> dict:
    return {
        "type": "object",
        "required": ["panel_id", "curve_id", "start_fraction", "end_fraction"],
        "properties": {
            "panel_id": {"type": "string", "minLength": 1},
            "curve_id": {"type": "string", "minLength": 1},
            "start_fraction": {"type": "number", "minimum": 0, "maximum": 1},
            "end_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    }


def seam_spec_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://wuxia.local/contracts/seam_spec_v2.schema.json",
        "title": "SeamSpec/2",
        "type": "object",
        "required": [
            "seam_id", "seam_type", "side_a", "side_b", "allowance_a_m",
            "allowance_b_m", "stitch_class", "ease_ratio", "gather_ratio",
        ],
        "properties": {
            "seam_id": {"type": "string", "minLength": 1},
            "seam_type": {"enum": ["PLAIN_SEAM", "GATHERED_SEAM", "GUSSET_INSERTION", "WAIST_JOIN"]},
            "side_a": _boundary_schema(),
            "side_b": _boundary_schema(),
            "allowance_a_m": {"type": "number", "minimum": 0, "maximum": 0.05},
            "allowance_b_m": {"type": "number", "minimum": 0, "maximum": 0.05},
            "stitch_class": {"type": "string", "minLength": 1},
            "ease_ratio": {"type": "number", "minimum": 0.5, "maximum": 2.5},
            "gather_ratio": {"type": "number", "minimum": 1, "maximum": 3},
            "notch_pair_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "fold_direction": {"type": "string"},
            "topstitch_offset_m": {"type": "number", "minimum": 0},
            "turn_of_cloth_m": {"type": "number", "minimum": 0},
        },
        "additionalProperties": False,
    }


def construction_graph_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://wuxia.local/contracts/construction_graph.schema.json",
        "title": "ConstructionGraph/1",
        "type": "object",
        "required": ["contract", "graph_id", "operations", "graph_sha256"],
        "properties": {
            "contract": {"const": "ConstructionGraph/1"},
            "graph_id": {"type": "string", "minLength": 1},
            "operations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["operation_id", "operation_type", "owner_ids", "depends_on"],
                    "properties": {
                        "operation_id": {"type": "string", "minLength": 1},
                        "operation_type": {"type": "string", "minLength": 1},
                        "owner_ids": {"type": "array", "items": {"type": "string"}},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": False,
                },
            },
            "graph_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "additionalProperties": False,
    }


def construction_line_schema() -> dict:
    point = {
        "type": "array",
        "prefixItems": [{"type": "number"}, {"type": "number"}],
        "minItems": 2,
        "maxItems": 2,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://wuxia.local/contracts/construction_line_package.schema.json",
        "title": "ConstructionLinePackage/1",
        "type": "object",
        "required": ["panel_id", "curve_id", "allowance_m", "stitch_line", "cut_line"],
        "properties": {
            "panel_id": {"type": "string"},
            "curve_id": {"type": "string"},
            "interval": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "allowance_m": {"type": "number", "minimum": 0},
            "stitch_line": {"type": "array", "minItems": 2, "items": point},
            "cut_line": {"type": "array", "minItems": 2, "items": point},
            "stitch_length_m": {"type": "number", "minimum": 0},
            "cut_length_m": {"type": "number", "minimum": 0},
            "sample_count": {"type": "integer", "minimum": 2},
        },
        "additionalProperties": False,
    }


def construction_package_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://wuxia.local/contracts/construction_package.schema.json",
        "title": "ConstructionPackage/1",
        "type": "object",
        "required": [
            "contract", "source_pattern_document_sha256", "seam_specs", "seam_lines",
            "edge_finishes", "notch_correspondence", "closures", "facings",
            "layer_pieces", "turn_of_cloth", "assembly_plan", "bill_of_materials",
            "package_sha256",
        ],
        "properties": {
            "contract": {"const": "ConstructionPackage/1"},
            "source_pattern_document_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "source_pattern_revision": {"type": "integer", "minimum": 0},
            "seam_specs": {"type": "array", "minItems": 1},
            "seam_lines": {"type": "array", "minItems": 1},
            "edge_finishes": {"type": "array", "minItems": 1},
            "notch_correspondence": {"type": "array", "minItems": 1},
            "closures": {"type": "array", "minItems": 1},
            "facings": {"type": "array", "minItems": 1},
            "layer_pieces": {"type": "array", "minItems": 1},
            "turn_of_cloth": {"type": "array", "minItems": 1},
            "assembly_plan": {"type": "object"},
            "bill_of_materials": {"type": "array", "minItems": 1},
            "source_document_mutated": {"const": False},
            "triangulation_executed": {"const": False},
            "warp_simulation_executed": {"const": False},
            "mesh_scaling": {"const": "FORBIDDEN"},
            "package_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "additionalProperties": False,
    }


def construction_schemas() -> dict[str, dict]:
    return {
        "seam_spec_v2.schema.json": seam_spec_schema(),
        "construction_graph.schema.json": construction_graph_schema(),
        "construction_line_package.schema.json": construction_line_schema(),
        "construction_package.schema.json": construction_package_schema(),
    }
