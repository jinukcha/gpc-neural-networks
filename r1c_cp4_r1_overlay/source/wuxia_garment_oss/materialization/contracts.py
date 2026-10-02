"""JSON Schema contracts for CP4-R1 products and receipts."""
from __future__ import annotations


def _object(required, properties):
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": True,
    }


def cp4_r1_schemas() -> dict[str, dict]:
    return {
        "component_triangulation_receipt.schema.json": _object(
            ["contract", "component_count", "vertex_count", "triangle_count"],
            {
                "contract": {"const": "ComponentTriangulationReceipt/1"},
                "component_count": {"type": "integer", "minimum": 1},
                "vertex_count": {"type": "integer", "minimum": 1},
                "triangle_count": {"type": "integer", "minimum": 1},
            },
        ),
        "seam_correspondence_receipt.schema.json": _object(
            ["contract", "interface_count", "direct_count", "reversed_count", "interfaces"],
            {
                "contract": {"const": "SeamCorrespondenceReceipt/1"},
                "interface_count": {"type": "integer", "minimum": 1},
                "direct_count": {"type": "integer", "minimum": 0},
                "reversed_count": {"type": "integer", "minimum": 0},
                "interfaces": {"type": "array", "minItems": 1},
            },
        ),
        "compiled_rest_metric.schema.json": _object(
            ["contract", "vertex_count", "edge_count", "settling_rest_owner"],
            {
                "contract": {"const": "CompiledArrangementRestMetric/1"},
                "vertex_count": {"type": "integer", "minimum": 1},
                "edge_count": {"type": "integer", "minimum": 1},
                "settling_rest_owner": {"const": "ARRANGEMENT_REST_METRIC"},
            },
        ),
        "warp_material_settling_receipt.schema.json": _object(
            ["contract", "runtime", "material", "frames", "post_settle_vertex_repair_count"],
            {
                "contract": {"const": "WarpMaterialSettlingReceipt/1"},
                "runtime": {"const": "warp-lang"},
                "material": {"type": "object"},
                "frames": {"type": "integer", "minimum": 1},
                "post_settle_vertex_repair_count": {"const": 0},
            },
        ),
        "technical_qualification_receipt.schema.json": _object(
            ["contract", "technical_pass", "gates", "visual_review"],
            {
                "contract": {"const": "CP4R1TechnicalQualificationReceipt/1"},
                "technical_pass": {"type": "boolean"},
                "gates": {"type": "object"},
                "visual_review": {"type": "string"},
            },
        ),
        "visual_evidence_receipt.schema.json": _object(
            ["contract", "required_views", "view_count", "visual_review"],
            {
                "contract": {"const": "CP4R1VisualEvidenceReceipt/1"},
                "required_views": {"type": "array", "minItems": 1},
                "view_count": {"type": "integer", "minimum": 1},
                "visual_review": {"enum": ["PASS", "FAIL"]},
            },
        ),
    }
