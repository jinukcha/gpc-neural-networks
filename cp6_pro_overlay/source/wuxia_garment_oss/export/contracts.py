"""JSON schemas for CP6 manufacturing and game garment products."""
from __future__ import annotations


def _schema(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def cp6_schemas() -> dict[str, dict]:
    string = {"type": "string", "minLength": 1}
    return {
        "manufacturing_pattern_package.schema.json": _schema(
            "ManufacturingPatternPackage/1",
            ["garment_id", "scale", "panels", "package_sha256"],
            {
                "garment_id": string,
                "scale": {"const": "1:1"},
                "panels": {"type": "array", "minItems": 1},
                "package_sha256": string,
            },
        ),
        "game_garment_product.schema.json": _schema(
            "GameGarmentProduct/1",
            ["product_id", "glb_sha256", "primitive_count", "receipt_sha256"],
            {
                "product_id": string,
                "glb_sha256": string,
                "primitive_count": {"type": "integer", "minimum": 1},
                "morph_target_names": {"type": "array"},
                "receipt_sha256": string,
            },
        ),
        "trousers_pattern_authority.schema.json": _schema(
            "TrousersPatternAuthority/1",
            ["document_id", "points_of_measure", "panel_ids", "authority_sha256"],
            {
                "document_id": string,
                "points_of_measure": {"type": "object", "minProperties": 15},
                "panel_ids": {"type": "array", "minItems": 7},
                "authority_sha256": string,
            },
        ),
        "trousers_topology_receipt.schema.json": _schema(
            "TrousersTopologyReceipt/1",
            ["vertex_count", "triangle_count", "topology_pass", "receipt_sha256"],
            {
                "vertex_count": {"type": "integer", "minimum": 1},
                "triangle_count": {"type": "integer", "minimum": 1},
                "topology_pass": {"type": "boolean"},
                "receipt_sha256": string,
            },
        ),
        "trousers_pose_qualification.schema.json": _schema(
            "TrousersPoseQualificationReceipt/1",
            ["material_id", "pose_id", "metrics", "gates", "pose_pass", "receipt_sha256"],
            {
                "material_id": string,
                "pose_id": string,
                "metrics": {"type": "object"},
                "gates": {"type": "object"},
                "pose_pass": {"type": "boolean"},
                "receipt_sha256": string,
            },
        ),
        "glb_fresh_reopen_receipt.schema.json": _schema(
            "GLBFreshReopenReceipt/1",
            ["path", "fresh_reopen_pass", "receipt_sha256"],
            {
                "path": string,
                "fresh_reopen_pass": {"type": "boolean"},
                "receipt_sha256": string,
            },
        ),
    }
