"""JSON schemas for rigged garment and Godot equip products."""
from __future__ import annotations


def _schema(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def product_schemas() -> dict[str, dict]:
    string = {"type": "string", "minLength": 1}
    return {
        "rigged_garment_product.schema.json": _schema(
            "RiggedGarmentProduct/1",
            ["garment_id", "path", "bone_count", "skin_count", "product_sha256"],
            {
                "garment_id": string,
                "path": string,
                "bone_count": {"const": 23},
                "skin_count": {"const": 1},
                "vertex_count": {"type": "integer", "minimum": 1},
                "product_sha256": string,
            },
        ),
        "godot_garment_equip_package.schema.json": _schema(
            "GodotGarmentEquipPackage/1",
            ["engine", "runtime_script", "products", "atomic_policy", "package_sha256"],
            {
                "engine": {"const": "Godot 4.7.2 Linux"},
                "runtime_script": string,
                "products": {"type": "object", "minProperties": 2},
                "atomic_policy": {"type": "object"},
                "package_sha256": string,
            },
        ),
        "character_swap_contract.schema.json": _schema(
            "CharacterSwapContract/1",
            ["compatible_adapter", "validate_before_commit", "contract_sha256"],
            {
                "compatible_adapter": string,
                "validate_before_commit": {"const": True},
                "failed_swap_preserves_current_target": {"type": "boolean"},
                "contract_sha256": string,
            },
        ),
        "godot_equip_receipt.schema.json": _schema(
            "GodotGarmentEquipReceipt/1",
            ["godot_version", "transactions", "consumer_pass"],
            {
                "godot_version": {"type": "object"},
                "transactions": {"type": "array", "minItems": 7},
                "consumer_pass": {"type": "boolean"},
            },
        ),
    }
