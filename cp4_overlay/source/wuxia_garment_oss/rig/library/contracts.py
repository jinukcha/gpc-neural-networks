"""JSON schemas for CP4 registry, outfit, occlusion, and transactions."""
from __future__ import annotations


def _schema(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def cp4_schemas() -> dict[str, dict]:
    string = {"type": "string", "minLength": 1}
    return {
        "garment_library_registry.schema.json": _schema(
            "GarmentLibraryRegistry/1",
            ["registry_id", "entries", "regional_thickness_limits_m", "registry_sha256"],
            {
                "registry_id": string,
                "entries": {"type": "array", "minItems": 2},
                "regional_thickness_limits_m": {"type": "object"},
                "registry_sha256": string,
            },
        ),
        "outfit_assembly_plan.schema.json": _schema(
            "OutfitAssemblyPlan/1",
            ["outfit_id", "garment_ids", "status", "plan_sha256"],
            {
                "outfit_id": string,
                "garment_ids": {"type": "array", "minItems": 1},
                "status": {"enum": ["ACCEPTED", "REJECTED_ATOMIC"]},
                "plan_sha256": string,
            },
        ),
        "body_occlusion_mask.schema.json": _schema(
            "BodyOcclusionMask/1",
            ["source_triangle_count", "hidden_triangle_count", "mask_pass", "receipt_sha256"],
            {
                "source_triangle_count": {"type": "integer", "minimum": 1},
                "hidden_triangle_count": {"type": "integer", "minimum": 1},
                "mask_pass": {"const": True},
                "receipt_sha256": string,
            },
        ),
        "outfit_transaction_receipt.schema.json": _schema(
            "OutfitTransactionSuiteReceipt/1",
            ["compatible_commit_pass", "incompatible_rejection_pass", "unequip_pass", "receipt_sha256"],
            {
                "compatible_commit_pass": {"const": True},
                "incompatible_rejection_pass": {"const": True},
                "unequip_pass": {"const": True},
                "receipt_sha256": string,
            },
        ),
        "outfit_runtime_receipt.schema.json": _schema(
            "GodotOutfitRuntimeReceipt/1",
            ["consumer_pass", "compatible_outfit_pass", "atomic_rejection_pass"],
            {
                "consumer_pass": {"const": True},
                "compatible_outfit_pass": {"const": True},
                "atomic_rejection_pass": {"const": True},
            },
        ),
    }
