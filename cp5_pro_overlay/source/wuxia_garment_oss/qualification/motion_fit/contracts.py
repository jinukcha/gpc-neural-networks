"""JSON schemas for CP5 motion-fit suite and map qualification products."""
from __future__ import annotations


def _object(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def motion_fit_schemas() -> dict[str, dict]:
    string = {"type": "string", "minLength": 1}
    number = {"type": "number"}
    return {
        "motion_fit_suite.schema.json": _object(
            "MotionFitSuite/1",
            ["suite_id", "poses", "suite_sha256"],
            {
                "suite_id": string,
                "poses": {"type": "array", "minItems": 10, "maxItems": 10},
                "frames_per_pose": {"type": "integer", "minimum": 4},
                "suite_sha256": string,
            },
        ),
        "pose_fit_qualification_receipt.schema.json": _object(
            "PoseFitQualificationReceipt/1",
            ["pose_id", "material_id", "metrics", "gates", "pose_pass", "receipt_sha256"],
            {
                "pose_id": string,
                "material_id": string,
                "metrics": {"type": "object"},
                "gates": {"type": "object"},
                "pose_pass": {"type": "boolean"},
                "receipt_sha256": string,
            },
        ),
        "fit_qualification_receipt_v2.schema.json": _object(
            "FitQualificationReceipt/2",
            ["suite", "material_ids", "scenario_count", "motion_fit_pass", "receipt_sha256"],
            {
                "suite": {"type": "object"},
                "material_ids": {"type": "array", "minItems": 3, "maxItems": 3},
                "scenario_count": {"const": 30},
                "motion_fit_pass": {"type": "boolean"},
                "product_acceptance": {"type": "boolean"},
                "receipt_sha256": string,
            },
        ),
        "motion_fit_map_package.schema.json": _object(
            "MotionFitMapPackage/1",
            ["pose_id", "material_id", "archive_path", "array_shapes", "archive_sha256"],
            {
                "pose_id": string,
                "material_id": string,
                "archive_path": string,
                "array_shapes": {"type": "object", "minProperties": 8},
                "archive_sha256": string,
            },
        ),
    }
