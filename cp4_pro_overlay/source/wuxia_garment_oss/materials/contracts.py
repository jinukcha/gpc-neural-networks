"""JSON schemas for professional material calibration products."""
from __future__ import annotations


def _object_schema(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def material_schemas() -> dict[str, dict]:
    string = {"type": "string", "minLength": 1}
    number = {"type": "number"}
    return {
        "material_measurement_set.schema.json": _object_schema(
            "MaterialMeasurementSet/1",
            ["material_id", "provenance", "series", "measurement_set_sha256"],
            {
                "material_id": string,
                "provenance": {"type": "object"},
                "series": {"type": "array", "minItems": 10, "maxItems": 10},
                "areal_density_kg_m2": number,
                "thickness_m": number,
                "measurement_set_sha256": string,
            },
        ),
        "calibration_experiment.schema.json": _object_schema(
            "CalibrationExperiment/1",
            ["material_id", "experiment_count", "calibration_pass", "receipt_sha256"],
            {
                "material_id": string,
                "experiment_count": {"const": 10},
                "calibration_pass": {"type": "boolean"},
                "maximum_normalized_rmse": number,
                "maximum_sigma_error": number,
                "receipt_sha256": string,
            },
        ),
        "material_sizing_profile_v2.schema.json": _object_schema(
            "MaterialSizingProfile/2",
            ["material_id", "recommended_meshing_edge_m", "profile_sha256"],
            {
                "material_id": string,
                "recommended_meshing_edge_m": number,
                "warp_shrinkage_ratio": number,
                "weft_shrinkage_ratio": number,
                "profile_sha256": string,
            },
        ),
        "warp_material_profile.schema.json": _object_schema(
            "WarpMaterialProfile/1",
            ["material_id", "parameters", "calibration_metrics", "profile_sha256"],
            {
                "material_id": string,
                "parameters": {"type": "object", "minProperties": 10},
                "calibration_metrics": {"type": "object", "minProperties": 10},
                "profile_sha256": string,
            },
        ),
        "material_backend_parity_receipt.schema.json": _object_schema(
            "MaterialBackendParityReceipt/1",
            ["material_id", "runtime", "parity_pass", "receipt_sha256"],
            {
                "material_id": string,
                "runtime": {"type": "object"},
                "parity_pass": {"type": "boolean"},
                "maximum_relative_response_error": number,
                "maximum_relative_metric_error": number,
                "receipt_sha256": string,
            },
        ),
    }
