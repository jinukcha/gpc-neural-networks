"""JSON Schema publication for the global sizing contracts."""
from __future__ import annotations


def _object_schema(contract: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": contract,
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": contract}, **properties},
        "additionalProperties": True,
    }


def contract_schemas() -> dict[str, dict]:
    number = {"type": "number"}
    identity = {"type": "string", "minLength": 1}
    return {
        "body_measurement_profile.schema.json": _object_schema(
            "BodyMeasurementProfile/1", ["body_id", "measurements"],
            {"body_id": identity, "measurements": {"type": "object"}},
        ),
        "garment_size_table.schema.json": _object_schema(
            "GarmentSizeTable/1", ["size_table_id", "base_size_id", "sizes"],
            {"size_table_id": identity, "base_size_id": identity, "sizes": {"type": "array", "minItems": 1}},
        ),
        "grade_rule_set.schema.json": _object_schema(
            "GradeRuleSet/1", ["grade_rules"], {"grade_rules": {"type": "array"}},
        ),
        "garment_fit_profile.schema.json": _object_schema(
            "GarmentFitProfile/1", ["fit_class"], {"fit_class": identity},
        ),
        "material_sizing_profile.schema.json": _object_schema(
            "MaterialSizingProfile/1", ["material_id"], {"material_id": identity, "thickness_m": number},
        ),
        "layer_stack_profile.schema.json": _object_schema(
            "LayerStackProfile/1", ["layer_stack_id"], {"layer_stack_id": identity},
        ),
        "garment_sizing_request.schema.json": _object_schema(
            "GarmentSizingRequest/1", ["request_id", "mode"], {"request_id": identity, "mode": identity},
        ),
        "sized_pattern_instance.schema.json": _object_schema(
            "SizedPatternInstance/1", ["instance_id", "selection"], {"instance_id": identity, "selection": {"type": "object"}},
        ),
        "garment_selection_receipt.schema.json": _object_schema(
            "GarmentSelectionReceipt/1", ["request_id", "admission"], {"request_id": identity, "admission": identity},
        ),
    }
