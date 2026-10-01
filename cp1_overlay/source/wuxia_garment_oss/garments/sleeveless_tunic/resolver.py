"""CP1 sleeveless-tunic selection and alteration pilot."""
from __future__ import annotations

from ...sizing.body_profile.model import BodyMeasurementProfile
from ...sizing.instance.model import canonical_sha256
from ...sizing.selection.resolver import SelectionRequest, resolve_selection
from ...sizing.size_table.model import GarmentSizeTable


CHEST_EASE_M = 0.099
WAIST_EASE_M = 0.120
HIP_EASE_M = 0.120


def _finished_measurements(body: BodyMeasurementProfile | None, table: GarmentSizeTable, size_id: str) -> dict:
    measurements = body.measurements if body is not None else table.entry(size_id).target
    return {
        "chest_circumference_m": measurements.chest_circumference + CHEST_EASE_M,
        "waist_circumference_m": measurements.waist_circumference + WAIST_EASE_M,
        "hip_circumference_m": measurements.hip_circumference + HIP_EASE_M,
        "shoulder_width_m": measurements.shoulder_width,
        "front_torso_length_m": measurements.front_torso_length,
        "back_torso_length_m": measurements.back_torso_length,
        "armscye_depth_m": measurements.armscye_depth,
    }


def resolve_tunic_instance(
    request: SelectionRequest,
    table: GarmentSizeTable,
    body: BodyMeasurementProfile | None,
) -> tuple[dict, dict]:
    receipt = resolve_selection(request, table, body)
    receipt_payload = receipt.to_dict()
    payload = {
        "contract": "SizedPatternInstance/1",
        "garment_design_id": "CP2B_SLEEVELESS_LONG_TUNIC",
        "topology_class": "SLEEVELESS_TUNIC_4_PANEL_V1",
        "sizing_mode": request.mode,
        "selection": {
            "selected_size_id": receipt.selected_size_id,
            "recommended_size_id": receipt.recommended_size_id,
            "shape_block": receipt.shape_block,
            "height_block": receipt.height_block,
            "admission": receipt.admission,
        },
        "resolved_finished_measurements": _finished_measurements(
            body, table, receipt.selected_size_id
        ),
        "grade_plan": receipt.grade_plan,
        "custom_alteration_plan": receipt.custom_alteration_plan,
        "pattern_compile": "DEFERRED_CP2",
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "publishable": receipt.admission in {"NORMAL_GRADE", "CUSTOM_ALTERATION"},
        "selection_receipt_sha256": receipt_payload["receipt_sha256"],
    }
    payload["instance_id"] = canonical_sha256(payload)
    return payload, receipt_payload
