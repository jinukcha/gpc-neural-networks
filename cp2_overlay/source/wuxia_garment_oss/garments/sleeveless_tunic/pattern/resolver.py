"""Resolve publishable CP1 selections into CP2 tunic POM and landmarks."""
from __future__ import annotations

from ....sizing.body_profile.model import BodyMeasurementProfile, BodyMeasurements
from ....sizing.instance.model import canonical_sha256
from ....sizing.pattern.model import PatternParameterPackage, PUBLISHABLE_ADMISSIONS
from ....sizing.size_table.model import GarmentSizeTable
from .landmarks import compile_panels
from .pom import resolve_poms


def _measurement_authority(
    receipt: dict,
    table: GarmentSizeTable,
    body: BodyMeasurementProfile | None,
) -> tuple[BodyMeasurements, str]:
    mode = str(receipt["mode"])
    selected_size = str(receipt["selected_size_id"])
    if mode == "STANDARD_SIZE":
        return table.entry(selected_size).target, f"SIZE_TABLE:{table.size_table_id}:{selected_size}"
    if body is None:
        raise ValueError(f"{mode} requires body measurements")
    body.validate()
    return body.measurements, f"BODY_PROFILE:{body.body_id}"


def resolve_pattern_parameters(
    receipt: dict,
    table: GarmentSizeTable,
    body: BodyMeasurementProfile | None,
) -> dict:
    admission = str(receipt["admission"])
    if admission not in PUBLISHABLE_ADMISSIONS:
        raise ValueError(f"selection is blocked for CP2 pattern publication: {admission}")
    measurements, source_id = _measurement_authority(receipt, table, body)
    poms = resolve_poms(measurements)
    panels, seams = compile_panels(poms)
    package = PatternParameterPackage(
        request_id=str(receipt["request_id"]),
        garment_design_id="CP2B_SLEEVELESS_LONG_TUNIC",
        topology_class="SLEEVELESS_TUNIC_4_PANEL_V1",
        sizing_mode=str(receipt["mode"]),
        selected_size_id=str(receipt["selected_size_id"]),
        shape_block=str(receipt["shape_block"]),
        height_block=str(receipt["height_block"]),
        admission=admission,
        selection_receipt_sha256=str(receipt["receipt_sha256"]),
        poms=poms,
        panels=panels,
        seam_pairs=seams,
        grade_plan=list(receipt.get("grade_plan", [])),
        custom_alteration_plan=list(receipt.get("custom_alteration_plan", [])),
        provenance={
            "measurement_authority": source_id,
            "measurement_sha256": canonical_sha256(measurements.to_dict()),
            "size_table_id": table.size_table_id,
            "size_table_sha256": canonical_sha256(table.to_dict()),
            "fit_profile": "REGULAR_TUNIC_V1",
            "material_profile": "MEDIUM_WOVEN_V1",
            "layer_stack": "BASE_BODY_ONLY_V1",
        },
        warnings=list(receipt.get("warnings", [])),
    )
    return package.to_dict()
