"""Success and atomic-rejection fixtures for CP1."""
from __future__ import annotations

import hashlib
import math

from wuxia_garment_oss.proportions.model.parameter import BoundSpec, ParameterDefinition
from wuxia_garment_oss.proportions.model.reference import ParameterReference
from wuxia_garment_oss.proportions.resolution.context import ParameterResolutionContext


def _source_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reference(scope: str, path: str, owner: str, quantity: str, unit: str, value: float) -> ParameterReference:
    return ParameterReference(
        scope,
        path,
        owner,
        quantity,
        unit,
        value,
        _source_hash(f"{scope}:{owner}"),
        1,
    )


def canonical_context() -> ParameterResolutionContext:
    references = (
        _reference("BODY_RELATIVE", "arm_length", "REFERENCE_BODY", "LENGTH", "m", 0.62),
        _reference("BLOCK_RELATIVE", "armscye_depth", "BASIC_TUNIC_BLOCK", "LENGTH", "m", 0.235),
        _reference("BLOCK_RELATIVE", "front_panel_width", "BASIC_TUNIC_BLOCK", "LENGTH", "m", 0.34),
        _reference("COMPONENT_RELATIVE", "neckline_depth", "BODICE_FRONT_BASIC", "LENGTH", "m", 0.085),
        _reference("COMPONENT_RELATIVE", "maximum_cap_ease", "SET_IN_SLEEVE_BASIC", "LENGTH", "m", 0.028),
        _reference("COMPONENT_RELATIVE", "turn_of_cloth_max", "FACING_BASIC", "LENGTH", "mm", 3.0),
        _reference("BOUNDARY_RELATIVE", "cap_front_arc_length", "SET_IN_SLEEVE_BASIC.cap_front", "LENGTH", "m", 0.315),
        _reference("BOUNDARY_RELATIVE", "armhole_total_length", "BODICE_ARMHOLE_PAIR", "LENGTH", "m", 0.62),
        _reference("MATERIAL_RELATIVE", "thickness", "WOOL_TWILL_MEDIUM_REFERENCE", "LENGTH", "mm", 0.78),
        _reference("MATERIAL_RELATIVE", "minimum_cap_ease", "WOOL_TWILL_MEDIUM_REFERENCE", "LENGTH", "mm", 8.0),
        _reference("MATERIAL_RELATIVE", "turn_of_cloth_min", "WOOL_TWILL_MEDIUM_REFERENCE", "LENGTH", "mm", 0.6),
        _reference("MATERIAL_RELATIVE", "areal_density", "WOOL_TWILL_MEDIUM_REFERENCE", "MASS_PER_AREA", "g/m2", 265.0),
    )
    return ParameterResolutionContext("R1C_CP1_CANONICAL_CONTEXT", references)


def canonical_definitions() -> tuple[ParameterDefinition, ...]:
    return (
        ParameterDefinition("seam_allowance", "ABSOLUTE", "LENGTH", "mm", 12.0, bounds=BoundSpec(0.006, 0.025, "REJECT")),
        ParameterDefinition("sleeve_length", "RELATIVE", "LENGTH", "m", reference_scope="BODY_RELATIVE", reference_path="arm_length", ratio=0.96, bounds=BoundSpec(0.42, 0.68, "REJECT"), owner_component_id="SET_IN_SLEEVE_BASIC"),
        ParameterDefinition("cap_height", "RELATIVE", "LENGTH", "m", reference_scope="BLOCK_RELATIVE", reference_path="armscye_depth", ratio=0.47, bounds=BoundSpec(0.09, 0.16, "REJECT"), owner_component_id="SET_IN_SLEEVE_BASIC"),
        ParameterDefinition("collar_depth", "RELATIVE", "LENGTH", "m", reference_scope="COMPONENT_RELATIVE", reference_path="neckline_depth", ratio=0.28, bounds=BoundSpec(0.015, 0.05, "REJECT"), owner_component_id="BODICE_FRONT_BASIC"),
        ParameterDefinition("front_pitch_notch", "RELATIVE", "LENGTH", "m", reference_scope="BOUNDARY_RELATIVE", reference_path="cap_front_arc_length", ratio=0.35, bounds=BoundSpec(0.08, 0.16, "REJECT"), owner_component_id="SET_IN_SLEEVE_BASIC"),
        ParameterDefinition("interfacing_offset", "RELATIVE", "LENGTH", "m", reference_scope="MATERIAL_RELATIVE", reference_path="thickness", ratio=2.0, bounds=BoundSpec(0.0005, 0.004, "REJECT"), owner_component_id="FACING_BASIC"),
        ParameterDefinition("cap_ease_ratio", "ABSOLUTE", "DIMENSIONLESS", "ratio", 1.055, bounds=BoundSpec(1.01, 1.12, "REJECT"), owner_component_id="SET_IN_SLEEVE_BASIC"),
        ParameterDefinition("cap_ease", "AUTO_DERIVED", "LENGTH", "m", expression="boundary.armhole_total_length * (param.cap_ease_ratio - 1.0)", bounds=BoundSpec(0.008, 0.028, "SAFE_CLAMP"), owner_component_id="SET_IN_SLEEVE_BASIC"),
        ParameterDefinition("turn_of_cloth", "AUTO_DERIVED", "LENGTH", "m", expression="clamp(material.thickness * 1.5, material.turn_of_cloth_min, component.turn_of_cloth_max)", bounds=BoundSpec(0.0006, 0.003, "REJECT"), owner_component_id="FACING_BASIC"),
        ParameterDefinition("pocket_width", "RELATIVE", "LENGTH", "m", reference_scope="BLOCK_RELATIVE", reference_path="front_panel_width", ratio=0.34, bounds=BoundSpec(0.08, 0.18, "REJECT"), owner_component_id="POCKET_PATCH_BASIC"),
        ParameterDefinition("pocket_height", "ABSOLUTE", "LENGTH", "cm", 16.0, bounds=BoundSpec(0.10, 0.24, "REJECT"), owner_component_id="POCKET_PATCH_BASIC"),
        ParameterDefinition("pocket_area", "AUTO_DERIVED", "AREA", "m2", expression="param.pocket_width * param.pocket_height", bounds=BoundSpec(0.008, 0.04, "REJECT"), owner_component_id="POCKET_PATCH_BASIC"),
        ParameterDefinition("arm_rest_pitch", "ABSOLUTE", "ANGLE", "deg", 12.0, bounds=BoundSpec(0.0, 0.5, "REJECT"), owner_component_id="SET_IN_SLEEVE_BASIC"),
        ParameterDefinition("material_mass", "RELATIVE", "MASS_PER_AREA", "kg/m2", reference_scope="MATERIAL_RELATIVE", reference_path="areal_density", ratio=1.0, bounds=BoundSpec(0.08, 0.6, "REJECT"), owner_component_id="GLOBAL"),
    )


def _cycle_fixture() -> tuple[ParameterDefinition, ...]:
    return (
        ParameterDefinition("cycle_a", "AUTO_DERIVED", "DIMENSIONLESS", "1", expression="param.cycle_b + 1.0"),
        ParameterDefinition("cycle_b", "AUTO_DERIVED", "DIMENSIONLESS", "1", expression="param.cycle_a + 1.0"),
    )


def rejection_fixtures() -> dict[str, tuple[ParameterDefinition, ...]]:
    return {
        "missing_reference": (
            ParameterDefinition("missing", "RELATIVE", "LENGTH", "m", reference_scope="BODY_RELATIVE", reference_path="unknown_length", ratio=1.0),
        ),
        "quantity_mismatch": (
            ParameterDefinition("wrong_quantity", "RELATIVE", "ANGLE", "rad", reference_scope="BODY_RELATIVE", reference_path="arm_length", ratio=1.0),
        ),
        "expression_quantity_mismatch": (
            ParameterDefinition("wrong_expression", "AUTO_DERIVED", "LENGTH", "m", expression="body.arm_length + material.areal_density"),
        ),
        "unit_mismatch": (
            ParameterDefinition("bad_unit", "ABSOLUTE", "ANGLE", "m", 1.0),
        ),
        "dependency_cycle": _cycle_fixture(),
        "non_finite_ratio": (
            ParameterDefinition("nan_ratio", "RELATIVE", "LENGTH", "m", reference_scope="BODY_RELATIVE", reference_path="arm_length", ratio=math.nan),
        ),
        "hard_bound": (
            ParameterDefinition("too_long", "ABSOLUTE", "LENGTH", "m", 1.0, bounds=BoundSpec(0.1, 0.5, "REJECT")),
        ),
        "alternate_component": (
            ParameterDefinition("cap_height", "ABSOLUTE", "LENGTH", "m", 0.40, bounds=BoundSpec(0.08, 0.30, "ALTERNATE_COMPONENT_REQUIRED")),
        ),
    }
