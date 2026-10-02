"""Canonical CP0 modular component, interface, and recipe fixtures."""
from __future__ import annotations

from wuxia_garment_oss.pattern_assembly.interface import ComponentInterfaceSpec, InterfaceEndpoint
from wuxia_garment_oss.pattern_assembly.recipe import ComponentInstance, GarmentAssemblyRecipe
from wuxia_garment_oss.pattern_components.model import BoundarySpec, PatternComponentDefinition
from wuxia_garment_oss.pattern_components.registry import PatternComponentRegistry


def _boundary(
    boundary_id: str,
    semantic_role: str,
    interface_family: str,
    orientation: str,
    disposition: str,
    notches: tuple[str, ...] = (),
) -> BoundarySpec:
    return BoundarySpec(
        boundary_id,
        semantic_role,
        interface_family,
        "CUBIC_BEZIER" if "armhole" in boundary_id or "cap" in boundary_id or "neckline" in boundary_id else "LINE",
        orientation,
        disposition,
        notches,
        (),
    )


def _bodice_front() -> PatternComponentDefinition:
    boundaries = (
        _boundary("shoulder_left", "LEFT_SHOULDER_FRONT", "SHOULDER_LEFT", "FORWARD", "SEWN", ("SHOULDER_LEFT",)),
        _boundary("shoulder_right", "RIGHT_SHOULDER_FRONT", "SHOULDER_RIGHT", "FORWARD", "SEWN", ("SHOULDER_RIGHT",)),
        _boundary("side_left", "LEFT_SIDE_FRONT", "SIDE_LEFT", "FORWARD", "SEWN"),
        _boundary("side_right", "RIGHT_SIDE_FRONT", "SIDE_RIGHT", "FORWARD", "SEWN"),
        _boundary("armhole_left_front", "LEFT_ARMHOLE_FRONT", "ARMHOLE_FRONT", "FORWARD", "SEWN", ("LEFT_FRONT_PITCH", "LEFT_SHOULDER_POINT")),
        _boundary("armhole_right_front", "RIGHT_ARMHOLE_FRONT", "ARMHOLE_FRONT", "FORWARD", "SEWN", ("RIGHT_FRONT_PITCH", "RIGHT_SHOULDER_POINT")),
        _boundary("neckline_front", "FRONT_NECKLINE", "NECKLINE_FRONT", "FORWARD", "FINISHED"),
        _boundary("hem_front", "FRONT_HEM", "HEM_FRONT", "FORWARD", "FINISHED"),
        _boundary("center_front_fold", "CENTER_FRONT_FOLD", "CENTER_FRONT", "FORWARD", "FOLD"),
    )
    return PatternComponentDefinition(
        "BODICE_FRONT_BASIC", 1, "BODICE", "BASIC_TUNIC_BLOCK", "SHELL", "EXACT_2D_PATTERN",
        boundaries,
        ("CF_NECK", "CF_HEM", "LEFT_UNDERARM", "RIGHT_UNDERARM", "LEFT_WAIST", "RIGHT_WAIST"),
        ("GRAINLINE", "WAIST_REFERENCE", "CHEST_REFERENCE"),
        ("body.front_chest_arc", "body.front_waist_arc", "design.bodice_length"),
        tuple(item.boundary_id for item in boundaries if item.disposition == "SEWN"),
        "NONE",
        "GUIDED_ONLY",
    )


def _bodice_back() -> PatternComponentDefinition:
    boundaries = (
        _boundary("shoulder_left", "LEFT_SHOULDER_BACK", "SHOULDER_LEFT", "REVERSE", "SEWN", ("SHOULDER_LEFT",)),
        _boundary("shoulder_right", "RIGHT_SHOULDER_BACK", "SHOULDER_RIGHT", "REVERSE", "SEWN", ("SHOULDER_RIGHT",)),
        _boundary("side_left", "LEFT_SIDE_BACK", "SIDE_LEFT", "REVERSE", "SEWN"),
        _boundary("side_right", "RIGHT_SIDE_BACK", "SIDE_RIGHT", "REVERSE", "SEWN"),
        _boundary("armhole_left_back", "LEFT_ARMHOLE_BACK", "ARMHOLE_BACK", "FORWARD", "SEWN", ("LEFT_BACK_PITCH", "LEFT_SHOULDER_POINT")),
        _boundary("armhole_right_back", "RIGHT_ARMHOLE_BACK", "ARMHOLE_BACK", "FORWARD", "SEWN", ("RIGHT_BACK_PITCH", "RIGHT_SHOULDER_POINT")),
        _boundary("neckline_back", "BACK_NECKLINE", "NECKLINE_BACK", "FORWARD", "FINISHED"),
        _boundary("hem_back", "BACK_HEM", "HEM_BACK", "FORWARD", "FINISHED"),
        _boundary("center_back_fold", "CENTER_BACK_FOLD", "CENTER_BACK", "FORWARD", "FOLD"),
    )
    return PatternComponentDefinition(
        "BODICE_BACK_BASIC", 1, "BODICE", "BASIC_TUNIC_BLOCK", "SHELL", "EXACT_2D_PATTERN",
        boundaries,
        ("CB_NECK", "CB_HEM", "LEFT_UNDERARM", "RIGHT_UNDERARM", "LEFT_WAIST", "RIGHT_WAIST"),
        ("GRAINLINE", "WAIST_REFERENCE", "BACK_BALANCE_REFERENCE"),
        ("body.back_chest_arc", "body.back_waist_arc", "design.bodice_length"),
        tuple(item.boundary_id for item in boundaries if item.disposition == "SEWN"),
        "NONE",
        "GUIDED_ONLY",
    )


def _set_in_sleeve() -> PatternComponentDefinition:
    boundaries = (
        _boundary("cap_front", "SLEEVE_CAP_FRONT", "ARMHOLE_FRONT", "REVERSE", "SEWN", ("FRONT_PITCH", "SHOULDER_POINT")),
        _boundary("cap_back", "SLEEVE_CAP_BACK", "ARMHOLE_BACK", "REVERSE", "SEWN", ("BACK_PITCH", "SHOULDER_POINT")),
        _boundary("underarm_front", "SLEEVE_UNDERARM_FRONT", "SLEEVE_UNDERARM", "FORWARD", "SEWN"),
        _boundary("underarm_back", "SLEEVE_UNDERARM_BACK", "SLEEVE_UNDERARM", "REVERSE", "SEWN"),
        _boundary("wrist", "SLEEVE_WRIST", "WRIST_FINISH", "FORWARD", "FINISHED"),
    )
    return PatternComponentDefinition(
        "SET_IN_SLEEVE_BASIC", 1, "SLEEVE", "SET_IN_ONE_PIECE", "SHELL", "EXACT_2D_PATTERN",
        boundaries,
        ("SHOULDER_POINT", "FRONT_PITCH", "BACK_PITCH", "BICEPS_FRONT", "BICEPS_BACK", "ELBOW_FRONT", "ELBOW_BACK", "WRIST_FRONT", "WRIST_BACK"),
        ("GRAINLINE", "BICEPS_LINE", "ELBOW_LINE", "WRIST_LINE", "UNDERARM_SEAM"),
        ("body.arm_length", "body.upper_arm_circumference", "body.elbow_circumference", "body.wrist_circumference", "design.cap_ease"),
        tuple(item.boundary_id for item in boundaries if item.disposition == "SEWN"),
        "MIRROR_ALLOWED",
        "GUIDED_ONLY",
    )


def component_registry() -> PatternComponentRegistry:
    return PatternComponentRegistry("R1C_CP0_PATTERN_COMPONENTS", (_bodice_front(), _bodice_back(), _set_in_sleeve()))


def _interface(interface_id, instance_a, boundary_a, instance_b, boundary_b, role, notch="NONE"):
    return ComponentInterfaceSpec(
        interface_id,
        InterfaceEndpoint(instance_a, boundary_a),
        InterfaceEndpoint(instance_b, boundary_b),
        role,
        "OPPOSED",
        "BOUNDED_EASE" if "cap" in interface_id else "EQUAL",
        1.0,
        1.08 if "cap" in interface_id else 1.0,
        notch,
        "MATCHED",
        "MATERIAL_DERIVED" if "cap" in interface_id else "NONE",
        (),
    )


def component_interfaces() -> tuple[ComponentInterfaceSpec, ...]:
    return (
        _interface("shoulder_left", "bodice_front", "shoulder_left", "bodice_back", "shoulder_left", "SHOULDER", "EXACT_ID"),
        _interface("shoulder_right", "bodice_front", "shoulder_right", "bodice_back", "shoulder_right", "SHOULDER", "EXACT_ID"),
        _interface("side_left", "bodice_front", "side_left", "bodice_back", "side_left", "SIDE_SEAM"),
        _interface("side_right", "bodice_front", "side_right", "bodice_back", "side_right", "SIDE_SEAM"),
        _interface("left_cap_front", "sleeve_left", "cap_front", "bodice_front", "armhole_left_front", "SLEEVE_CAP", "NORMALIZED_ARC"),
        _interface("left_cap_back", "sleeve_left", "cap_back", "bodice_back", "armhole_left_back", "SLEEVE_CAP", "NORMALIZED_ARC"),
        _interface("right_cap_front", "sleeve_right", "cap_front", "bodice_front", "armhole_right_front", "SLEEVE_CAP", "NORMALIZED_ARC"),
        _interface("right_cap_back", "sleeve_right", "cap_back", "bodice_back", "armhole_right_back", "SLEEVE_CAP", "NORMALIZED_ARC"),
        _interface("left_underarm", "sleeve_left", "underarm_front", "sleeve_left", "underarm_back", "UNDERARM_SEAM"),
        _interface("right_underarm", "sleeve_right", "underarm_front", "sleeve_right", "underarm_back", "UNDERARM_SEAM"),
    )


def assembly_recipe(include_right_underarm: bool = True) -> GarmentAssemblyRecipe:
    interfaces = [item.interface_id for item in component_interfaces()]
    if not include_right_underarm:
        interfaces.remove("right_underarm")
    return GarmentAssemblyRecipe(
        "SLEEVED_TUNIC_MODULAR_REFERENCE",
        "SLEEVED_TUNIC",
        (
            ComponentInstance("bodice_front", "BODICE_FRONT_BASIC", None, {}),
            ComponentInstance("bodice_back", "BODICE_BACK_BASIC", None, {}),
            ComponentInstance("sleeve_left", "SET_IN_SLEEVE_BASIC", None, {}),
            ComponentInstance("sleeve_right", "SET_IN_SLEEVE_BASIC", "sleeve_left", {}),
        ),
        tuple(interfaces),
        ("SHELL",),
        ("STRICT", "GUIDED", "SAFE_AUTO"),
        "HOLD",
        "R1C_TECHNICAL_PROFILE_V1",
        "R1C_VISUAL_PROFILE_V1",
    )
