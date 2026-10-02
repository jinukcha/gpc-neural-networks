"""CP2 component definitions, geometry library, interfaces, and canonical recipe."""
from __future__ import annotations

from wuxia_garment_oss.pattern_assembly.interface import ComponentInterfaceSpec, InterfaceEndpoint
from wuxia_garment_oss.pattern_assembly.recipe import ComponentInstance, GarmentAssemblyRecipe
from wuxia_garment_oss.pattern_components.model import BoundarySpec, PatternComponentDefinition, canonical_sha256
from wuxia_garment_oss.pattern_components.registry import PatternComponentRegistry
from wuxia_garment_oss.pattern_geometry.accessories import build_collar, build_cuff, build_gore
from wuxia_garment_oss.pattern_geometry.bodice import build_bodice
from wuxia_garment_oss.pattern_geometry.curve import boundary_length
from wuxia_garment_oss.pattern_geometry.inputs import GeometryInputs
from wuxia_garment_oss.pattern_geometry.sleeve import build_left_sleeve, mirror_sleeve


def _boundary(boundary_id, role, family, curve, orientation, disposition, notches=()):
    return BoundarySpec(boundary_id, role, family, curve, orientation, disposition, tuple(notches), ())


def _bodice_definition(front: bool) -> PatternComponentDefinition:
    suffix = "FRONT" if front else "BACK"
    orientation = "FORWARD" if front else "REVERSE"
    armhole_family = "ARMHOLE_FRONT" if front else "ARMHOLE_BACK"
    boundaries = (
        _boundary(f"hem_{suffix.lower()}", f"{suffix}_HEM", f"HEM_{suffix}", "LINE", "FORWARD", "FINISHED"),
        _boundary("side_left", f"LEFT_SIDE_{suffix}", "SIDE_LEFT", "LINE", orientation, "SEWN"),
        _boundary("side_right", f"RIGHT_SIDE_{suffix}", "SIDE_RIGHT", "LINE", orientation, "SEWN"),
        _boundary(f"armhole_left_{suffix.lower()}", f"LEFT_ARMHOLE_{suffix}", armhole_family, "CUBIC_BEZIER", "FORWARD", "SEWN", ("PITCH", "SHOULDER_POINT")),
        _boundary(f"armhole_right_{suffix.lower()}", f"RIGHT_ARMHOLE_{suffix}", armhole_family, "CUBIC_BEZIER", "FORWARD", "SEWN", ("PITCH", "SHOULDER_POINT")),
        _boundary("shoulder_left", f"LEFT_SHOULDER_{suffix}", "SHOULDER_LEFT", "LINE", orientation, "SEWN", ("SHOULDER_POINT",)),
        _boundary("shoulder_right", f"RIGHT_SHOULDER_{suffix}", "SHOULDER_RIGHT", "LINE", orientation, "SEWN", ("SHOULDER_POINT",)),
        _boundary(f"neckline_{suffix.lower()}", f"{suffix}_NECKLINE", f"NECKLINE_{suffix}", "CUBIC_BEZIER", "FORWARD", "SEWN"),
        _boundary(f"center_{suffix.lower()}_fold", f"CENTER_{suffix}_FOLD", f"CENTER_{suffix}", "LINE", "FORWARD", "FOLD"),
    )
    return PatternComponentDefinition(
        f"BODICE_{suffix}_BASIC", 2, "BODICE", "BASIC_TUNIC_BLOCK", "SHELL", "EXACT_2D_PATTERN",
        boundaries,
        (f"{suffix}_CENTER_NECK", f"{suffix}_LEFT_UNDERARM", f"{suffix}_RIGHT_UNDERARM"),
        ("GRAINLINE", "WAIST_REFERENCE", "CENTER_FOLD"),
        ("block.front_panel_width", "block.armscye_depth", "component.neckline_depth"),
        tuple(item.boundary_id for item in boundaries if item.disposition == "SEWN"),
        "NONE",
        "GUIDED_ONLY",
    )


def _sleeve_definition() -> PatternComponentDefinition:
    boundaries = (
        _boundary("cap_front", "SLEEVE_CAP_FRONT", "ARMHOLE_FRONT", "CUBIC_BEZIER", "REVERSE", "SEWN", ("FRONT_PITCH", "SHOULDER_POINT")),
        _boundary("cap_back", "SLEEVE_CAP_BACK", "ARMHOLE_BACK", "CUBIC_BEZIER", "REVERSE", "SEWN", ("BACK_PITCH", "SHOULDER_POINT")),
        _boundary("underarm_front", "SLEEVE_UNDERARM_FRONT", "SLEEVE_UNDERARM", "LINE", "FORWARD", "SEWN"),
        _boundary("underarm_back", "SLEEVE_UNDERARM_BACK", "SLEEVE_UNDERARM", "LINE", "REVERSE", "SEWN"),
        _boundary("wrist", "SLEEVE_WRIST", "WRIST_ATTACH", "LINE", "FORWARD", "SEWN"),
    )
    return PatternComponentDefinition(
        "SET_IN_SLEEVE_BASIC", 2, "SLEEVE", "SET_IN_ONE_PIECE", "SHELL", "EXACT_2D_PATTERN",
        boundaries,
        ("SHOULDER_POINT", "FRONT_PITCH", "BACK_PITCH", "FRONT_ELBOW", "BACK_ELBOW"),
        ("GRAINLINE", "BICEPS_LINE", "ELBOW_LINE", "UNDERARM_SEAM"),
        ("sleeve_length", "cap_height", "cap_ease", "front_pitch_notch", "arm_rest_pitch"),
        tuple(item.boundary_id for item in boundaries),
        "MIRROR_ALLOWED",
        "GUIDED_ONLY",
    )


def _collar_definition() -> PatternComponentDefinition:
    boundaries = (
        _boundary("front_attach", "COLLAR_FRONT_NECK_ATTACH", "NECKLINE_FRONT", "LINE", "REVERSE", "SEWN"),
        _boundary("back_attach", "COLLAR_BACK_NECK_ATTACH", "NECKLINE_BACK", "LINE", "REVERSE", "SEWN"),
        _boundary("outer", "COLLAR_OUTER_EDGE", "COLLAR_OUTER", "CUBIC_BEZIER", "FORWARD", "FINISHED"),
        _boundary("front_end", "COLLAR_FRONT_END", "COLLAR_END", "LINE", "FORWARD", "FINISHED"),
        _boundary("back_end", "COLLAR_BACK_END", "COLLAR_END", "LINE", "FORWARD", "FINISHED"),
    )
    return PatternComponentDefinition(
        "COLLAR_STAND_BASIC", 1, "COLLAR", "STAND_COLLAR", "SHELL", "EXACT_2D_PATTERN",
        boundaries,
        ("FRONT_END_LOWER", "FRONT_BACK_JUNCTION_LOWER", "BACK_END_LOWER"),
        ("ROLL_LINE",),
        ("collar_depth", "turn_of_cloth"),
        ("front_attach", "back_attach"),
        "NONE",
        "GUIDED_ONLY",
    )


def _cuff_definition() -> PatternComponentDefinition:
    boundaries = (
        _boundary("sleeve_attach", "CUFF_SLEEVE_ATTACH", "WRIST_ATTACH", "LINE", "REVERSE", "SEWN"),
        _boundary("end_left", "CUFF_END_LEFT", "CUFF_END", "LINE", "FORWARD", "SEWN"),
        _boundary("end_right", "CUFF_END_RIGHT", "CUFF_END", "LINE", "REVERSE", "SEWN"),
        _boundary("outer", "CUFF_OUTER_EDGE", "CUFF_OUTER", "LINE", "FORWARD", "FINISHED"),
    )
    return PatternComponentDefinition(
        "CUFF_STRAIGHT_BASIC", 1, "CUFF", "STRAIGHT_CUFF", "SHELL", "EXACT_2D_PATTERN",
        boundaries,
        ("ATTACH_LEFT", "ATTACH_RIGHT", "OUTER_LEFT", "OUTER_RIGHT"),
        ("FOLD_LINE",),
        ("seam_allowance",),
        ("sleeve_attach", "end_left", "end_right"),
        "MIRROR_ALLOWED",
        "GUIDED_ONLY",
    )


def _gore_definition() -> PatternComponentDefinition:
    boundaries = (
        _boundary("front_attach", "GORE_FRONT_ATTACH", "GORE_FRONT", "LINE", "FORWARD", "SEWN"),
        _boundary("back_attach", "GORE_BACK_ATTACH", "GORE_BACK", "LINE", "REVERSE", "SEWN"),
        _boundary("hem", "GORE_HEM", "GORE_HEM", "LINE", "FORWARD", "FINISHED"),
    )
    return PatternComponentDefinition(
        "SIDE_GORE_BASIC", 1, "GORE", "TRIANGULAR_SIDE_GORE", "SHELL", "EXACT_2D_PATTERN",
        boundaries,
        ("APEX", "FRONT_HEM", "BACK_HEM"),
        ("GRAINLINE",),
        ("block.front_panel_width", "body.arm_length"),
        ("front_attach", "back_attach"),
        "MIRROR_ALLOWED",
        "GUIDED_ONLY",
    )


def component_registry() -> PatternComponentRegistry:
    return PatternComponentRegistry(
        "R1C_CP2_COMPONENT_LIBRARY",
        (_bodice_definition(True), _bodice_definition(False), _sleeve_definition(), _collar_definition(), _cuff_definition(), _gore_definition()),
    )


def build_geometry_library(inputs: GeometryInputs) -> tuple[dict, dict[str, object]]:
    front = build_bodice(inputs, True)
    back = build_bodice(inputs, False)
    front_armhole = boundary_length(front.boundary("armhole_left_front"), front.segment_map())
    back_armhole = boundary_length(back.boundary("armhole_left_back"), back.segment_map())
    left_sleeve = build_left_sleeve(inputs, front_armhole, back_armhole)
    right_sleeve = mirror_sleeve(left_sleeve)
    wrist = boundary_length(left_sleeve.boundary("wrist"), left_sleeve.segment_map())
    front_neck = boundary_length(front.boundary("neckline_front"), front.segment_map())
    back_neck = boundary_length(back.boundary("neckline_back"), back.segment_map())
    collar = build_collar(inputs, front_neck, back_neck)
    cuff_left = build_cuff(inputs, "cuff_left", wrist, None)
    cuff_right = build_cuff(inputs, "cuff_right", wrist, "cuff_left")
    gore_left = build_gore(inputs, "gore_left", None)
    gore_right = build_gore(inputs, "gore_right", "gore_left")
    authorities = (front, back, left_sleeve, right_sleeve, collar, cuff_left, cuff_right, gore_left, gore_right)
    payload = {
        "contract": "PatternGeometryLibrary/1",
        "library_id": "R1C_CP2_EXACT_GEOMETRY_LIBRARY",
        "resolved_parameter_set_sha256": inputs.resolved_set_sha256,
        "authority_count": len(authorities),
        "authorities": [item.to_dict() for item in authorities],
        "triangulation_executed": False,
        "simulation_executed": False,
    }
    payload["library_sha256"] = canonical_sha256(payload)
    return payload, {item.instance_id: item for item in authorities}


def _interface(interface_id, left_instance, left_boundary, right_instance, right_boundary, role, length_policy="EQUAL", notch_policy="NONE", ratio_max=1.002):
    return ComponentInterfaceSpec(
        interface_id,
        InterfaceEndpoint(left_instance, left_boundary),
        InterfaceEndpoint(right_instance, right_boundary),
        role,
        "OPPOSED",
        length_policy,
        1.0,
        ratio_max,
        notch_policy,
        "MATCHED",
        "MATERIAL_DERIVED" if role in {"SLEEVE_CAP", "COLLAR_ATTACH"} else "NONE",
        (),
    )


def component_interfaces() -> tuple[ComponentInterfaceSpec, ...]:
    return (
        _interface("shoulder_left", "bodice_front", "shoulder_left", "bodice_back", "shoulder_left", "SHOULDER", ratio_max=1.04),
        _interface("shoulder_right", "bodice_front", "shoulder_right", "bodice_back", "shoulder_right", "SHOULDER", ratio_max=1.04),
        _interface("side_left", "bodice_front", "side_left", "bodice_back", "side_left", "SIDE_SEAM", ratio_max=1.03),
        _interface("side_right", "bodice_front", "side_right", "bodice_back", "side_right", "SIDE_SEAM", ratio_max=1.03),
        _interface("left_cap_front", "sleeve_left", "cap_front", "bodice_front", "armhole_left_front", "SLEEVE_CAP", "BOUNDED_EASE", "NORMALIZED_ARC", 1.08),
        _interface("left_cap_back", "sleeve_left", "cap_back", "bodice_back", "armhole_left_back", "SLEEVE_CAP", "BOUNDED_EASE", "NORMALIZED_ARC", 1.08),
        _interface("right_cap_front", "sleeve_right", "cap_front", "bodice_front", "armhole_right_front", "SLEEVE_CAP", "BOUNDED_EASE", "NORMALIZED_ARC", 1.08),
        _interface("right_cap_back", "sleeve_right", "cap_back", "bodice_back", "armhole_right_back", "SLEEVE_CAP", "BOUNDED_EASE", "NORMALIZED_ARC", 1.08),
        _interface("left_underarm", "sleeve_left", "underarm_front", "sleeve_left", "underarm_back", "UNDERARM_SEAM", ratio_max=1.03),
        _interface("right_underarm", "sleeve_right", "underarm_front", "sleeve_right", "underarm_back", "UNDERARM_SEAM", ratio_max=1.03),
        _interface("collar_front", "collar", "front_attach", "bodice_front", "neckline_front", "COLLAR_ATTACH", "EQUAL", "NONE", 1.002),
        _interface("collar_back", "collar", "back_attach", "bodice_back", "neckline_back", "COLLAR_ATTACH", "EQUAL", "NONE", 1.002),
        _interface("cuff_left_attach", "cuff_left", "sleeve_attach", "sleeve_left", "wrist", "CUFF_ATTACH", ratio_max=1.002),
        _interface("cuff_right_attach", "cuff_right", "sleeve_attach", "sleeve_right", "wrist", "CUFF_ATTACH", ratio_max=1.002),
        _interface("cuff_left_end", "cuff_left", "end_left", "cuff_left", "end_right", "CUFF_END_SEAM", ratio_max=1.002),
        _interface("cuff_right_end", "cuff_right", "end_left", "cuff_right", "end_right", "CUFF_END_SEAM", ratio_max=1.002),
    )


def assembly_recipe() -> GarmentAssemblyRecipe:
    return GarmentAssemblyRecipe(
        "R1C_CP2_SLEEVED_TUNIC_ASSEMBLY",
        "SLEEVED_TUNIC",
        (
            ComponentInstance("bodice_front", "BODICE_FRONT_BASIC", None, {}),
            ComponentInstance("bodice_back", "BODICE_BACK_BASIC", None, {}),
            ComponentInstance("sleeve_left", "SET_IN_SLEEVE_BASIC", None, {}),
            ComponentInstance("sleeve_right", "SET_IN_SLEEVE_BASIC", "sleeve_left", {}),
            ComponentInstance("collar", "COLLAR_STAND_BASIC", None, {}),
            ComponentInstance("cuff_left", "CUFF_STRAIGHT_BASIC", None, {}),
            ComponentInstance("cuff_right", "CUFF_STRAIGHT_BASIC", "cuff_left", {}),
        ),
        tuple(item.interface_id for item in component_interfaces()),
        ("SHELL",),
        ("STRICT", "GUIDED", "SAFE_AUTO"),
        "HOLD",
        "R1C_CP2_GEOMETRY_TECHNICAL_PROFILE",
        "R1C_VISUAL_PROFILE_V1",
    )
