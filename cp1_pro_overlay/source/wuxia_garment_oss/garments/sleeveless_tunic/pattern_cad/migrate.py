"""Migrate the CP2 tunic M parameter authority into PatternDocument/1."""
from __future__ import annotations

from typing import Mapping

from ....anthropometry.v2.model import BodyMeasurementProfileV2
from ....pattern_cad.document.model import (
    PatternConstraint,
    PatternCurve,
    PatternDocument,
    PatternPoint,
    canonical_sha256,
)


def _inputs(profile: BodyMeasurementProfileV2) -> dict[str, float]:
    values = {f"body_{name}": profile.value(name) for name in (
        "stature", "front_chest_arc", "back_chest_arc", "front_waist_arc",
        "back_waist_arc", "hip_circumference", "shoulder_width",
        "front_torso_length", "back_torso_length", "armscye_depth",
    )}
    values.update({
        "front_chest_ease": 0.050,
        "back_chest_ease": 0.049,
        "front_waist_ease": 0.060,
        "back_waist_ease": 0.060,
        "hip_ease": 0.120,
        "shoulder_drop": 0.070,
        "armscye_mobility": 0.060,
        "back_balance": 0.020,
        "reference_shoulder_width": 0.410,
        "reference_neck_half": 0.083,
        "neck_shoulder_factor": 0.180,
        "minimum_neck_half": 0.070,
        "maximum_neck_half": 0.100,
        "reference_front_neck_depth": 0.160,
        "reference_back_neck_depth": 0.072,
        "front_neck_length_factor": 0.350,
        "back_neck_length_factor": 0.250,
        "reference_front_torso": 0.455,
        "reference_back_torso": 0.435,
        "reference_stature": 1.740,
        "reference_skirt_length": 0.820,
        "skirt_stature_factor": 0.450,
        "reference_hip_drop": 0.200,
        "hip_drop_stature_factor": 0.100,
        "hem_flare": 0.025,
    })
    return values


def _expressions() -> dict[str, str]:
    return {
        "front_chest_arc": "body_front_chest_arc + front_chest_ease",
        "back_chest_arc": "body_back_chest_arc + back_chest_ease",
        "front_waist_arc": "body_front_waist_arc + front_waist_ease",
        "back_waist_arc": "body_back_waist_arc + back_waist_ease",
        "front_chest_half": "front_chest_arc / 2",
        "back_chest_half": "back_chest_arc / 2",
        "front_waist_half": "front_waist_arc / 2",
        "back_waist_half": "back_waist_arc / 2",
        "shoulder_half": "body_shoulder_width / 2",
        "neck_half": "min(maximum_neck_half, max(minimum_neck_half, reference_neck_half + neck_shoulder_factor * (body_shoulder_width - reference_shoulder_width)))",
        "front_shoulder_y": "body_front_torso_length",
        "back_shoulder_y": "body_back_torso_length + back_balance",
        "front_top_y": "front_shoulder_y + shoulder_drop",
        "back_top_y": "back_shoulder_y + shoulder_drop",
        "underarm_y": "body_armscye_depth + armscye_mobility",
        "front_neck_depth": "reference_front_neck_depth + front_neck_length_factor * (body_front_torso_length - reference_front_torso)",
        "back_neck_depth": "reference_back_neck_depth + back_neck_length_factor * (body_back_torso_length - reference_back_torso)",
        "front_neck_center_y": "front_top_y - front_neck_depth",
        "back_neck_center_y": "back_top_y - back_neck_depth",
        "front_armhole_y": "underarm_y + 0.55 * (front_shoulder_y - underarm_y)",
        "back_armhole_y": "underarm_y + 0.55 * (back_shoulder_y - underarm_y)",
        "front_armhole_x": "0.55 * shoulder_half + 0.45 * front_chest_half",
        "back_armhole_x": "0.55 * shoulder_half + 0.45 * back_chest_half",
        "finished_hip": "body_hip_circumference + hip_ease",
        "hip_half": "finished_hip / 4",
        "skirt_length": "reference_skirt_length + skirt_stature_factor * (body_stature - reference_stature)",
        "hip_drop": "reference_hip_drop + hip_drop_stature_factor * (body_stature - reference_stature)",
        "hip_y": "-hip_drop",
        "hem_y": "-skirt_length",
        "hem_half": "max(front_waist_half, back_waist_half, hip_half) + hem_flare",
        "zero": "0",
        "neck_min_distance": "0.12",
        "front_waist_width": "2 * front_waist_half",
        "back_waist_width": "2 * back_waist_half",
    }


def _point(panel: str, name: str, x: str, y: str) -> PatternPoint:
    point_id = f"{panel}.{name}"
    return PatternPoint(point_id, x, y, panel)


def _bodice_points(panel: str, front: bool) -> list[PatternPoint]:
    side = "front" if front else "back"
    chest = f"{side}_chest_half"
    waist = f"{side}_waist_half"
    shoulder_y = f"{side}_shoulder_y"
    top_y = f"{side}_top_y"
    neck_y = f"{side}_neck_center_y"
    armhole_x = f"{side}_armhole_x"
    armhole_y = f"{side}_armhole_y"
    return [
        _point(panel, "neck_center", "zero", neck_y),
        _point(panel, "neck_left", "-neck_half", top_y),
        _point(panel, "neck_right", "neck_half", top_y),
        _point(panel, "shoulder_left", "-shoulder_half", shoulder_y),
        _point(panel, "shoulder_right", "shoulder_half", shoulder_y),
        _point(panel, "armhole_left_mid", f"-{armhole_x}", armhole_y),
        _point(panel, "armhole_right_mid", armhole_x, armhole_y),
        _point(panel, "underarm_left", f"-{chest}", "underarm_y"),
        _point(panel, "underarm_right", chest, "underarm_y"),
        _point(panel, "waist_left", f"-{waist}", "zero"),
        _point(panel, "waist_right", waist, "zero"),
    ]


def _skirt_points(panel: str, front: bool) -> list[PatternPoint]:
    side = "front" if front else "back"
    waist = f"{side}_waist_half"
    return [
        _point(panel, "waist_left", f"-{waist}", "zero"),
        _point(panel, "waist_right", waist, "zero"),
        _point(panel, "hip_left", "-hip_half", "hip_y"),
        _point(panel, "hip_right", "hip_half", "hip_y"),
        _point(panel, "hem_left", "-hem_half", "hem_y"),
        _point(panel, "hem_right", "hem_half", "hem_y"),
    ]


def _curve(panel: str, name: str, kind: str, points: tuple[str, ...], role: str, disposition: str) -> PatternCurve:
    return PatternCurve(
        curve_id=f"{panel}.{name}", panel_id=panel, curve_type=kind,
        point_ids=tuple(f"{panel}.{point}" for point in points),
        boundary_role=role, disposition=disposition,
    )


def _bodice_curves(panel: str) -> list[PatternCurve]:
    return [
        _curve(panel, "neckline", "QUADRATIC_BEZIER", ("neck_left", "neck_center", "neck_right"), "NECKLINE", "OPEN"),
        _curve(panel, "shoulder_left", "LINE", ("neck_left", "shoulder_left"), "SHOULDER_LEFT", "SEWN"),
        _curve(panel, "shoulder_right", "LINE", ("shoulder_right", "neck_right"), "SHOULDER_RIGHT", "SEWN"),
        _curve(panel, "armhole_left", "QUADRATIC_BEZIER", ("shoulder_left", "armhole_left_mid", "underarm_left"), "ARMHOLE_LEFT", "OPEN"),
        _curve(panel, "armhole_right", "QUADRATIC_BEZIER", ("underarm_right", "armhole_right_mid", "shoulder_right"), "ARMHOLE_RIGHT", "OPEN"),
        _curve(panel, "side_left", "LINE", ("underarm_left", "waist_left"), "SIDE_LEFT", "SEWN"),
        _curve(panel, "side_right", "LINE", ("waist_right", "underarm_right"), "SIDE_RIGHT", "SEWN"),
        _curve(panel, "waist", "LINE", ("waist_left", "waist_right"), "WAIST", "SEWN"),
    ]


def _skirt_curves(panel: str) -> list[PatternCurve]:
    return [
        _curve(panel, "waist", "LINE", ("waist_left", "waist_right"), "WAIST", "SEWN"),
        _curve(panel, "side_left", "QUADRATIC_BEZIER", ("waist_left", "hip_left", "hem_left"), "SIDE_LEFT", "SEWN"),
        _curve(panel, "side_right", "QUADRATIC_BEZIER", ("hem_right", "hip_right", "waist_right"), "SIDE_RIGHT", "SEWN"),
        _curve(panel, "hem", "LINE", ("hem_left", "hem_right"), "HEM", "OPEN"),
    ]


def _constraint(constraint_id: str, kind: str, points: tuple[str, str], target: str | None = None, strength: str = "HARD") -> PatternConstraint:
    return PatternConstraint(constraint_id, kind, points, strength, 1.0e-10, target)


def _constraints() -> dict[str, PatternConstraint]:
    rows: list[PatternConstraint] = []
    for panel in ("bodice_front", "bodice_back"):
        for pair in (("neck_left", "neck_right"), ("shoulder_left", "shoulder_right"), ("underarm_left", "underarm_right"), ("waist_left", "waist_right")):
            rows.append(_constraint(f"{panel}.symmetry.{pair[0]}", "SYMMETRY_X", (f"{panel}.{pair[0]}", f"{panel}.{pair[1]}")))
        rows.append(_constraint(f"{panel}.shoulder_horizontal", "HORIZONTAL", (f"{panel}.shoulder_left", f"{panel}.shoulder_right")))
        rows.append(_constraint(f"{panel}.neck_min", "MIN_DISTANCE", (f"{panel}.neck_left", f"{panel}.neck_right"), "neck_min_distance"))
    for panel, target in (("skirt_front", "front_waist_width"), ("skirt_back", "back_waist_width")):
        rows.append(_constraint(f"{panel}.waist_width", "FIXED_DISTANCE", (f"{panel}.waist_left", f"{panel}.waist_right"), target))
        rows.append(_constraint(f"{panel}.hem_horizontal", "HORIZONTAL", (f"{panel}.hem_left", f"{panel}.hem_right")))
    return {row.constraint_id: row for row in rows}


def migrate_reference_tunic(
    parameter_package: Mapping[str, object],
    profile: BodyMeasurementProfileV2,
) -> PatternDocument:
    profile.validate()
    panels = ("bodice_front", "bodice_back", "skirt_front", "skirt_back")
    points = [*_bodice_points("bodice_front", True), *_bodice_points("bodice_back", False), *_skirt_points("skirt_front", True), *_skirt_points("skirt_back", False)]
    curves = [*_bodice_curves("bodice_front"), *_bodice_curves("bodice_back"), *_skirt_curves("skirt_front"), *_skirt_curves("skirt_back")]
    source_hash = canonical_sha256(dict(parameter_package))
    document = PatternDocument(
        document_id="CP2B_TUNIC_M_PATTERN_DOCUMENT",
        revision=0,
        parent_revision=None,
        garment_design_id="CP2B_SLEEVELESS_LONG_TUNIC",
        body_profile_sha256=str(profile.to_dict()["profile_sha256"]),
        inputs=_inputs(profile),
        expressions=_expressions(),
        points={point.point_id: point for point in points},
        curves={curve.curve_id: curve for curve in curves},
        constraints=_constraints(),
        panel_ids=panels,
        metadata={
            "migration_source": "GarmentPatternParameterPackage/1",
            "source_package_sha256": source_hash,
            "source_package_id": parameter_package.get("package_id"),
            "triangulation_executed": False,
            "warp_simulation_executed": False,
            "mesh_scaling": "FORBIDDEN",
            "byte_exact_cp0_predecessor": False,
            "recovery_note": "CP0 ZIP inaccessible; anthropometry authority reconstructed from CP3 sizing inputs",
        },
    )
    document.validate_structure()
    return document
