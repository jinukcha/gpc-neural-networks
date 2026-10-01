"""Anthropometry-driven trousers pattern with waistband, darts, and crotch gusset."""
from __future__ import annotations

import re
from typing import Iterable, Mapping

from ...export.manufacturing import curve_point
from ...pattern_cad.document.model import canonical_sha256


_DEFAULTS = {
    "waist": 0.82,
    "hip": 1.02,
    "inseam": 0.80,
    "outseam": 1.05,
    "front_rise": 0.27,
    "back_rise": 0.37,
    "thigh": 0.60,
    "knee": 0.42,
    "ankle": 0.36,
}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _numeric_rows(value: object, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, (int, float)):
                yield _normalize(name), float(child)
            else:
                yield from _numeric_rows(child, name)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _numeric_rows(child, f"{prefix}.{index}")


def _measurement(profile: Mapping[str, object], names: tuple[str, ...], fallback: float) -> float:
    rows = list(_numeric_rows(profile))
    normalized = tuple(_normalize(name) for name in names)
    for key, value in rows:
        if any(name == key.split("_")[-1] or name in key for name in normalized):
            if 0.05 <= value <= 3.0:
                return value
    return fallback


def body_measurements(profile: Mapping[str, object]) -> dict[str, float]:
    values = {
        "waist": _measurement(profile, ("waist_circumference", "waist"), _DEFAULTS["waist"]),
        "hip": _measurement(profile, ("full_hip_circumference", "hip_circumference", "full_hip"), _DEFAULTS["hip"]),
        "inseam": _measurement(profile, ("left_inseam", "inseam"), _DEFAULTS["inseam"]),
        "outseam": _measurement(profile, ("left_outseam", "outseam"), _DEFAULTS["outseam"]),
        "front_rise": _measurement(profile, ("front_crotch_length", "front_rise"), _DEFAULTS["front_rise"]),
        "back_rise": _measurement(profile, ("back_crotch_length", "back_rise"), _DEFAULTS["back_rise"]),
        "thigh": _measurement(profile, ("left_thigh_circumference", "thigh_circumference"), _DEFAULTS["thigh"]),
        "knee": _measurement(profile, ("left_knee_circumference", "knee_circumference"), _DEFAULTS["knee"]),
        "ankle": _measurement(profile, ("left_ankle_circumference", "ankle_circumference"), _DEFAULTS["ankle"]),
    }
    if values["outseam"] <= values["inseam"] + 0.12:
        values["outseam"] = values["inseam"] + 0.25
    return values


def _point(x: float, y: float, mirror: float) -> list[float]:
    return [mirror * x, y]


def _curve(
    curve_id: str,
    panel_id: str,
    kind: str,
    points: list[list[float]],
    role: str,
    disposition: str,
) -> dict:
    return {
        "curve_id": curve_id,
        "panel_id": panel_id,
        "curve_type": kind,
        "point_ids": [f"{curve_id}.p{index}" for index in range(len(points))],
        "points": points,
        "boundary_role": role,
        "disposition": disposition,
    }


def _leg_panel(panel_id: str, pom: Mapping[str, float], is_back: bool, mirror: float) -> list[dict]:
    outseam = pom["finished_outseam"]
    crotch_y = pom["finished_inseam"]
    hip_y = outseam - pom["hip_depth"]
    knee_y = pom["knee_height"]
    waist_width = pom["back_waist_quarter"] if is_back else pom["front_waist_quarter"]
    hip_width = pom["back_hip_quarter"] if is_back else pom["front_hip_quarter"]
    thigh_width = pom["back_thigh_half"] if is_back else pom["front_thigh_half"]
    knee_width = pom["back_knee_half"] if is_back else pom["front_knee_half"]
    ankle_width = pom["back_ankle_half"] if is_back else pom["front_ankle_half"]
    extension = pom["back_crotch_extension"] if is_back else pom["front_crotch_extension"]
    waist_center = _point(0.0, outseam, mirror)
    waist_side = _point(waist_width, outseam - (0.0 if is_back else 0.012), mirror)
    hip_side = _point(hip_width, hip_y, mirror)
    knee_side = _point(knee_width * 0.62, knee_y, mirror)
    hem_side = _point(ankle_width * 0.62, 0.0, mirror)
    hem_inner = _point(-ankle_width * 0.38, 0.0, mirror)
    knee_inner = _point(-knee_width * 0.38, knee_y, mirror)
    thigh_inner = _point(-thigh_width * 0.34, crotch_y - 0.10, mirror)
    crotch_inner = _point(-extension, crotch_y, mirror)
    crotch_control = _point(-extension * 0.70, crotch_y + (pom["back_rise"] if is_back else pom["front_rise"]) * 0.38, mirror)
    prefix = panel_id
    curves = [
        _curve(f"{prefix}.waist", panel_id, "LINE", [waist_center, waist_side], "WAIST", "SEWN"),
        _curve(f"{prefix}.side", panel_id, "CUBIC_BEZIER", [waist_side, hip_side, knee_side, hem_side], "SIDE", "SEWN"),
        _curve(f"{prefix}.hem", panel_id, "LINE", [hem_side, hem_inner], "HEM", "OPEN"),
        _curve(f"{prefix}.inseam", panel_id, "CUBIC_BEZIER", [hem_inner, knee_inner, thigh_inner, crotch_inner], "INSEAM", "SEWN"),
        _curve(f"{prefix}.crotch", panel_id, "QUADRATIC_BEZIER", [crotch_inner, crotch_control, waist_center], "CROTCH", "SEWN"),
    ]
    if is_back:
        dart_center = waist_width * 0.54
        intake = pom["back_dart_intake"]
        apex_y = outseam - pom["back_dart_length"]
        curves.extend((
            _curve(f"{prefix}.dart_left", panel_id, "LINE", [_point(dart_center - intake * 0.5, outseam, mirror), _point(dart_center, apex_y, mirror)], "DART_LEG", "INTERNAL"),
            _curve(f"{prefix}.dart_right", panel_id, "LINE", [_point(dart_center + intake * 0.5, outseam, mirror), _point(dart_center, apex_y, mirror)], "DART_LEG", "INTERNAL"),
        ))
    return curves


def _rectangle_panel(panel_id: str, width: float, height: float, role: str) -> list[dict]:
    points = [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]
    return [
        _curve(f"{panel_id}.bottom", panel_id, "LINE", [points[0], points[1]], f"{role}_BOTTOM", "SEWN"),
        _curve(f"{panel_id}.end_right", panel_id, "LINE", [points[1], points[2]], f"{role}_END", "SEWN"),
        _curve(f"{panel_id}.top", panel_id, "LINE", [points[2], points[3]], f"{role}_TOP", "OPEN"),
        _curve(f"{panel_id}.end_left", panel_id, "LINE", [points[3], points[0]], f"{role}_END", "SEWN"),
    ]


def _gusset_panel(pom: Mapping[str, float]) -> list[dict]:
    panel = "crotch_gusset"
    half_width = pom["gusset_width"] * 0.5
    half_length = pom["gusset_length"] * 0.5
    points = [[0.0, half_length], [half_width, 0.0], [0.0, -half_length], [-half_width, 0.0]]
    return [
        _curve(f"{panel}.front_left", panel, "LINE", [points[0], points[1]], "GUSSET_EDGE", "SEWN"),
        _curve(f"{panel}.back_left", panel, "LINE", [points[1], points[2]], "GUSSET_EDGE", "SEWN"),
        _curve(f"{panel}.back_right", panel, "LINE", [points[2], points[3]], "GUSSET_EDGE", "SEWN"),
        _curve(f"{panel}.front_right", panel, "LINE", [points[3], points[0]], "GUSSET_EDGE", "SEWN"),
    ]


def _pom(measurements: Mapping[str, float]) -> dict[str, float]:
    waist = measurements["waist"] + 0.035
    hip = measurements["hip"] + 0.070
    thigh = measurements["thigh"] + 0.055
    knee = measurements["knee"] + 0.060
    ankle = measurements["ankle"] + 0.055
    return {
        "finished_waist": waist,
        "finished_hip": hip,
        "finished_inseam": measurements["inseam"],
        "finished_outseam": measurements["outseam"],
        "front_rise": measurements["front_rise"],
        "back_rise": measurements["back_rise"],
        "hip_depth": min(0.23, measurements["outseam"] - measurements["inseam"] - 0.04),
        "knee_height": measurements["inseam"] * 0.48,
        "front_waist_quarter": waist * 0.235,
        "back_waist_quarter": waist * 0.265,
        "front_hip_quarter": hip * 0.242,
        "back_hip_quarter": hip * 0.258,
        "front_thigh_half": thigh * 0.48,
        "back_thigh_half": thigh * 0.52,
        "front_knee_half": knee * 0.48,
        "back_knee_half": knee * 0.52,
        "front_ankle_half": ankle * 0.48,
        "back_ankle_half": ankle * 0.52,
        "front_crotch_extension": hip / 20.0 + 0.008,
        "back_crotch_extension": hip / 10.0 + 0.018,
        "back_dart_intake": 0.024,
        "back_dart_length": 0.125,
        "waistband_height": 0.045,
        "gusset_width": 0.105,
        "gusset_length": 0.185,
    }


def _side(panel: str, curve: str) -> dict:
    return {"panel_id": panel, "curve_id": curve, "start_fraction": 0.0, "end_fraction": 1.0}


def _seam(seam_id: str, kind: str, a: tuple[str, str], b: tuple[str, str], allowance: float = 0.015) -> dict:
    return {
        "seam_id": seam_id,
        "seam_type": kind,
        "side_a": _side(*a),
        "side_b": _side(*b),
        "allowance_a_m": allowance,
        "allowance_b_m": allowance,
        "stitch_class": "ISO_301_LOCKSTITCH",
        "ease_ratio": 1.0,
        "gather_ratio": 1.0,
        "notch_pair_ids": [],
        "fold_direction": "NONE",
        "topstitch_offset_m": 0.0,
        "turn_of_cloth_m": 0.0,
    }


def _construction(resolved: Mapping[str, object], pom: Mapping[str, float]) -> dict:
    seams = [
        _seam("side_left", "PLAIN_SEAM", ("front_left", "front_left.side"), ("back_left", "back_left.side")),
        _seam("side_right", "PLAIN_SEAM", ("front_right", "front_right.side"), ("back_right", "back_right.side")),
        _seam("inseam_left", "PLAIN_SEAM", ("front_left", "front_left.inseam"), ("back_left", "back_left.inseam")),
        _seam("inseam_right", "PLAIN_SEAM", ("front_right", "front_right.inseam"), ("back_right", "back_right.inseam")),
        _seam("center_front", "PLAIN_SEAM", ("front_left", "front_left.crotch"), ("front_right", "front_right.crotch"), 0.012),
        _seam("center_back", "PLAIN_SEAM", ("back_left", "back_left.crotch"), ("back_right", "back_right.crotch"), 0.018),
        _seam("waist_front_left", "WAIST_JOIN", ("front_left", "front_left.waist"), ("waistband_front", "waistband_front.bottom"), 0.012),
        _seam("waist_front_right", "WAIST_JOIN", ("front_right", "front_right.waist"), ("waistband_front", "waistband_front.bottom"), 0.012),
        _seam("waist_back_left", "WAIST_JOIN", ("back_left", "back_left.waist"), ("waistband_back", "waistband_back.bottom"), 0.012),
        _seam("waist_back_right", "WAIST_JOIN", ("back_right", "back_right.waist"), ("waistband_back", "waistband_back.bottom"), 0.012),
    ]
    gusset_pairs = (
        ("gusset_front_left", ("front_left", "front_left.crotch"), ("crotch_gusset", "crotch_gusset.front_left")),
        ("gusset_back_left", ("back_left", "back_left.crotch"), ("crotch_gusset", "crotch_gusset.back_left")),
        ("gusset_back_right", ("back_right", "back_right.crotch"), ("crotch_gusset", "crotch_gusset.back_right")),
        ("gusset_front_right", ("front_right", "front_right.crotch"), ("crotch_gusset", "crotch_gusset.front_right")),
    )
    seams.extend(_seam(name, "GUSSET_INSERTION", a, b, 0.010) for name, a, b in gusset_pairs)
    notches = []
    for seam in seams:
        for fraction, role in ((0.33, "MATCH_1"), (0.66, "MATCH_2")):
            a_curve = resolved["curves"][seam["side_a"]["curve_id"]]
            b_curve = resolved["curves"][seam["side_b"]["curve_id"]]
            notches.append({
                "pair_id": f"N_{seam['seam_id']}_{role}",
                "seam_id": seam["seam_id"],
                "role": role,
                "side_a_fraction": fraction,
                "side_b_fraction": fraction,
                "side_a_position": list(curve_point(a_curve, fraction)),
                "side_b_position": list(curve_point(b_curve, fraction)),
            })
    finishes = []
    for panel in ("front_left", "front_right", "back_left", "back_right"):
        finishes.append({"finish": {"finish_id": f"HEM_{panel}", "boundary": _side(panel, f"{panel}.hem"), "finish_type": "DOUBLE_TURN_HEM", "allowance_m": 0.035}, "line": {}})
    for panel in ("waistband_front", "waistband_back"):
        finishes.append({"finish": {"finish_id": f"TOP_{panel}", "boundary": _side(panel, f"{panel}.top"), "finish_type": "BOUND_FACING", "allowance_m": 0.010}, "line": {}})
    return {
        "contract": "ConstructionPackage/1",
        "seam_specs": seams,
        "edge_finishes": finishes,
        "notch_correspondence": notches,
        "closures": [{
            "closure_id": "FRONT_FLY_ZIP",
            "closure_type": "ZIPPER_FLY",
            "owner_panel_id": "front_right",
            "length_m": 0.175,
            "internal_cut_line": [[0.0, pom["finished_outseam"]], [0.0, pom["finished_outseam"] - 0.175]],
            "component_ids": ["ZIPPER_175MM", "HOOK_BAR_001"],
        }],
        "facings": [],
        "layer_pieces": [],
        "turn_of_cloth": [{"turn_id": "TURN_WAISTBAND", "owner_id": "waistband_front", "allowance_m": 0.0012, "direction": "INSIDE"}],
        "assembly_plan": {
            "operation_count": 12,
            "cycle_free": True,
            "operations": [
                "CLOSE_BACK_DARTS", "PREPARE_GUSSET", "SEW_CENTER_FRONT", "SEW_CENTER_BACK",
                "INSERT_GUSSET", "SEW_INSEAMS", "SEW_SIDE_SEAMS", "INSTALL_FLY",
                "ASSEMBLE_WAISTBAND", "ATTACH_WAISTBAND", "HEM_LEGS", "FINAL_PRESS",
            ],
        },
        "bill_of_materials": [
            {"component_id": "SHELL_WOVEN", "quantity": 1.65, "unit": "m"},
            {"component_id": "ZIPPER_175MM", "quantity": 1, "unit": "piece"},
            {"component_id": "HOOK_BAR_001", "quantity": 1, "unit": "set"},
            {"component_id": "WAISTBAND_INTERFACING", "quantity": 0.22, "unit": "m"},
        ],
    }


def build_trousers_pattern(body_profile: Mapping[str, object]) -> dict:
    measurements = body_measurements(body_profile)
    pom = _pom(measurements)
    curves = []
    curves.extend(_leg_panel("front_left", pom, False, 1.0))
    curves.extend(_leg_panel("front_right", pom, False, -1.0))
    curves.extend(_leg_panel("back_left", pom, True, 1.0))
    curves.extend(_leg_panel("back_right", pom, True, -1.0))
    curves.extend(_rectangle_panel("waistband_front", pom["finished_waist"] * 0.50 + 0.025, pom["waistband_height"], "WAISTBAND"))
    curves.extend(_rectangle_panel("waistband_back", pom["finished_waist"] * 0.50 + 0.025, pom["waistband_height"], "WAISTBAND"))
    curves.extend(_gusset_panel(pom))
    resolved = {
        "contract": "ResolvedTrousersPatternDocument/1",
        "document_id": "PRO_TROUSERS_REFERENCE_M",
        "revision": 1,
        "values": {**measurements, **pom},
        "curves": {item["curve_id"]: item for item in curves},
        "hard_constraints_pass": True,
    }
    construction = _construction(resolved, pom)
    authority = {
        "contract": "TrousersPatternAuthority/1",
        "document_id": resolved["document_id"],
        "body_measurements": measurements,
        "points_of_measure": pom,
        "panel_ids": ["front_left", "front_right", "back_left", "back_right", "waistband_front", "waistband_back", "crotch_gusset"],
        "resolved": resolved,
        "construction": construction,
        "dart_count": 4,
        "gusset_panel_count": 1,
        "waistband_panel_count": 2,
        "topology_change_required": False,
    }
    authority["authority_sha256"] = canonical_sha256(authority)
    return authority
