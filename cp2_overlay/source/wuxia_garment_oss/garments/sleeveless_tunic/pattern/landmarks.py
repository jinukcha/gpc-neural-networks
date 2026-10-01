"""Compile named 2D landmarks, open boundaries, and seam identities."""
from __future__ import annotations


def _point(x: float, y: float) -> list[float]:
    return [float(x), float(y)]


def _bodice_landmarks(poms: dict[str, float], front: bool) -> dict[str, list[float]]:
    side = "front" if front else "back"
    chest_half = poms[f"{side}_chest_half_m"]
    waist_half = poms[f"{side}_waist_half_m"]
    shoulder_height = poms[f"{side}_shoulder_height_m"]
    top_height = poms[f"{side}_top_height_m"]
    neck_depth = poms[f"{side}_neck_depth_m"]
    shoulder_half = poms["shoulder_half_m"]
    neck_half = poms["neck_half_m"]
    underarm_height = poms["underarm_height_m"]
    armhole_y = underarm_height + 0.55 * (shoulder_height - underarm_height)
    armhole_x = 0.55 * shoulder_half + 0.45 * chest_half
    return {
        "neck_center": _point(0.0, top_height - neck_depth),
        "neck_left": _point(-neck_half, top_height),
        "neck_right": _point(neck_half, top_height),
        "shoulder_left": _point(-shoulder_half, shoulder_height),
        "shoulder_right": _point(shoulder_half, shoulder_height),
        "armhole_left_mid": _point(-armhole_x, armhole_y),
        "armhole_right_mid": _point(armhole_x, armhole_y),
        "underarm_left": _point(-chest_half, underarm_height),
        "underarm_right": _point(chest_half, underarm_height),
        "waist_left": _point(-waist_half, 0.0),
        "waist_right": _point(waist_half, 0.0),
    }


def _skirt_landmarks(poms: dict[str, float], front: bool) -> dict[str, list[float]]:
    side = "front" if front else "back"
    waist_half = poms[f"{side}_waist_half_m"]
    hip_half = poms["hip_half_m"]
    hem_half = poms["hem_half_m"]
    hip_y = -poms["hip_drop_m"]
    hem_y = -poms["skirt_length_m"]
    return {
        "waist_left": _point(-waist_half, 0.0),
        "waist_right": _point(waist_half, 0.0),
        "hip_left": _point(-hip_half, hip_y),
        "hip_right": _point(hip_half, hip_y),
        "hem_left": _point(-hem_half, hem_y),
        "hem_right": _point(hem_half, hem_y),
    }


def _boundary(
    boundary_id: str,
    order: list[str],
    disposition: str,
    orientation: str,
) -> dict:
    return {
        "boundary_id": boundary_id,
        "landmark_order": order,
        "disposition": disposition,
        "orientation": orientation,
    }


def _bodice_panel(panel_id: str, landmarks: dict[str, list[float]]) -> dict:
    return {
        "panel_id": panel_id,
        "landmarks": landmarks,
        "outline_order": [
            "neck_center", "neck_right", "shoulder_right",
            "armhole_right_mid", "underarm_right", "waist_right",
            "waist_left", "underarm_left", "armhole_left_mid",
            "shoulder_left", "neck_left", "neck_center",
        ],
        "boundaries": [
            _boundary("neckline", ["neck_left", "neck_center", "neck_right"], "OPEN", "LEFT_TO_RIGHT"),
            _boundary("shoulder_left", ["neck_left", "shoulder_left"], "SEWN", "NECK_TO_SIDE"),
            _boundary("shoulder_right", ["shoulder_right", "neck_right"], "SEWN", "SIDE_TO_NECK"),
            _boundary("armhole_left", ["shoulder_left", "armhole_left_mid", "underarm_left"], "OPEN", "TOP_TO_BOTTOM"),
            _boundary("armhole_right", ["underarm_right", "armhole_right_mid", "shoulder_right"], "OPEN", "BOTTOM_TO_TOP"),
            _boundary("side_left", ["underarm_left", "waist_left"], "SEWN", "TOP_TO_BOTTOM"),
            _boundary("side_right", ["waist_right", "underarm_right"], "SEWN", "BOTTOM_TO_TOP"),
            _boundary("waist", ["waist_left", "waist_right"], "SEWN", "LEFT_TO_RIGHT"),
        ],
    }


def _skirt_panel(panel_id: str, landmarks: dict[str, list[float]]) -> dict:
    return {
        "panel_id": panel_id,
        "landmarks": landmarks,
        "outline_order": [
            "waist_left", "waist_right", "hip_right",
            "hem_right", "hem_left", "hip_left", "waist_left",
        ],
        "boundaries": [
            _boundary("waist", ["waist_left", "waist_right"], "SEWN", "LEFT_TO_RIGHT"),
            _boundary("side_left", ["waist_left", "hip_left", "hem_left"], "SEWN", "TOP_TO_BOTTOM"),
            _boundary("side_right", ["hem_right", "hip_right", "waist_right"], "SEWN", "BOTTOM_TO_TOP"),
            _boundary("hem", ["hem_left", "hem_right"], "OPEN", "LEFT_TO_RIGHT"),
        ],
    }


def _seam_pair(seam_id: str, panel_a: str, boundary_a: str, panel_b: str, boundary_b: str) -> dict:
    return {
        "seam_id": seam_id,
        "a": {"panel_id": panel_a, "boundary_id": boundary_a},
        "b": {"panel_id": panel_b, "boundary_id": boundary_b},
        "correspondence": "NORMALIZED_ARC_LENGTH_DEFERRED_CP3",
    }


def compile_panels(poms: dict[str, float]) -> tuple[list[dict], list[dict]]:
    panels = [
        _bodice_panel("bodice_front", _bodice_landmarks(poms, True)),
        _bodice_panel("bodice_back", _bodice_landmarks(poms, False)),
        _skirt_panel("skirt_front", _skirt_landmarks(poms, True)),
        _skirt_panel("skirt_back", _skirt_landmarks(poms, False)),
    ]
    seams = [
        _seam_pair("shoulder_left", "bodice_front", "shoulder_left", "bodice_back", "shoulder_left"),
        _seam_pair("shoulder_right", "bodice_front", "shoulder_right", "bodice_back", "shoulder_right"),
        _seam_pair("bodice_side_left", "bodice_front", "side_left", "bodice_back", "side_left"),
        _seam_pair("bodice_side_right", "bodice_front", "side_right", "bodice_back", "side_right"),
        _seam_pair("waist_front", "bodice_front", "waist", "skirt_front", "waist"),
        _seam_pair("waist_back", "bodice_back", "waist", "skirt_back", "waist"),
        _seam_pair("skirt_side_left", "skirt_front", "side_left", "skirt_back", "side_left"),
        _seam_pair("skirt_side_right", "skirt_front", "side_right", "skirt_back", "side_right"),
    ]
    return panels, seams
