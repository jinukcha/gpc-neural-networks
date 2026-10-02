"""Immutable CP1 inputs consumed by CP2 component geometry builders."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class GeometryInputs:
    resolved_set_sha256: str
    seam_allowance_m: float
    sleeve_length_m: float
    cap_height_m: float
    cap_ease_m: float
    cap_ease_ratio: float
    front_pitch_notch_m: float
    collar_depth_m: float
    turn_of_cloth_m: float
    arm_rest_pitch_rad: float
    half_panel_width_m: float
    armscye_depth_m: float
    neckline_depth_m: float
    body_arm_length_m: float
    material_thickness_m: float

    def to_dict(self) -> dict:
        return {
            "contract": "ComponentGeometryInput/1",
            **self.__dict__,
        }


def _parameter_map(payload: dict) -> dict[str, float]:
    return {
        item["parameter_id"]: float(item["resolved_value_si"])
        for item in payload["parameters"]
    }


def _reference_map(payload: dict) -> dict[str, float]:
    return {
        item["qualified_path"]: float(item["value_si"])
        for item in payload["references"]
    }


def load_geometry_inputs(root: Path) -> GeometryInputs:
    build = root / "build/r1c_cp1"
    resolved = json.loads((build / "resolved_parameter_set.json").read_text(encoding="utf-8"))
    context = json.loads((build / "resolution_context.json").read_text(encoding="utf-8"))
    parameters = _parameter_map(resolved)
    references = _reference_map(context)
    return GeometryInputs(
        resolved_set_sha256=resolved["resolved_set_sha256"],
        seam_allowance_m=parameters["seam_allowance"],
        sleeve_length_m=parameters["sleeve_length"],
        cap_height_m=parameters["cap_height"],
        cap_ease_m=parameters["cap_ease"],
        cap_ease_ratio=parameters["cap_ease_ratio"],
        front_pitch_notch_m=parameters["front_pitch_notch"],
        collar_depth_m=parameters["collar_depth"],
        turn_of_cloth_m=parameters["turn_of_cloth"],
        arm_rest_pitch_rad=parameters["arm_rest_pitch"],
        half_panel_width_m=references["block.front_panel_width"],
        armscye_depth_m=references["block.armscye_depth"],
        neckline_depth_m=references["component.neckline_depth"],
        body_arm_length_m=references["body.arm_length"],
        material_thickness_m=references["material.thickness"],
    )
