"""Body-shape and height-block selection for the CP2B tunic pilot."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..body_profile.model import BodyMeasurements


@dataclass(frozen=True)
class BlockSelection:
    shape_block: str
    height_block: str
    shape_flags: tuple[str, ...]
    evidence: dict[str, float]

    @property
    def combined_id(self) -> str:
        return f"{self.shape_block}+{self.height_block}"


SHAPE_ADJUSTMENTS = {
    "REGULAR": {},
    "BROAD_SHOULDER": {"shoulder_width": 0.030},
    "FULL_CHEST": {"chest_circumference": 0.040, "front_chest_arc": 0.040},
    "FULL_ABDOMEN": {"waist_circumference": 0.050, "front_waist_arc": 0.050},
}
HEIGHT_ADJUSTMENTS = {
    "REGULAR": {},
    "TALL": {
        "stature": 0.080,
        "front_torso_length": 0.025,
        "back_torso_length": 0.025,
        "armscye_depth": 0.010,
    },
    "SHORT": {
        "stature": -0.080,
        "front_torso_length": -0.025,
        "back_torso_length": -0.025,
        "armscye_depth": -0.010,
    },
}


def select_blocks(body: BodyMeasurements, target: BodyMeasurements) -> BlockSelection:
    shoulder_delta = body.shoulder_width - target.shoulder_width
    front_chest_delta = body.front_chest_arc - target.front_chest_arc
    back_chest_delta = body.back_chest_arc - target.back_chest_arc
    front_waist_delta = body.front_waist_arc - target.front_waist_arc
    back_waist_delta = body.back_waist_arc - target.back_waist_arc
    chest_prominence = front_chest_delta - back_chest_delta
    abdomen_prominence = front_waist_delta - back_waist_delta
    flags = []
    strengths = {}
    if shoulder_delta > 0.018:
        flags.append("BROAD_SHOULDER")
        strengths["BROAD_SHOULDER"] = shoulder_delta / 0.018
    if front_chest_delta > 0.025 and chest_prominence > 0.018:
        flags.append("FULL_CHEST")
        strengths["FULL_CHEST"] = min(front_chest_delta / 0.025, chest_prominence / 0.018)
    if front_waist_delta > 0.035 and abdomen_prominence > 0.020:
        flags.append("FULL_ABDOMEN")
        strengths["FULL_ABDOMEN"] = min(front_waist_delta / 0.035, abdomen_prominence / 0.020)
    shape = max(flags, key=lambda item: strengths[item]) if flags else "REGULAR"
    torso_delta = 0.5 * (
        body.front_torso_length + body.back_torso_length
        - target.front_torso_length - target.back_torso_length
    )
    stature_delta = body.stature - target.stature
    height = "TALL" if stature_delta > 0.070 or torso_delta > 0.025 else "REGULAR"
    if stature_delta < -0.070 or torso_delta < -0.025:
        height = "SHORT"
    return BlockSelection(
        shape_block=shape,
        height_block=height,
        shape_flags=tuple(flags),
        evidence={
            "shoulder_delta_m": shoulder_delta,
            "chest_prominence_m": chest_prominence,
            "abdomen_prominence_m": abdomen_prominence,
            "torso_delta_m": torso_delta,
            "stature_delta_m": stature_delta,
        },
    )


def apply_blocks(target: BodyMeasurements, selection: BlockSelection) -> BodyMeasurements:
    values = asdict(target)
    for mapping in (
        SHAPE_ADJUSTMENTS[selection.shape_block],
        HEIGHT_ADJUSTMENTS[selection.height_block],
    ):
        for name, delta in mapping.items():
            values[name] += delta
    if selection.shape_block == "FULL_CHEST":
        values["back_chest_arc"] = values["chest_circumference"] - values["front_chest_arc"]
    if selection.shape_block == "FULL_ABDOMEN":
        values["back_waist_arc"] = values["waist_circumference"] - values["front_waist_arc"]
    return BodyMeasurements.from_dict(values)
