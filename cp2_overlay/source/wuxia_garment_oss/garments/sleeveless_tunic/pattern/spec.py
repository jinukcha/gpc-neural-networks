"""Design constants for the CP2B sleeveless long tunic family."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TunicDesignSpec:
    front_chest_ease_m: float = 0.050
    back_chest_ease_m: float = 0.049
    front_waist_ease_m: float = 0.060
    back_waist_ease_m: float = 0.060
    hip_ease_m: float = 0.120
    shoulder_drop_m: float = 0.070
    armscye_mobility_m: float = 0.060
    back_balance_m: float = 0.020
    reference_shoulder_width_m: float = 0.410
    reference_neck_half_m: float = 0.083
    neck_shoulder_factor: float = 0.180
    minimum_neck_half_m: float = 0.070
    maximum_neck_half_m: float = 0.100
    reference_front_neck_depth_m: float = 0.160
    reference_back_neck_depth_m: float = 0.072
    front_neck_length_factor: float = 0.350
    back_neck_length_factor: float = 0.250
    reference_front_torso_m: float = 0.455
    reference_back_torso_m: float = 0.435
    reference_stature_m: float = 1.740
    reference_skirt_length_m: float = 0.820
    skirt_stature_factor: float = 0.450
    reference_hip_drop_m: float = 0.200
    hip_drop_stature_factor: float = 0.100
    hem_flare_m: float = 0.025

    def clamp_neck_half(self, value: float) -> float:
        return min(self.maximum_neck_half_m, max(self.minimum_neck_half_m, value))
