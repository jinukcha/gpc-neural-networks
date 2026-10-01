"""Body measurement authority shared by all wearable families."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping


MEASUREMENT_FIELDS = (
    "stature",
    "chest_circumference",
    "front_chest_arc",
    "back_chest_arc",
    "waist_circumference",
    "front_waist_arc",
    "back_waist_arc",
    "hip_circumference",
    "shoulder_width",
    "front_torso_length",
    "back_torso_length",
    "armscye_depth",
)


@dataclass(frozen=True)
class BodyMeasurements:
    stature: float
    chest_circumference: float
    front_chest_arc: float
    back_chest_arc: float
    waist_circumference: float
    front_waist_arc: float
    back_waist_arc: float
    hip_circumference: float
    shoulder_width: float
    front_torso_length: float
    back_torso_length: float
    armscye_depth: float

    def validate(self) -> None:
        values = asdict(self)
        invalid = [name for name, value in values.items() if not (0.0 < float(value) < 3.0)]
        if invalid:
            raise ValueError(f"invalid body measurements: {invalid}")
        chest_error = abs(self.front_chest_arc + self.back_chest_arc - self.chest_circumference)
        waist_error = abs(self.front_waist_arc + self.back_waist_arc - self.waist_circumference)
        if chest_error > 0.002 or waist_error > 0.002:
            raise ValueError("front/back arcs must close within 2 mm")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {name: float(value) for name, value in asdict(self).items()}

    def with_updates(self, updates: Mapping[str, float]) -> "BodyMeasurements":
        unknown = sorted(set(updates) - set(MEASUREMENT_FIELDS))
        if unknown:
            raise ValueError(f"unknown measurements: {unknown}")
        result = replace(self, **{key: float(value) for key, value in updates.items()})
        result.validate()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BodyMeasurements":
        missing = [name for name in MEASUREMENT_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"missing body measurements: {missing}")
        result = cls(**{name: float(payload[name]) for name in MEASUREMENT_FIELDS})
        result.validate()
        return result


@dataclass(frozen=True)
class BodyMeasurementProfile:
    body_id: str
    measurements: BodyMeasurements
    source: str = "MANUAL_REFERENCE_FIXTURE"
    confidence: float = 1.0

    def validate(self) -> None:
        if not self.body_id:
            raise ValueError("body_id is required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0,1]")
        self.measurements.validate()

    def to_dict(self) -> dict:
        self.validate()
        return {
            "contract": "BodyMeasurementProfile/1",
            "body_id": self.body_id,
            "units": "m",
            "source": self.source,
            "confidence": float(self.confidence),
            "measurements": self.measurements.to_dict(),
        }
