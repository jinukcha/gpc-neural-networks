"""Anthropometry V2 measurement authority used by professional garment CAD."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping

ALLOWED_METHODS = {
    "DIRECT_MESH_SECTION",
    "DIRECT_LANDMARK_DISTANCE",
    "DIRECT_MANUAL_TAPE",
    "SCAN_DERIVED",
    "TABLE_SEED",
    "FORMULA_DERIVED",
    "USER_OVERRIDE",
    "MIGRATED_V1",
    "MISSING",
}

REQUIRED_TORSO = (
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


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MeasurementRecord:
    measurement_id: str
    value_m: float | None
    method: str
    confidence: float
    tolerance_m: float
    frame: str = "RH_Z_UP_PELVIS_CENTERED"
    landmark_ids: tuple[str, ...] = ()
    section_id: str | None = None
    source_revision: str = "ANTHROPOMETRY_V2_CP0_RECOVERY"

    def validate(self) -> None:
        if not self.measurement_id:
            raise ValueError("measurement_id is required")
        if self.method not in ALLOWED_METHODS:
            raise ValueError(f"unsupported measurement method: {self.method}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.tolerance_m < 0.0:
            raise ValueError("tolerance must be non-negative")
        if self.method == "MISSING":
            if self.value_m is not None:
                raise ValueError("MISSING record must not carry a value")
        elif self.value_m is None or not 0.0 < float(self.value_m) < 4.0:
            raise ValueError(f"invalid measurement value: {self.measurement_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["landmark_ids"] = list(self.landmark_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MeasurementRecord":
        result = cls(
            measurement_id=str(payload["measurement_id"]),
            value_m=None if payload.get("value_m") is None else float(payload["value_m"]),
            method=str(payload["method"]),
            confidence=float(payload["confidence"]),
            tolerance_m=float(payload["tolerance_m"]),
            frame=str(payload.get("frame", "RH_Z_UP_PELVIS_CENTERED")),
            landmark_ids=tuple(str(x) for x in payload.get("landmark_ids", [])),
            section_id=None if payload.get("section_id") is None else str(payload["section_id"]),
            source_revision=str(payload.get("source_revision", "ANTHROPOMETRY_V2_CP0_RECOVERY")),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class BodyMeasurementProfileV2:
    profile_id: str
    records: Mapping[str, MeasurementRecord]
    source_mesh_sha256: str
    landmark_package_sha256: str | None = None
    section_package_sha256: str | None = None
    provenance: Mapping[str, object] | None = None

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id is required")
        missing = sorted(set(REQUIRED_TORSO) - set(self.records))
        if missing:
            raise ValueError(f"missing required torso measurements: {missing}")
        for name, record in self.records.items():
            if name != record.measurement_id:
                raise ValueError(f"measurement key mismatch: {name}")
            record.validate()
        self._validate_arc_closure("chest_circumference", "front_chest_arc", "back_chest_arc")
        self._validate_arc_closure("waist_circumference", "front_waist_arc", "back_waist_arc")

    def _value(self, name: str) -> float:
        value = self.records[name].value_m
        if value is None:
            raise ValueError(f"measurement has no numeric value: {name}")
        return float(value)

    def _validate_arc_closure(self, circumference: str, front: str, back: str) -> None:
        error = abs(self._value(front) + self._value(back) - self._value(circumference))
        if error > 0.002:
            raise ValueError(f"arc closure exceeds 2 mm for {circumference}: {error}")

    def value(self, name: str) -> float:
        self.validate()
        return self._value(name)

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "BodyMeasurementProfile/2",
            "profile_id": self.profile_id,
            "units": "m",
            "source_mesh_sha256": self.source_mesh_sha256,
            "landmark_package_sha256": self.landmark_package_sha256,
            "section_package_sha256": self.section_package_sha256,
            "records": {name: self.records[name].to_dict() for name in sorted(self.records)},
            "provenance": dict(self.provenance or {}),
        }
        payload["profile_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BodyMeasurementProfileV2":
        records_payload = payload.get("records")
        if not isinstance(records_payload, Mapping):
            raise ValueError("records object is required")
        result = cls(
            profile_id=str(payload["profile_id"]),
            records={str(k): MeasurementRecord.from_dict(v) for k, v in records_payload.items()},
            source_mesh_sha256=str(payload.get("source_mesh_sha256", "UNAVAILABLE")),
            landmark_package_sha256=(
                None if payload.get("landmark_package_sha256") is None
                else str(payload["landmark_package_sha256"])
            ),
            section_package_sha256=(
                None if payload.get("section_package_sha256") is None
                else str(payload["section_package_sha256"])
            ),
            provenance=dict(payload.get("provenance", {})),
        )
        result.validate()
        return result
