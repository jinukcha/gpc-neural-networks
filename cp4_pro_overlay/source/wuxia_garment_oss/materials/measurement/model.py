"""Material measurement contracts with explicit provenance and uncertainty."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from ...pattern_cad.document.model import canonical_sha256


SUPPORTED_EXPERIMENTS = {
    "WARP_TENSILE",
    "WEFT_TENSILE",
    "BIAS_SHEAR",
    "WARP_BENDING",
    "WEFT_BENDING",
    "THICKNESS_COMPRESSION",
    "STATIC_FRICTION",
    "DYNAMIC_FRICTION",
    "WARP_SHRINKAGE",
    "WEFT_SHRINKAGE",
}


@dataclass(frozen=True)
class MeasurementSeries:
    series_id: str
    experiment_type: str
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    sigma: tuple[float, ...]
    specimen_count: int
    method_id: str

    def validate(self) -> None:
        count = len(self.x)
        if not self.series_id or self.experiment_type not in SUPPORTED_EXPERIMENTS:
            raise ValueError("invalid measurement series identity or experiment")
        if count < 3 or len(self.y) != count or len(self.sigma) != count:
            raise ValueError(f"invalid series length: {self.series_id}")
        if self.specimen_count < 1 or not self.method_id:
            raise ValueError(f"missing specimen provenance: {self.series_id}")
        values = (*self.x, *self.y, *self.sigma)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite measurement: {self.series_id}")
        if any(float(value) <= 0.0 for value in self.sigma):
            raise ValueError(f"non-positive uncertainty: {self.series_id}")

    def to_dict(self) -> dict:
        self.validate()
        return {
            **asdict(self),
            "x": list(self.x),
            "y": list(self.y),
            "sigma": list(self.sigma),
        }


@dataclass(frozen=True)
class MaterialMeasurementSet:
    material_id: str
    display_name: str
    specimen_batch_id: str
    source_kind: str
    certification_status: str
    environment_temperature_c: float
    environment_relative_humidity: float
    areal_density_kg_m2: float
    thickness_m: float
    recommended_meshing_edge_m: float
    series: tuple[MeasurementSeries, ...]
    notes: str

    def validate(self) -> None:
        if not self.material_id or not self.specimen_batch_id:
            raise ValueError("material and batch identity are required")
        if self.source_kind != "PROJECT_REFERENCE_LAB_SERIES":
            raise ValueError("CP4 fixtures must declare project reference provenance")
        if self.certification_status != "NOT_EXTERNAL_LAB_CERTIFIED":
            raise ValueError("external certification must not be implied")
        if not 0.0 < self.areal_density_kg_m2 < 5.0:
            raise ValueError("invalid areal density")
        if not 0.0 < self.thickness_m < 0.02:
            raise ValueError("invalid thickness")
        if not 0.001 <= self.recommended_meshing_edge_m <= 0.05:
            raise ValueError("invalid meshing edge")
        ids = [item.series_id for item in self.series]
        kinds = {item.experiment_type for item in self.series}
        if len(ids) != len(set(ids)) or kinds != SUPPORTED_EXPERIMENTS:
            raise ValueError("material series must cover each required experiment once")
        for item in self.series:
            item.validate()

    def series_by_type(self, experiment_type: str) -> MeasurementSeries:
        for item in self.series:
            if item.experiment_type == experiment_type:
                return item
        raise KeyError(experiment_type)

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "MaterialMeasurementSet/1",
            "material_id": self.material_id,
            "display_name": self.display_name,
            "specimen_batch_id": self.specimen_batch_id,
            "provenance": {
                "source_kind": self.source_kind,
                "certification_status": self.certification_status,
                "notes": self.notes,
            },
            "environment": {
                "temperature_c": self.environment_temperature_c,
                "relative_humidity": self.environment_relative_humidity,
            },
            "areal_density_kg_m2": self.areal_density_kg_m2,
            "thickness_m": self.thickness_m,
            "recommended_meshing_edge_m": self.recommended_meshing_edge_m,
            "series": [item.to_dict() for item in self.series],
        }
        payload["measurement_set_sha256"] = canonical_sha256(payload)
        return payload
