"""Stable external parameter references with explicit provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .quantity import canonical_unit, convert_to_si, validate_quantity


REFERENCE_SCOPES = {
    "BODY_RELATIVE": "body",
    "BLOCK_RELATIVE": "block",
    "COMPONENT_RELATIVE": "component",
    "BOUNDARY_RELATIVE": "boundary",
    "MATERIAL_RELATIVE": "material",
}


@dataclass(frozen=True)
class ParameterReference:
    scope: str
    path: str
    owner_id: str
    quantity: str
    unit: str
    value: float
    source_package_sha256: str
    source_revision: int

    @property
    def qualified_path(self) -> str:
        prefix = REFERENCE_SCOPES.get(self.scope)
        return f"{prefix}.{self.path}" if prefix else self.path

    @property
    def value_si(self) -> float:
        return convert_to_si(self.value, self.unit, self.quantity)

    def validate(self) -> None:
        if self.scope not in REFERENCE_SCOPES:
            raise ValueError(f"unsupported reference scope: {self.scope}")
        validate_quantity(self.quantity)
        if not self.path or "." in self.path[:1] or not self.owner_id:
            raise ValueError("invalid reference identity")
        if len(self.source_package_sha256) != 64 or self.source_revision < 1:
            raise ValueError("invalid reference provenance")
        if not math.isfinite(self.value_si):
            raise ValueError("non-finite reference")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "contract": "ParameterReference/1",
            **asdict(self),
            "qualified_path": self.qualified_path,
            "value_si": self.value_si,
            "canonical_unit": canonical_unit(self.quantity),
        }
