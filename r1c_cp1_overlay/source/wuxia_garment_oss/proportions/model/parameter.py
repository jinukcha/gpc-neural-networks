"""Parameter definition and bounded-resolution policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .quantity import canonical_unit, convert_to_si, validate_quantity
from .reference import REFERENCE_SCOPES


PARAMETER_MODES = {"ABSOLUTE", "RELATIVE", "AUTO_DERIVED"}
BOUND_DISPOSITIONS = {"REJECT", "SAFE_CLAMP", "ALTERNATE_COMPONENT_REQUIRED"}


@dataclass(frozen=True)
class BoundSpec:
    minimum_si: float | None = None
    maximum_si: float | None = None
    disposition: str = "REJECT"

    def validate(self) -> None:
        if self.disposition not in BOUND_DISPOSITIONS:
            raise ValueError(f"unsupported bound disposition: {self.disposition}")
        values = (value for value in (self.minimum_si, self.maximum_si) if value is not None)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("non-finite bound")
        if self.minimum_si is not None and self.maximum_si is not None:
            if self.minimum_si > self.maximum_si:
                raise ValueError("minimum bound exceeds maximum")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ParameterDefinition:
    parameter_id: str
    mode: str
    quantity: str
    unit: str
    absolute_value: float | None = None
    reference_scope: str | None = None
    reference_path: str | None = None
    ratio: float | None = None
    expression: str | None = None
    bounds: BoundSpec = BoundSpec()
    owner_component_id: str = "GLOBAL"
    description: str = ""

    def validate(self) -> None:
        if not self.parameter_id or "." in self.parameter_id:
            raise ValueError("parameter_id must be a stable local identifier")
        if self.mode not in PARAMETER_MODES:
            raise ValueError(f"unsupported parameter mode: {self.mode}")
        validate_quantity(self.quantity)
        self.bounds.validate()
        self._validate_mode_payload()
        self._validate_unit()

    def _validate_unit(self) -> None:
        if self.mode == "ABSOLUTE":
            convert_to_si(self.absolute_value or 0.0, self.unit, self.quantity)
        elif self.unit != canonical_unit(self.quantity):
            raise ValueError(f"resolved parameter unit must be canonical: {self.unit}")

    def _validate_mode_payload(self) -> None:
        if self.mode == "ABSOLUTE":
            if self.absolute_value is None or not math.isfinite(self.absolute_value):
                raise ValueError("ABSOLUTE requires finite absolute_value")
            if any(value is not None for value in (self.reference_scope, self.reference_path, self.ratio, self.expression)):
                raise ValueError("ABSOLUTE contains incompatible fields")
            return
        if self.mode == "RELATIVE":
            if self.reference_scope not in REFERENCE_SCOPES or not self.reference_path:
                raise ValueError("RELATIVE requires supported reference scope and path")
            if self.ratio is None or not math.isfinite(self.ratio):
                raise ValueError("RELATIVE requires finite ratio")
            if self.absolute_value is not None or self.expression is not None:
                raise ValueError("RELATIVE contains incompatible fields")
            return
        if not self.expression:
            raise ValueError("AUTO_DERIVED requires expression")
        if any(value is not None for value in (self.absolute_value, self.reference_scope, self.reference_path, self.ratio)):
            raise ValueError("AUTO_DERIVED contains incompatible fields")

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["contract"] = "ParameterDefinition/1"
        payload["bounds"] = self.bounds.to_dict()
        return payload
