"""Bound application and explicit repair disposition."""
from __future__ import annotations

from dataclasses import dataclass

from ..model.parameter import BoundSpec


@dataclass(frozen=True)
class BoundOutcome:
    requested_value_si: float
    resolved_value_si: float
    disposition: str
    action: str
    minimum_si: float | None
    maximum_si: float | None

    @property
    def changed(self) -> bool:
        return self.requested_value_si != self.resolved_value_si

    def to_dict(self) -> dict:
        return {
            "requested_value_si": self.requested_value_si,
            "resolved_value_si": self.resolved_value_si,
            "disposition": self.disposition,
            "action": self.action,
            "minimum_si": self.minimum_si,
            "maximum_si": self.maximum_si,
            "delta_si": self.resolved_value_si - self.requested_value_si,
        }


class BoundFailure(ValueError):
    def __init__(self, code: str, message: str, spec: BoundSpec, value_si: float):
        super().__init__(message)
        self.code = code
        self.spec = spec
        self.value_si = value_si


def apply_bounds(value_si: float, spec: BoundSpec) -> BoundOutcome:
    spec.validate()
    below = spec.minimum_si is not None and value_si < spec.minimum_si
    above = spec.maximum_si is not None and value_si > spec.maximum_si
    if not below and not above:
        return BoundOutcome(value_si, value_si, spec.disposition, "UNCHANGED", spec.minimum_si, spec.maximum_si)
    if spec.disposition == "SAFE_CLAMP":
        result = value_si
        if spec.minimum_si is not None:
            result = max(result, spec.minimum_si)
        if spec.maximum_si is not None:
            result = min(result, spec.maximum_si)
        return BoundOutcome(value_si, result, spec.disposition, "CLAMPED", spec.minimum_si, spec.maximum_si)
    if spec.disposition == "ALTERNATE_COMPONENT_REQUIRED":
        raise BoundFailure(
            "ALTERNATE_COMPONENT_REQUIRED",
            "parameter exceeds the topology-preserving range",
            spec,
            value_si,
        )
    raise BoundFailure("BOUND_REJECTED", "parameter violates a hard bound", spec, value_si)
