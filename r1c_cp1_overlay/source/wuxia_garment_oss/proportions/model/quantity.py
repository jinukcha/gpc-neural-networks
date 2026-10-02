"""Physical quantity and SI-unit rules for parameter evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import math


QUANTITIES = {
    "LENGTH": (1, 0, 0),
    "AREA": (2, 0, 0),
    "ANGLE": (0, 0, 1),
    "MASS_PER_AREA": (-2, 1, 0),
    "DIMENSIONLESS": (0, 0, 0),
}

CANONICAL_UNITS = {
    "LENGTH": "m",
    "AREA": "m2",
    "ANGLE": "rad",
    "MASS_PER_AREA": "kg/m2",
    "DIMENSIONLESS": "1",
}

UNIT_RULES = {
    "m": ("LENGTH", 1.0),
    "cm": ("LENGTH", 0.01),
    "mm": ("LENGTH", 0.001),
    "m2": ("AREA", 1.0),
    "cm2": ("AREA", 0.0001),
    "rad": ("ANGLE", 1.0),
    "deg": ("ANGLE", math.pi / 180.0),
    "kg/m2": ("MASS_PER_AREA", 1.0),
    "g/m2": ("MASS_PER_AREA", 0.001),
    "1": ("DIMENSIONLESS", 1.0),
    "ratio": ("DIMENSIONLESS", 1.0),
}


@dataclass(frozen=True)
class TypedValue:
    value_si: float
    dimension: tuple[int, int, int]

    def validate(self) -> None:
        if not math.isfinite(self.value_si):
            raise ValueError("non-finite typed value")

    def to_dict(self) -> dict:
        self.validate()
        return {"value_si": self.value_si, "dimension": list(self.dimension)}


def validate_quantity(quantity: str) -> None:
    if quantity not in QUANTITIES:
        raise ValueError(f"unsupported quantity: {quantity}")


def quantity_dimension(quantity: str) -> tuple[int, int, int]:
    validate_quantity(quantity)
    return QUANTITIES[quantity]


def quantity_for_dimension(dimension: tuple[int, int, int]) -> str | None:
    for quantity, candidate in QUANTITIES.items():
        if candidate == dimension:
            return quantity
    return None


def convert_to_si(value: float, unit: str, expected_quantity: str) -> float:
    validate_quantity(expected_quantity)
    if unit not in UNIT_RULES:
        raise ValueError(f"unsupported unit: {unit}")
    quantity, scale = UNIT_RULES[unit]
    if quantity != expected_quantity:
        raise ValueError(f"unit quantity mismatch: {unit} is {quantity}, expected {expected_quantity}")
    result = float(value) * scale
    if not math.isfinite(result):
        raise ValueError("non-finite converted value")
    return result


def typed_literal(value: float, quantity: str = "DIMENSIONLESS") -> TypedValue:
    validate_quantity(quantity)
    result = TypedValue(float(value), quantity_dimension(quantity))
    result.validate()
    return result


def combine_dimensions(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    operation: str,
) -> tuple[int, int, int]:
    sign = 1 if operation == "MULTIPLY" else -1
    return tuple(a + sign * b for a, b in zip(left, right, strict=True))


def canonical_unit(quantity: str) -> str:
    validate_quantity(quantity)
    return CANONICAL_UNITS[quantity]
