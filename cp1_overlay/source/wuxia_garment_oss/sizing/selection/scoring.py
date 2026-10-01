"""Normalized multi-dimensional base-size scoring."""
from __future__ import annotations

from dataclasses import dataclass

from ..body_profile.model import BodyMeasurements
from ..size_table.model import GarmentSizeTable, SizeEntry


@dataclass(frozen=True)
class DimensionRule:
    name: str
    tolerance_m: float
    weight: float


TUNIC_DIMENSION_RULES = (
    DimensionRule("chest_circumference", 0.040, 1.40),
    DimensionRule("waist_circumference", 0.045, 0.90),
    DimensionRule("hip_circumference", 0.050, 0.55),
    DimensionRule("shoulder_width", 0.018, 1.30),
    DimensionRule("front_torso_length", 0.020, 1.00),
    DimensionRule("back_torso_length", 0.020, 0.90),
    DimensionRule("armscye_depth", 0.012, 0.85),
    DimensionRule("stature", 0.060, 0.35),
)


def robust_error(normalized: float) -> float:
    value = abs(float(normalized))
    return 0.5 * value * value if value <= 1.0 else value - 0.5


def score_entry(
    body: BodyMeasurements,
    entry: SizeEntry,
    rules: tuple[DimensionRule, ...] = TUNIC_DIMENSION_RULES,
) -> dict:
    dimensions = {}
    weighted = 0.0
    total_weight = 0.0
    for rule in rules:
        delta = float(getattr(body, rule.name) - getattr(entry.target, rule.name))
        normalized = delta / rule.tolerance_m
        contribution = rule.weight * robust_error(normalized)
        dimensions[rule.name] = {
            "delta_m": delta,
            "normalized_error": normalized,
            "weighted_contribution": contribution,
        }
        weighted += contribution
        total_weight += rule.weight
    return {
        "size_id": entry.size_id,
        "score": weighted / total_weight,
        "dimensions": dimensions,
    }


def rank_sizes(body: BodyMeasurements, table: GarmentSizeTable) -> list[dict]:
    ranked = [score_entry(body, entry) for entry in table.sizes]
    return sorted(ranked, key=lambda item: (item["score"], table.ordered_ids().index(item["size_id"])))
