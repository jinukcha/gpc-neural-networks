"""Compile compatible outfits and reject invalid combinations atomically."""
from __future__ import annotations

from collections import defaultdict

from .model import GarmentLibraryRegistry, OutfitAssemblyPlan


_LAYER_RANK = {name: index for index, name in enumerate(("BASE", "INNER", "MID", "OUTER", "ARMOR", "ACCESSORY"))}


def _slot_conflicts(entries) -> list[str]:
    owners: dict[str, list] = defaultdict(list)
    for entry in entries:
        for slot in entry.exclusive_slots:
            owners[slot].append(entry)
    return [
        f"EXCLUSIVE_SLOT_CONFLICT:{slot}:" + ",".join(item.garment_id for item in items)
        for slot, items in sorted(owners.items())
        if len(items) > 1
    ]


def _family_conflicts(entries) -> list[str]:
    failures = []
    for left in entries:
        for right in entries:
            if left.garment_id >= right.garment_id:
                continue
            if right.family_id in left.incompatible_families or left.family_id in right.incompatible_families:
                failures.append(f"INCOMPATIBLE_FAMILY:{left.family_id}:{right.family_id}")
    return failures


def _thickness(entries, limits: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    totals: dict[str, float] = defaultdict(float)
    for entry in entries:
        for region in entry.coverage:
            totals[region.region_id] += region.thickness_m
    failures = [
        f"THICKNESS_BUDGET:{region}:{value:.6f}>{limits.get(region, 0.0):.6f}"
        for region, value in sorted(totals.items())
        if value > limits.get(region, 0.0) + 1.0e-12
    ]
    return dict(totals), failures


def _layer_order(entries) -> tuple[str, ...]:
    return tuple(item.garment_id for item in sorted(entries, key=lambda value: (_LAYER_RANK[value.layer_class], value.garment_id)))


def _coverage(entries) -> tuple[tuple[str, ...], tuple[str, ...]]:
    hidden = set()
    safety = set()
    for entry in entries:
        for region in entry.coverage:
            if region.occludes_body:
                hidden.add(region.region_id)
            if region.safety_band_m > 0.0:
                safety.add(region.region_id)
    return tuple(sorted(hidden - safety)), tuple(sorted(safety))


def compile_outfit(
    registry: GarmentLibraryRegistry,
    outfit_id: str,
    target_rig_id: str,
    garment_ids: tuple[str, ...],
) -> OutfitAssemblyPlan:
    registry.validate()
    reasons: list[str] = []
    if len(garment_ids) != len(set(garment_ids)):
        reasons.append("DUPLICATE_GARMENT_ID")
    entries = []
    for garment_id in garment_ids:
        try:
            entries.append(registry.entry(garment_id))
        except KeyError:
            reasons.append(f"UNKNOWN_GARMENT:{garment_id}")
    if entries:
        reasons.extend(_slot_conflicts(entries))
        reasons.extend(_family_conflicts(entries))
        totals, thickness_failures = _thickness(entries, registry.regional_thickness_limits_m)
        reasons.extend(thickness_failures)
        hidden, safety = _coverage(entries)
    else:
        totals, hidden, safety = {}, (), ()
        reasons.append("EMPTY_OUTFIT")
    status = "ACCEPTED" if not reasons else "REJECTED_ATOMIC"
    return OutfitAssemblyPlan(
        outfit_id=outfit_id,
        target_rig_id=target_rig_id,
        garment_ids=tuple(garment_ids),
        layer_order=_layer_order(entries),
        region_thickness_m=totals,
        hidden_body_regions=hidden,
        preserved_safety_regions=safety,
        status=status,
        rejection_reasons=tuple(sorted(set(reasons))),
    )
