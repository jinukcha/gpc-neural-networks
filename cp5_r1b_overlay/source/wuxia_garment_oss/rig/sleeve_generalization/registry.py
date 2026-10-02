"""Extend the immutable CP4 registry with two CP5 sleeve-family products."""
from __future__ import annotations

from pathlib import Path

from wuxia_garment_oss.rig.library.model import CoverageRegion, GarmentFamilyEntry, GarmentLibraryRegistry


def _coverage(payload: dict) -> tuple[CoverageRegion, ...]:
    return tuple(CoverageRegion(**item) for item in payload["coverage"])


def _entry(payload: dict) -> GarmentFamilyEntry:
    return GarmentFamilyEntry(
        garment_id=payload["garment_id"],
        family_id=payload["family_id"],
        product_path=payload["product_path"],
        product_sha256=payload["product_sha256"],
        layer_class=payload["layer_class"],
        slots=tuple(payload["slots"]),
        exclusive_slots=tuple(payload.get("exclusive_slots", ())),
        coverage=_coverage(payload),
        required_bones=tuple(payload["required_bones"]),
        incompatible_families=tuple(payload.get("incompatible_families", ())),
        metadata=dict(payload.get("metadata", {})),
    )


def _cp5_entry(product: dict, bones: tuple[str, ...], robe: bool) -> GarmentFamilyEntry:
    if robe:
        coverage = (
            CoverageRegion("CHEST", 0.0052, True, 0.008),
            CoverageRegion("TORSO", 0.0052, True),
            CoverageRegion("WAIST", 0.0055, True, 0.010),
            CoverageRegion("PELVIS", 0.0055, True, 0.012),
            CoverageRegion("LEFT_THIGH", 0.0040, True, 0.008),
            CoverageRegion("RIGHT_THIGH", 0.0040, True, 0.008),
            CoverageRegion("LEFT_ARM", 0.0046, True, 0.006),
            CoverageRegion("RIGHT_ARM", 0.0046, True, 0.006),
            CoverageRegion("LEFT_FOREARM", 0.0042, True, 0.006),
            CoverageRegion("RIGHT_FOREARM", 0.0042, True, 0.006),
        )
        family, layer = "STRAIGHT_SLEEVE_ROBE", "OUTER"
        slots = ("UPPER_BODY", "FULL_LENGTH_ROBE", "ARMS")
        incompatible = ("SLEEVELESS_TUNIC", "SLEEVED_TUNIC")
    else:
        coverage = (
            CoverageRegion("CHEST", 0.0046, True, 0.008),
            CoverageRegion("TORSO", 0.0046, True),
            CoverageRegion("WAIST", 0.0048, True, 0.010),
            CoverageRegion("LEFT_ARM", 0.0042, True, 0.006),
            CoverageRegion("RIGHT_ARM", 0.0042, True, 0.006),
            CoverageRegion("LEFT_FOREARM", 0.0038, True, 0.006),
            CoverageRegion("RIGHT_FOREARM", 0.0038, True, 0.006),
        )
        family, layer = "SLEEVED_TUNIC", "MID"
        slots = ("UPPER_BODY", "LONG_TOP", "ARMS")
        incompatible = ("SLEEVELESS_TUNIC", "STRAIGHT_SLEEVE_ROBE")
    return GarmentFamilyEntry(
        garment_id=product["product_id"],
        family_id=family,
        product_path=product["path"],
        product_sha256=product["glb_sha256"],
        layer_class=layer,
        slots=slots,
        exclusive_slots=("UPPER_BODY_PRIMARY",),
        coverage=coverage,
        required_bones=bones,
        incompatible_families=incompatible,
        metadata={
            "source_checkpoint": "GARMENT_CAD_PRO_R1B_CP5",
            "rigged_product": True,
            "direct_arm_measurement": True,
            "corrective_target_names": product["corrective_target_names"],
        },
    )


def extend_registry(root: Path, cp4_registry: dict, products: tuple[dict, dict], bones: tuple[str, ...]) -> GarmentLibraryRegistry:
    entries = tuple(_entry(item) for item in cp4_registry["entries"])
    additions = (_cp5_entry(products[0], bones, False), _cp5_entry(products[1], bones, True))
    limits = dict(cp4_registry["regional_thickness_limits_m"])
    limits.update({
        "LEFT_ARM": 0.012,
        "RIGHT_ARM": 0.012,
        "LEFT_FOREARM": 0.010,
        "RIGHT_FOREARM": 0.010,
    })
    registry = GarmentLibraryRegistry(
        registry_id="GARMENT_LIBRARY_R1B_CP5",
        entries=entries + additions,
        layer_order=tuple(cp4_registry["layer_order"]),
        regional_thickness_limits_m=limits,
    )
    registry.validate(root)
    return registry
