"""Contracts for registry entries and deterministic outfit plans."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CoverageRegion:
    region_id: str
    thickness_m: float
    occludes_body: bool
    safety_band_m: float = 0.0

    def validate(self) -> None:
        if not self.region_id:
            raise ValueError("coverage region requires identity")
        if not 0.0 <= self.thickness_m <= 0.05:
            raise ValueError(f"invalid thickness for {self.region_id}")
        if not 0.0 <= self.safety_band_m <= 0.05:
            raise ValueError(f"invalid safety band for {self.region_id}")


@dataclass(frozen=True)
class GarmentFamilyEntry:
    garment_id: str
    family_id: str
    product_path: str
    product_sha256: str
    layer_class: str
    slots: tuple[str, ...]
    coverage: tuple[CoverageRegion, ...]
    required_bones: tuple[str, ...]
    incompatible_families: tuple[str, ...] = ()
    exclusive_slots: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def validate(self, root: Path | None = None) -> None:
        if not self.garment_id or not self.family_id:
            raise ValueError("garment identity is required")
        if self.layer_class not in {"BASE", "INNER", "MID", "OUTER", "ARMOR", "ACCESSORY"}:
            raise ValueError(f"invalid layer class: {self.layer_class}")
        if not self.slots or len(self.slots) != len(set(self.slots)):
            raise ValueError(f"invalid slots for {self.garment_id}")
        if len(self.coverage) != len({item.region_id for item in self.coverage}):
            raise ValueError(f"duplicate coverage region in {self.garment_id}")
        for item in self.coverage:
            item.validate()
        if root is not None:
            path = root / self.product_path
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != self.product_sha256:
                raise ValueError(f"product hash mismatch: {self.garment_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "garment_id": self.garment_id,
            "family_id": self.family_id,
            "product_path": self.product_path,
            "product_sha256": self.product_sha256,
            "layer_class": self.layer_class,
            "slots": list(self.slots),
            "exclusive_slots": list(self.exclusive_slots),
            "coverage": [asdict(item) for item in self.coverage],
            "required_bones": list(self.required_bones),
            "incompatible_families": list(self.incompatible_families),
            "metadata": self.metadata,
        }
        return payload


@dataclass(frozen=True)
class GarmentLibraryRegistry:
    registry_id: str
    entries: tuple[GarmentFamilyEntry, ...]
    layer_order: tuple[str, ...]
    regional_thickness_limits_m: dict[str, float]

    def validate(self, root: Path | None = None) -> None:
        if not self.registry_id:
            raise ValueError("registry identity is required")
        ids = [entry.garment_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate garment identity")
        if tuple(self.layer_order) != ("BASE", "INNER", "MID", "OUTER", "ARMOR", "ACCESSORY"):
            raise ValueError("unexpected layer order")
        if any(limit <= 0.0 for limit in self.regional_thickness_limits_m.values()):
            raise ValueError("thickness limits must be positive")
        for entry in self.entries:
            entry.validate(root)

    def entry(self, garment_id: str) -> GarmentFamilyEntry:
        for item in self.entries:
            if item.garment_id == garment_id:
                return item
        raise KeyError(garment_id)

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "GarmentLibraryRegistry/1",
            "registry_id": self.registry_id,
            "layer_order": list(self.layer_order),
            "regional_thickness_limits_m": dict(sorted(self.regional_thickness_limits_m.items())),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        payload["registry_sha256"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class OutfitAssemblyPlan:
    outfit_id: str
    target_rig_id: str
    garment_ids: tuple[str, ...]
    layer_order: tuple[str, ...]
    region_thickness_m: dict[str, float]
    hidden_body_regions: tuple[str, ...]
    preserved_safety_regions: tuple[str, ...]
    status: str
    rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = {
            "contract": "OutfitAssemblyPlan/1",
            "outfit_id": self.outfit_id,
            "target_rig_id": self.target_rig_id,
            "garment_ids": list(self.garment_ids),
            "layer_order": list(self.layer_order),
            "region_thickness_m": dict(sorted(self.region_thickness_m.items())),
            "hidden_body_regions": list(self.hidden_body_regions),
            "preserved_safety_regions": list(self.preserved_safety_regions),
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
        }
        payload["plan_sha256"] = canonical_sha256(payload)
        return payload
