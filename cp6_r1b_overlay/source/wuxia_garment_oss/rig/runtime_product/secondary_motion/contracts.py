"""CP6 secondary-motion contracts and product profiles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class SecondaryMotionDomain:
    domain_id: str
    owner_component: str
    blend_shape_name: str
    driver_bones: tuple[str, ...]
    max_weight: float
    natural_frequency_hz: float
    damping_ratio: float
    driver_gain: float
    enabled_lods: tuple[str, ...] = ("LOD0", "LOD1", "LOD2")

    def validate(self) -> None:
        if not self.domain_id or not self.owner_component or not self.blend_shape_name:
            raise ValueError("secondary-motion domain identity is required")
        if not self.driver_bones:
            raise ValueError(f"{self.domain_id} requires at least one driver bone")
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError(f"invalid weight bound for {self.domain_id}")
        if not 0.25 <= self.natural_frequency_hz <= 12.0:
            raise ValueError(f"invalid natural frequency for {self.domain_id}")
        if not 0.15 <= self.damping_ratio <= 2.0:
            raise ValueError(f"invalid damping for {self.domain_id}")
        if not 0.0 < self.driver_gain <= 4.0:
            raise ValueError(f"invalid driver gain for {self.domain_id}")
        if set(self.enabled_lods) != {"LOD0", "LOD1", "LOD2"}:
            raise ValueError(f"secondary motion must transfer to all LODs: {self.domain_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["driver_bones"] = list(self.driver_bones)
        payload["enabled_lods"] = list(self.enabled_lods)
        return payload


@dataclass(frozen=True)
class SecondaryMotionProfile:
    profile_id: str
    product_id: str
    domains: tuple[SecondaryMotionDomain, ...]
    fixed_step_hz: int = 60
    substeps: int = 2

    def validate(self) -> None:
        if not self.profile_id or not self.product_id:
            raise ValueError("secondary-motion profile identity is required")
        ids = [item.domain_id for item in self.domains]
        shapes = [item.blend_shape_name for item in self.domains]
        if not ids or len(ids) != len(set(ids)) or len(shapes) != len(set(shapes)):
            raise ValueError(f"invalid domain identity in {self.profile_id}")
        if self.fixed_step_hz not in {30, 60, 120} or not 1 <= self.substeps <= 8:
            raise ValueError(f"invalid integration profile: {self.profile_id}")
        for item in self.domains:
            item.validate()

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "SecondaryMotionProfile/1",
            "profile_id": self.profile_id,
            "product_id": self.product_id,
            "fixed_step_hz": self.fixed_step_hz,
            "substeps": self.substeps,
            "integration": "BOUNDED_CRITICALLY_DAMPED_SPRING",
            "state_transfer": "BLEND_SHAPE_NAME_AND_DOMAIN_ID",
            "domains": [item.to_dict() for item in self.domains],
        }
        payload["profile_sha256"] = canonical_sha256(payload)
        return payload


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sleeve_domains() -> tuple[SecondaryMotionDomain, SecondaryMotionDomain]:
    return (
        SecondaryMotionDomain(
            domain_id="LEFT_SLEEVE_INERTIA",
            owner_component="LEFT_SLEEVE",
            blend_shape_name="SECONDARY_L_SLEEVE",
            driver_bones=("L_CLAVICLE", "L_UPPER_ARM", "L_FOREARM"),
            max_weight=0.72,
            natural_frequency_hz=3.1,
            damping_ratio=0.74,
            driver_gain=0.88,
        ),
        SecondaryMotionDomain(
            domain_id="RIGHT_SLEEVE_INERTIA",
            owner_component="RIGHT_SLEEVE",
            blend_shape_name="SECONDARY_R_SLEEVE",
            driver_bones=("R_CLAVICLE", "R_UPPER_ARM", "R_FOREARM"),
            max_weight=0.72,
            natural_frequency_hz=3.1,
            damping_ratio=0.74,
            driver_gain=0.88,
        ),
    )


def product_profile(product_id: str, product_kind: str) -> SecondaryMotionProfile:
    domains: tuple[SecondaryMotionDomain, ...] = _sleeve_domains()
    if product_kind == "STRAIGHT_SLEEVE_ROBE":
        domains += (
            SecondaryMotionDomain(
                domain_id="ROBE_HEM_INERTIA",
                owner_component="STRAIGHT_ROBE_SKIRT",
                blend_shape_name="SECONDARY_ROBE_HEM",
                driver_bones=("PELVIS", "L_THIGH", "R_THIGH"),
                max_weight=0.58,
                natural_frequency_hz=1.65,
                damping_ratio=0.82,
                driver_gain=0.62,
            ),
        )
    elif product_kind != "SLEEVED_TUNIC":
        raise ValueError(f"unsupported CP6 product kind: {product_kind}")
    return SecondaryMotionProfile(
        profile_id=f"{product_id}_SECONDARY_R1B",
        product_id=product_id,
        domains=domains,
    )
