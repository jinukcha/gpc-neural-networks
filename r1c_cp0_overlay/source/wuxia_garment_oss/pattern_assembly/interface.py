"""Boundary-to-boundary construction interface contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from wuxia_garment_oss.pattern_components.model import canonical_sha256


@dataclass(frozen=True)
class InterfaceEndpoint:
    component_instance_id: str
    boundary_id: str

    def validate(self) -> None:
        if not self.component_instance_id or not self.boundary_id:
            raise ValueError("interface endpoint requires component instance and boundary")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ComponentInterfaceSpec:
    interface_id: str
    endpoint_a: InterfaceEndpoint
    endpoint_b: InterfaceEndpoint
    semantic_role: str
    orientation_relation: str
    length_policy: str
    ratio_min: float
    ratio_max: float
    notch_policy: str
    seam_allowance_policy: str
    turn_of_cloth_policy: str
    construction_predecessors: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.interface_id or not self.semantic_role:
            raise ValueError("interface identity and semantic role are required")
        self.endpoint_a.validate()
        self.endpoint_b.validate()
        if self.endpoint_a == self.endpoint_b:
            raise ValueError("an interface cannot connect a boundary to itself")
        if self.orientation_relation not in {"OPPOSED", "SAME_FOR_FOLD"}:
            raise ValueError(f"unsupported orientation relation: {self.orientation_relation}")
        if self.length_policy not in {"EQUAL", "BOUNDED_EASE", "GATHERED"}:
            raise ValueError(f"unsupported length policy: {self.length_policy}")
        if not 0.0 < self.ratio_min <= self.ratio_max:
            raise ValueError("invalid interface ratio bounds")
        if self.notch_policy not in {"EXACT_ID", "NORMALIZED_ARC", "NONE"}:
            raise ValueError("invalid notch policy")
        if self.seam_allowance_policy not in {"OWNER_A", "OWNER_B", "MATCHED", "EXPLICIT"}:
            raise ValueError("invalid seam allowance policy")
        if self.turn_of_cloth_policy not in {"NONE", "MATERIAL_DERIVED", "EXPLICIT"}:
            raise ValueError("invalid turn-of-cloth policy")
        if len(self.construction_predecessors) != len(set(self.construction_predecessors)):
            raise ValueError("duplicate construction predecessor")

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "ComponentInterfaceSpec/1",
            "interface_id": self.interface_id,
            "endpoint_a": self.endpoint_a.to_dict(),
            "endpoint_b": self.endpoint_b.to_dict(),
            "semantic_role": self.semantic_role,
            "orientation_relation": self.orientation_relation,
            "length_policy": self.length_policy,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "notch_policy": self.notch_policy,
            "seam_allowance_policy": self.seam_allowance_policy,
            "turn_of_cloth_policy": self.turn_of_cloth_policy,
            "construction_predecessors": list(self.construction_predecessors),
        }
        payload["interface_sha256"] = canonical_sha256(payload)
        return payload
