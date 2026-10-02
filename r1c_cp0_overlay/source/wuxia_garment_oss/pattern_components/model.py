"""Stable contracts for reusable 2D garment-pattern components."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


_ALLOWED_DISPOSITIONS = {"OPEN", "SEWN", "FINISHED", "FOLD", "CUT_INTERNAL"}
_ALLOWED_ORIENTATIONS = {"FORWARD", "REVERSE"}
_ALLOWED_LAYER_ROLES = {"SHELL", "LINING", "FACING", "INTERFACING", "HARDWARE_MOUNT"}


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class BoundarySpec:
    boundary_id: str
    semantic_role: str
    interface_family: str
    curve_kind: str
    orientation: str
    disposition: str
    notch_ids: tuple[str, ...] = ()
    parameter_bindings: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.boundary_id or not self.semantic_role or not self.interface_family:
            raise ValueError("boundary identity, role, and interface family are required")
        if self.orientation not in _ALLOWED_ORIENTATIONS:
            raise ValueError(f"unsupported boundary orientation: {self.orientation}")
        if self.disposition not in _ALLOWED_DISPOSITIONS:
            raise ValueError(f"unsupported boundary disposition: {self.disposition}")
        if self.curve_kind not in {"LINE", "CIRCULAR_ARC", "QUADRATIC_BEZIER", "CUBIC_BEZIER", "SPLINE"}:
            raise ValueError(f"unsupported curve kind: {self.curve_kind}")
        if len(self.notch_ids) != len(set(self.notch_ids)):
            raise ValueError(f"duplicate notch identity on {self.boundary_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["notch_ids"] = list(self.notch_ids)
        payload["parameter_bindings"] = list(self.parameter_bindings)
        return payload


@dataclass(frozen=True)
class PatternComponentDefinition:
    component_id: str
    revision: int
    category: str
    family: str
    layer_role: str
    geometry_authority: str
    boundaries: tuple[BoundarySpec, ...]
    landmarks: tuple[str, ...]
    construction_features: tuple[str, ...]
    parameter_ports: tuple[str, ...]
    interface_ports: tuple[str, ...]
    mirror_policy: str
    topology_change_policy: str

    def validate(self) -> None:
        if not self.component_id or self.revision < 1:
            raise ValueError("component identity and positive revision are required")
        if self.layer_role not in _ALLOWED_LAYER_ROLES:
            raise ValueError(f"unsupported layer role: {self.layer_role}")
        if self.geometry_authority != "EXACT_2D_PATTERN":
            raise ValueError("R1C components must be owned by exact 2D pattern authority")
        if self.mirror_policy not in {"NONE", "MIRROR_ALLOWED", "MIRROR_REQUIRED"}:
            raise ValueError(f"unsupported mirror policy: {self.mirror_policy}")
        if self.topology_change_policy not in {"FORBIDDEN", "GUIDED_ONLY"}:
            raise ValueError("topology change policy must be explicit")
        boundary_ids = [item.boundary_id for item in self.boundaries]
        if not boundary_ids or len(boundary_ids) != len(set(boundary_ids)):
            raise ValueError(f"component has missing or duplicate boundaries: {self.component_id}")
        for item in self.boundaries:
            item.validate()
        if not set(self.interface_ports).issubset(boundary_ids):
            raise ValueError(f"interface ports reference unknown boundaries: {self.component_id}")
        for values, label in (
            (self.landmarks, "landmark"),
            (self.construction_features, "construction feature"),
            (self.parameter_ports, "parameter port"),
            (self.interface_ports, "interface port"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} in {self.component_id}")

    def boundary(self, boundary_id: str) -> BoundarySpec:
        for item in self.boundaries:
            if item.boundary_id == boundary_id:
                return item
        raise KeyError(f"unknown boundary {self.component_id}.{boundary_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "PatternComponentDefinition/1",
            "component_id": self.component_id,
            "revision": self.revision,
            "category": self.category,
            "family": self.family,
            "layer_role": self.layer_role,
            "geometry_authority": self.geometry_authority,
            "boundaries": [item.to_dict() for item in self.boundaries],
            "landmarks": list(self.landmarks),
            "construction_features": list(self.construction_features),
            "parameter_ports": list(self.parameter_ports),
            "interface_ports": list(self.interface_ports),
            "mirror_policy": self.mirror_policy,
            "topology_change_policy": self.topology_change_policy,
            "three_dimensional_primitive_authority": False,
        }
        payload["component_sha256"] = canonical_sha256(payload)
        return payload
