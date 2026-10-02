"""Garment assembly recipe contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from wuxia_garment_oss.pattern_components.model import canonical_sha256


@dataclass(frozen=True)
class ComponentInstance:
    instance_id: str
    component_id: str
    mirror_source_instance_id: str | None
    parameter_bindings: dict[str, str]

    def validate(self) -> None:
        if not self.instance_id or not self.component_id:
            raise ValueError("component instance requires stable identities")
        if self.mirror_source_instance_id == self.instance_id:
            raise ValueError("component instance cannot mirror itself")
        if any(not key or not value for key, value in self.parameter_bindings.items()):
            raise ValueError(f"invalid parameter binding on {self.instance_id}")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class GarmentAssemblyRecipe:
    recipe_id: str
    garment_family: str
    component_instances: tuple[ComponentInstance, ...]
    interface_ids: tuple[str, ...]
    required_layers: tuple[str, ...]
    allowed_completion_modes: tuple[str, ...]
    topology_change_policy: str
    technical_profile_id: str
    visual_profile_id: str

    def validate(self) -> None:
        if not self.recipe_id or not self.garment_family:
            raise ValueError("recipe identity and garment family are required")
        instance_ids = [item.instance_id for item in self.component_instances]
        if not instance_ids or len(instance_ids) != len(set(instance_ids)):
            raise ValueError("recipe has missing or duplicate component instances")
        for item in self.component_instances:
            item.validate()
        if len(self.interface_ids) != len(set(self.interface_ids)):
            raise ValueError("recipe has duplicate interface identities")
        if not self.required_layers or len(self.required_layers) != len(set(self.required_layers)):
            raise ValueError("recipe must declare unique required layers")
        allowed = {"STRICT", "GUIDED", "SAFE_AUTO"}
        if not self.allowed_completion_modes or not set(self.allowed_completion_modes).issubset(allowed):
            raise ValueError("recipe has invalid completion modes")
        if self.topology_change_policy not in {"HOLD", "GUIDED_DECISION_REQUIRED"}:
            raise ValueError("topology-changing completion must not auto-commit")
        if not self.technical_profile_id or not self.visual_profile_id:
            raise ValueError("technical and visual profiles are required")
        instances = {item.instance_id for item in self.component_instances}
        for item in self.component_instances:
            if item.mirror_source_instance_id and item.mirror_source_instance_id not in instances:
                raise ValueError(f"unknown mirror source: {item.mirror_source_instance_id}")

    def instance_map(self) -> dict[str, ComponentInstance]:
        return {item.instance_id: item for item in self.component_instances}

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "GarmentAssemblyRecipe/1",
            "recipe_id": self.recipe_id,
            "garment_family": self.garment_family,
            "component_instances": [item.to_dict() for item in self.component_instances],
            "interface_ids": list(self.interface_ids),
            "required_layers": list(self.required_layers),
            "allowed_completion_modes": list(self.allowed_completion_modes),
            "topology_change_policy": self.topology_change_policy,
            "technical_profile_id": self.technical_profile_id,
            "visual_profile_id": self.visual_profile_id,
            "three_dimensional_primitive_fallback": False,
        }
        payload["recipe_sha256"] = canonical_sha256(payload)
        return payload
