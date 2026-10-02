"""Registry ownership for reusable pattern components."""
from __future__ import annotations

from dataclasses import dataclass

from .model import PatternComponentDefinition, canonical_sha256


@dataclass(frozen=True)
class PatternComponentRegistry:
    registry_id: str
    components: tuple[PatternComponentDefinition, ...]

    def validate(self) -> None:
        if not self.registry_id:
            raise ValueError("registry identity is required")
        ids = [item.component_id for item in self.components]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("component registry contains missing or duplicate identities")
        for item in self.components:
            item.validate()

    def get(self, component_id: str) -> PatternComponentDefinition:
        for item in self.components:
            if item.component_id == component_id:
                return item
        raise KeyError(component_id)

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "PatternComponentRegistry/1",
            "registry_id": self.registry_id,
            "component_count": len(self.components),
            "components": [item.to_dict() for item in self.components],
        }
        payload["registry_sha256"] = canonical_sha256(payload)
        return payload
