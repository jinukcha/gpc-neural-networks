"""Immutable reference context consumed by the ratio parameter resolver."""
from __future__ import annotations

from dataclasses import dataclass

from ..model.quantity import TypedValue, quantity_dimension
from ..model.reference import ParameterReference
from .receipt import canonical_sha256


@dataclass(frozen=True)
class ParameterResolutionContext:
    context_id: str
    references: tuple[ParameterReference, ...]

    def validate(self) -> None:
        if not self.context_id:
            raise ValueError("context_id is required")
        paths = []
        for reference in self.references:
            reference.validate()
            paths.append(reference.qualified_path)
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate reference path")

    def reference(self, scope: str, path: str) -> ParameterReference:
        for item in self.references:
            if item.scope == scope and item.path == path:
                return item
        raise KeyError(f"missing reference: {scope}:{path}")

    def typed_environment(self) -> dict[str, TypedValue]:
        self.validate()
        return {
            item.qualified_path: TypedValue(item.value_si, quantity_dimension(item.quantity))
            for item in self.references
        }

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "ParameterResolutionContext/1",
            "context_id": self.context_id,
            "references": [item.to_dict() for item in sorted(self.references, key=lambda value: value.qualified_path)],
        }
        payload["context_sha256"] = canonical_sha256(payload)
        return payload
