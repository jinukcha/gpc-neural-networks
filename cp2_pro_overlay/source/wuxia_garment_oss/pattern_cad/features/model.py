"""Pattern feature graph authority for topology-changing operations."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..document.model import canonical_sha256


SUPPORTED_FEATURES = {"DART", "PLEAT", "GATHER", "GUSSET"}


@dataclass(frozen=True)
class PatternFeatureSpec:
    feature_id: str
    feature_type: str
    owner_id: str
    parameters: dict[str, object]
    depends_on: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.feature_id or not self.owner_id:
            raise ValueError("feature identity and owner are required")
        if self.feature_type not in SUPPORTED_FEATURES:
            raise ValueError(f"unsupported feature type: {self.feature_type}")
        if self.feature_id in self.depends_on:
            raise ValueError(f"feature cannot depend on itself: {self.feature_id}")

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class PatternFeatureGraph:
    feature_graph_id: str
    features: tuple[PatternFeatureSpec, ...]

    def validate(self) -> None:
        if not self.feature_graph_id or not self.features:
            raise ValueError("feature graph identity and features are required")
        ids = [item.feature_id for item in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate feature identities")
        known = set(ids)
        seen: set[str] = set()
        for feature in self.features:
            feature.validate()
            missing = sorted(set(feature.depends_on) - known)
            if missing:
                raise ValueError(f"unknown feature dependencies: {feature.feature_id}: {missing}")
            unresolved = sorted(set(feature.depends_on) - seen)
            if unresolved:
                raise ValueError(f"feature order violates dependencies: {feature.feature_id}: {unresolved}")
            seen.add(feature.feature_id)

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "PatternFeatureGraph/1",
            "feature_graph_id": self.feature_graph_id,
            "features": [item.to_dict() for item in self.features],
            "triangulation_executed": False,
            "warp_simulation_executed": False,
        }
        payload["feature_graph_sha256"] = canonical_sha256(payload)
        return payload
