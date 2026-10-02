"""Visual evidence, gate, and acceptance-profile contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from wuxia_garment_oss.pattern_components.model import canonical_sha256


@dataclass(frozen=True)
class ViewRequirement:
    view_id: str
    category: str
    minimum_width: int
    minimum_height: int
    body_visibility: str
    required_overlays: tuple[str, ...] = ()
    applicable_families: tuple[str, ...] = ("ALL",)

    def validate(self) -> None:
        if not self.view_id:
            raise ValueError("view identity is required")
        if self.category not in {"PRIMARY", "DETAIL", "DIAGNOSTIC", "MOTION", "LOD_ACTUAL_DISTANCE", "LOD_EQUAL_COVERAGE"}:
            raise ValueError(f"unsupported view category: {self.category}")
        if self.minimum_width < 1024 or self.minimum_height < 1024:
            raise ValueError("visual evidence must be at least 1024px in each dimension")
        if self.body_visibility not in {"VISIBLE", "OCCLUDED", "OPTIONAL", "NOT_APPLICABLE"}:
            raise ValueError("invalid body-visibility requirement")
        if len(self.required_overlays) != len(set(self.required_overlays)):
            raise ValueError(f"duplicate overlays on {self.view_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["required_overlays"] = list(self.required_overlays)
        payload["applicable_families"] = list(self.applicable_families)
        return payload


@dataclass(frozen=True)
class VisualGateSpec:
    gate_id: str
    severity: str
    metric: str
    comparator: str
    threshold: float | bool
    owner_scope: str
    repair_disposition: str
    evidence_categories: tuple[str, ...]

    def validate(self) -> None:
        if not self.gate_id or not self.metric or not self.owner_scope:
            raise ValueError("gate identity, metric, and owner scope are required")
        if self.severity not in {"ZERO_TOLERANCE", "BOUNDED", "INFORMATIONAL"}:
            raise ValueError(f"unsupported gate severity: {self.severity}")
        if self.comparator not in {"EQ", "LE", "GE", "TRUE"}:
            raise ValueError(f"unsupported gate comparator: {self.comparator}")
        if self.repair_disposition not in {"SAFE_AUTO", "GUIDED", "HOLD"}:
            raise ValueError(f"invalid repair disposition: {self.repair_disposition}")
        if not self.evidence_categories:
            raise ValueError(f"gate lacks evidence categories: {self.gate_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = asdict(self)
        payload["evidence_categories"] = list(self.evidence_categories)
        return payload


@dataclass(frozen=True)
class VisualAcceptanceProfile:
    profile_id: str
    revision: int
    required_views: tuple[ViewRequirement, ...]
    gates: tuple[VisualGateSpec, ...]
    acceptance_expression: str

    def validate(self) -> None:
        if not self.profile_id or self.revision < 1:
            raise ValueError("visual profile identity and positive revision are required")
        view_ids = [item.view_id for item in self.required_views]
        gate_ids = [item.gate_id for item in self.gates]
        if not view_ids or len(view_ids) != len(set(view_ids)):
            raise ValueError("visual profile has missing or duplicate view identities")
        if not gate_ids or len(gate_ids) != len(set(gate_ids)):
            raise ValueError("visual profile has missing or duplicate gate identities")
        for item in self.required_views:
            item.validate()
        for item in self.gates:
            item.validate()
        required_primary = {"front", "left", "right", "back", "front_three_quarter", "back_three_quarter"}
        if not required_primary.issubset(set(view_ids)):
            raise ValueError("visual profile is missing primary product views")
        required_categories = {"PRIMARY", "DETAIL", "DIAGNOSTIC", "MOTION", "LOD_ACTUAL_DISTANCE", "LOD_EQUAL_COVERAGE"}
        actual_categories = {item.category for item in self.required_views}
        if not required_categories.issubset(actual_categories):
            raise ValueError("visual profile is missing an evidence category")
        if self.acceptance_expression != "TECHNICAL_PASS_AND_VISUAL_PASS":
            raise ValueError("product acceptance must require technical and visual pass")

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "VisualAcceptanceProfile/1",
            "profile_id": self.profile_id,
            "revision": self.revision,
            "required_views": [item.to_dict() for item in self.required_views],
            "gates": [item.to_dict() for item in self.gates],
            "acceptance_expression": self.acceptance_expression,
        }
        payload["profile_sha256"] = canonical_sha256(payload)
        return payload
