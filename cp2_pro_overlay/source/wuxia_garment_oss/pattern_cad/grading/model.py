"""Canonical industrial grading and notch contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..document.model import PatternDocument, canonical_sha256


@dataclass(frozen=True)
class GradePointRule:
    grade_point_id: str
    point_id: str
    semantic_role: str

    def validate(self, document: PatternDocument) -> None:
        if not self.grade_point_id or not self.semantic_role:
            raise ValueError("grade point identity and role are required")
        if self.point_id not in document.points:
            raise ValueError(f"unknown grade point owner: {self.point_id}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SizeGrade:
    size_id: str
    ordinal: int
    input_deltas: dict[str, float]

    def validate(self, document: PatternDocument) -> None:
        if not self.size_id:
            raise ValueError("size identity is required")
        unknown = sorted(set(self.input_deltas) - set(document.inputs))
        if unknown:
            raise ValueError(f"unknown grading inputs: {unknown}")
        if any(not isinstance(value, (int, float)) for value in self.input_deltas.values()):
            raise ValueError(f"non-numeric grade delta: {self.size_id}")

    def to_dict(self) -> dict:
        return {
            "size_id": self.size_id,
            "ordinal": int(self.ordinal),
            "input_deltas": {name: float(value) for name, value in sorted(self.input_deltas.items())},
        }


@dataclass(frozen=True)
class GradeRuleSetV2:
    grade_rule_set_id: str
    base_size_id: str
    ordered_sizes: tuple[SizeGrade, ...]
    grade_points: tuple[GradePointRule, ...]

    def validate(self, document: PatternDocument) -> None:
        if not self.grade_rule_set_id or not self.ordered_sizes:
            raise ValueError("grade rule set identity and sizes are required")
        size_ids = [item.size_id for item in self.ordered_sizes]
        if len(size_ids) != len(set(size_ids)):
            raise ValueError("duplicate size identities")
        if self.base_size_id not in size_ids:
            raise ValueError("base size is absent")
        ordinals = [item.ordinal for item in self.ordered_sizes]
        if ordinals != sorted(ordinals):
            raise ValueError("sizes must be ordered by ordinal")
        point_ids = [item.grade_point_id for item in self.grade_points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("duplicate grade point identities")
        for size in self.ordered_sizes:
            size.validate(document)
        for point in self.grade_points:
            point.validate(document)

    def size(self, size_id: str) -> SizeGrade:
        for item in self.ordered_sizes:
            if item.size_id == size_id:
                return item
        raise KeyError(size_id)

    def to_dict(self, document: PatternDocument) -> dict:
        self.validate(document)
        payload = {
            "contract": "GradeRuleSet/2",
            "grade_rule_set_id": self.grade_rule_set_id,
            "base_size_id": self.base_size_id,
            "ordered_size_ids": [item.size_id for item in self.ordered_sizes],
            "sizes": [item.to_dict() for item in self.ordered_sizes],
            "grade_points": [item.to_dict() for item in self.grade_points],
            "source_pattern_sha256": document.to_dict()["document_sha256"],
            "mesh_scaling": "FORBIDDEN",
        }
        payload["grade_rule_set_sha256"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class PatternNotch:
    notch_id: str
    curve_id: str
    arc_fraction: float
    semantic_role: str

    def validate(self, document: PatternDocument) -> None:
        if not self.notch_id or not self.semantic_role:
            raise ValueError("notch identity and role are required")
        if self.curve_id not in document.curves:
            raise ValueError(f"unknown notch curve: {self.curve_id}")
        if not 0.0 <= self.arc_fraction <= 1.0:
            raise ValueError(f"invalid notch fraction: {self.notch_id}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PatternNotchSet:
    notch_set_id: str
    notches: tuple[PatternNotch, ...]

    def validate(self, document: PatternDocument) -> None:
        ids = [item.notch_id for item in self.notches]
        if not self.notch_set_id or len(ids) != len(set(ids)):
            raise ValueError("invalid notch-set identity or duplicate notch")
        for notch in self.notches:
            notch.validate(document)

    def to_dict(self, document: PatternDocument) -> dict:
        self.validate(document)
        payload = {
            "contract": "PatternNotchSet/1",
            "notch_set_id": self.notch_set_id,
            "notches": [item.to_dict() for item in self.notches],
            "source_pattern_sha256": document.to_dict()["document_sha256"],
        }
        payload["notch_set_sha256"] = canonical_sha256(payload)
        return payload
