"""Canonical PatternDocument/1 authority."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PatternPoint:
    point_id: str
    x_expression: str
    y_expression: str
    panel_id: str

    def validate(self) -> None:
        if not self.point_id or not self.panel_id:
            raise ValueError("point identity and panel ownership are required")
        if not self.x_expression or not self.y_expression:
            raise ValueError(f"point expressions are required: {self.point_id}")


@dataclass(frozen=True)
class PatternCurve:
    curve_id: str
    panel_id: str
    curve_type: str
    point_ids: tuple[str, ...]
    boundary_role: str
    disposition: str

    def validate(self) -> None:
        expected = {"LINE": 2, "QUADRATIC_BEZIER": 3, "CUBIC_BEZIER": 4}
        if self.curve_type not in expected:
            raise ValueError(f"unsupported curve type: {self.curve_type}")
        if len(self.point_ids) != expected[self.curve_type]:
            raise ValueError(f"curve point count mismatch: {self.curve_id}")
        if self.disposition not in {"OPEN", "SEWN", "INTERNAL"}:
            raise ValueError(f"unsupported curve disposition: {self.disposition}")


@dataclass(frozen=True)
class PatternConstraint:
    constraint_id: str
    kind: str
    point_ids: tuple[str, ...]
    strength: str
    tolerance: float
    target_expression: str | None = None

    def validate(self) -> None:
        if self.kind not in {
            "COINCIDENT",
            "HORIZONTAL",
            "VERTICAL",
            "FIXED_DISTANCE",
            "MIN_DISTANCE",
            "SYMMETRY_X",
        }:
            raise ValueError(f"unsupported constraint kind: {self.kind}")
        if self.strength not in {"HARD", "SOFT"}:
            raise ValueError(f"unsupported constraint strength: {self.strength}")
        if self.tolerance < 0.0:
            raise ValueError("constraint tolerance must be non-negative")
        if self.kind in {"FIXED_DISTANCE", "MIN_DISTANCE"} and not self.target_expression:
            raise ValueError(f"target expression required: {self.constraint_id}")


@dataclass
class PatternDocument:
    document_id: str
    revision: int
    parent_revision: int | None
    garment_design_id: str
    body_profile_sha256: str
    inputs: dict[str, float]
    expressions: dict[str, str]
    points: dict[str, PatternPoint]
    curves: dict[str, PatternCurve]
    constraints: dict[str, PatternConstraint]
    panel_ids: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def validate_structure(self) -> None:
        if not self.document_id or self.revision < 0:
            raise ValueError("invalid document identity or revision")
        if len(self.panel_ids) != len(set(self.panel_ids)):
            raise ValueError("duplicate panel IDs")
        for point in self.points.values():
            point.validate()
            if point.panel_id not in self.panel_ids:
                raise ValueError(f"unknown point panel: {point.point_id}")
        for curve in self.curves.values():
            curve.validate()
            if curve.panel_id not in self.panel_ids:
                raise ValueError(f"unknown curve panel: {curve.curve_id}")
            missing = sorted(set(curve.point_ids) - set(self.points))
            if missing:
                raise ValueError(f"unknown curve points: {curve.curve_id}: {missing}")
        for constraint in self.constraints.values():
            constraint.validate()
            missing = sorted(set(constraint.point_ids) - set(self.points))
            if missing:
                raise ValueError(f"unknown constraint points: {constraint.constraint_id}: {missing}")

    def clone(self) -> "PatternDocument":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        self.validate_structure()
        payload = {
            "contract": "PatternDocument/1",
            "document_id": self.document_id,
            "revision": self.revision,
            "parent_revision": self.parent_revision,
            "garment_design_id": self.garment_design_id,
            "body_profile_sha256": self.body_profile_sha256,
            "inputs": {name: self.inputs[name] for name in sorted(self.inputs)},
            "expressions": {name: self.expressions[name] for name in sorted(self.expressions)},
            "points": {name: asdict(self.points[name]) for name in sorted(self.points)},
            "curves": {
                name: {
                    **asdict(self.curves[name]),
                    "point_ids": list(self.curves[name].point_ids),
                }
                for name in sorted(self.curves)
            },
            "constraints": {
                name: {
                    **asdict(self.constraints[name]),
                    "point_ids": list(self.constraints[name].point_ids),
                }
                for name in sorted(self.constraints)
            },
            "panel_ids": list(self.panel_ids),
            "metadata": dict(self.metadata),
        }
        payload["document_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PatternDocument":
        points = {
            str(name): PatternPoint(**value)
            for name, value in dict(payload["points"]).items()
        }
        curves = {
            str(name): PatternCurve(
                curve_id=str(value["curve_id"]),
                panel_id=str(value["panel_id"]),
                curve_type=str(value["curve_type"]),
                point_ids=tuple(str(item) for item in value["point_ids"]),
                boundary_role=str(value["boundary_role"]),
                disposition=str(value["disposition"]),
            )
            for name, value in dict(payload["curves"]).items()
        }
        constraints = {
            str(name): PatternConstraint(
                constraint_id=str(value["constraint_id"]),
                kind=str(value["kind"]),
                point_ids=tuple(str(item) for item in value["point_ids"]),
                strength=str(value["strength"]),
                tolerance=float(value["tolerance"]),
                target_expression=(
                    None if value.get("target_expression") is None
                    else str(value["target_expression"])
                ),
            )
            for name, value in dict(payload["constraints"]).items()
        }
        result = cls(
            document_id=str(payload["document_id"]),
            revision=int(payload["revision"]),
            parent_revision=(
                None if payload.get("parent_revision") is None
                else int(payload["parent_revision"])
            ),
            garment_design_id=str(payload["garment_design_id"]),
            body_profile_sha256=str(payload["body_profile_sha256"]),
            inputs={str(k): float(v) for k, v in dict(payload["inputs"]).items()},
            expressions={str(k): str(v) for k, v in dict(payload["expressions"]).items()},
            points=points,
            curves=curves,
            constraints=constraints,
            panel_ids=tuple(str(item) for item in payload["panel_ids"]),
            metadata=dict(payload.get("metadata", {})),
        )
        result.validate_structure()
        return result
