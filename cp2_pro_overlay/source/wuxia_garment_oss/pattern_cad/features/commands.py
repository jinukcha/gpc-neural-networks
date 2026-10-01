"""Atomic topology-aware feature commands."""
from __future__ import annotations

from dataclasses import dataclass

from ..document.model import PatternConstraint, PatternCurve, PatternDocument, PatternPoint
from ..document.resolver import require_resolved
from .model import PatternFeatureSpec


def _number(value: object, name: str) -> float:
    result = float(value)
    if not -10.0 < result < 10.0:
        raise ValueError(f"feature parameter out of range: {name}")
    return result


def _point(panel_id: str, point_id: str, x: float, y: float) -> PatternPoint:
    return PatternPoint(point_id, f"{x:.12g}", f"{y:.12g}", panel_id)


def _record_feature(document: PatternDocument, spec: PatternFeatureSpec, created: list[str]) -> None:
    registry = list(document.metadata.get("pattern_features", []))
    if any(item["feature_id"] == spec.feature_id for item in registry):
        raise ValueError(f"feature already exists: {spec.feature_id}")
    registry.append({**spec.to_dict(), "created_ids": sorted(created)})
    document.metadata = {**document.metadata, "pattern_features": registry}


def _panel_bounds(document: PatternDocument, panel_id: str) -> tuple[float, float, float, float]:
    if panel_id not in document.panel_ids:
        raise ValueError(f"unknown feature panel: {panel_id}")
    resolved = require_resolved(document)
    points = [
        value for point_id, value in resolved["points"].items()
        if document.points[point_id].panel_id == panel_id
    ]
    if not points:
        raise ValueError(f"panel has no points: {panel_id}")
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    return min(xs), max(xs), min(ys), max(ys)


@dataclass(frozen=True)
class DartCommand:
    spec: PatternFeatureSpec

    @property
    def command_type(self) -> str:
        return "ADD_DART"

    def to_dict(self) -> dict:
        return self.spec.to_dict()

    def apply(self, document: PatternDocument) -> None:
        panel = self.spec.owner_id
        x_min, x_max, y_min, y_max = _panel_bounds(document, panel)
        center = _number(self.spec.parameters.get("center_x", 0.0), "center_x")
        waist_y = _number(self.spec.parameters.get("waist_y", 0.0), "waist_y")
        apex_x = _number(self.spec.parameters.get("apex_x", center), "apex_x")
        apex_y = _number(self.spec.parameters.get("apex_y", 0.18), "apex_y")
        intake = _number(self.spec.parameters.get("intake_m", 0.024), "intake_m")
        if intake <= 0.0 or not (x_min <= center <= x_max and y_min <= apex_y <= y_max):
            raise ValueError("dart lies outside its owner panel or has non-positive intake")
        prefix = f"{panel}.feature.{self.spec.feature_id}"
        left_id, right_id, apex_id = f"{prefix}.leg_left", f"{prefix}.leg_right", f"{prefix}.apex"
        document.points[left_id] = _point(panel, left_id, center - intake * 0.5, waist_y)
        document.points[right_id] = _point(panel, right_id, center + intake * 0.5, waist_y)
        document.points[apex_id] = _point(panel, apex_id, apex_x, apex_y)
        for name, leg in (("left_leg", left_id), ("right_leg", right_id)):
            curve_id = f"{prefix}.{name}"
            document.curves[curve_id] = PatternCurve(
                curve_id, panel, "LINE", (leg, apex_id), "DART_LEG", "INTERNAL"
            )
        target_id = f"{self.spec.feature_id}_intake"
        document.inputs[target_id] = intake
        constraint_id = f"{prefix}.intake"
        document.constraints[constraint_id] = PatternConstraint(
            constraint_id, "FIXED_DISTANCE", (left_id, right_id), "HARD", 1.0e-10, target_id
        )
        _record_feature(document, self.spec, [left_id, right_id, apex_id, constraint_id])


@dataclass(frozen=True)
class PleatCommand:
    spec: PatternFeatureSpec

    @property
    def command_type(self) -> str:
        return "ADD_PLEAT"

    def to_dict(self) -> dict:
        return self.spec.to_dict()

    def apply(self, document: PatternDocument) -> None:
        panel = self.spec.owner_id
        x_min, x_max, y_min, y_max = _panel_bounds(document, panel)
        center = _number(self.spec.parameters.get("center_x", 0.0), "center_x")
        depth = _number(self.spec.parameters.get("depth_m", 0.018), "depth_m")
        bottom_margin = _number(self.spec.parameters.get("bottom_margin_m", 0.04), "bottom_margin_m")
        if depth <= 0.0 or bottom_margin < 0.0 or not x_min < center - depth < center + depth < x_max:
            raise ValueError("invalid pleat depth or position")
        y_top = y_max
        y_bottom = y_min + bottom_margin
        if y_bottom >= y_top:
            raise ValueError("pleat guide has no vertical extent")
        prefix = f"{panel}.feature.{self.spec.feature_id}"
        created: list[str] = []
        for role, x in (("fold_in", center - depth), ("fold_center", center), ("fold_out", center + depth)):
            top_id, bottom_id = f"{prefix}.{role}.top", f"{prefix}.{role}.bottom"
            curve_id = f"{prefix}.{role}"
            constraint_id = f"{curve_id}.vertical"
            document.points[top_id] = _point(panel, top_id, x, y_top)
            document.points[bottom_id] = _point(panel, bottom_id, x, y_bottom)
            document.curves[curve_id] = PatternCurve(
                curve_id, panel, "LINE", (top_id, bottom_id), "PLEAT_FOLD", "INTERNAL"
            )
            document.constraints[constraint_id] = PatternConstraint(
                constraint_id, "VERTICAL", (top_id, bottom_id), "HARD", 1.0e-10
            )
            created.extend((top_id, bottom_id, curve_id, constraint_id))
        _record_feature(document, self.spec, created)


@dataclass(frozen=True)
class GatherCommand:
    spec: PatternFeatureSpec

    @property
    def command_type(self) -> str:
        return "ADD_GATHER"

    def to_dict(self) -> dict:
        return self.spec.to_dict()

    def apply(self, document: PatternDocument) -> None:
        curve_id = self.spec.owner_id
        if curve_id not in document.curves:
            raise ValueError(f"unknown gathered curve: {curve_id}")
        ratio = _number(self.spec.parameters.get("ratio", 1.12), "ratio")
        start = _number(self.spec.parameters.get("start_fraction", 0.0), "start_fraction")
        end = _number(self.spec.parameters.get("end_fraction", 1.0), "end_fraction")
        if not 1.0 < ratio <= 2.5 or not 0.0 <= start < end <= 1.0:
            raise ValueError("invalid gather ratio or arc interval")
        _record_feature(document, self.spec, [curve_id])


@dataclass(frozen=True)
class GussetCommand:
    spec: PatternFeatureSpec

    @property
    def command_type(self) -> str:
        return "ADD_GUSSET"

    def to_dict(self) -> dict:
        return self.spec.to_dict()

    def apply(self, document: PatternDocument) -> None:
        panel = self.spec.owner_id
        width = _number(self.spec.parameters.get("width_m", 0.09), "width_m")
        height = _number(self.spec.parameters.get("height_m", 0.12), "height_m")
        center_x = _number(self.spec.parameters.get("center_x", 0.0), "center_x")
        center_y = _number(self.spec.parameters.get("center_y", 0.68), "center_y")
        if panel in document.panel_ids or width <= 0.0 or height <= 0.0:
            raise ValueError("gusset panel already exists or has invalid size")
        document.panel_ids = (*document.panel_ids, panel)
        prefix = f"{panel}.feature.{self.spec.feature_id}"
        coordinates = {
            "left": (center_x - width * 0.5, center_y),
            "top": (center_x, center_y + height * 0.5),
            "right": (center_x + width * 0.5, center_y),
            "bottom": (center_x, center_y - height * 0.5),
        }
        point_ids = {name: f"{prefix}.{name}" for name in coordinates}
        for name, coordinate in coordinates.items():
            document.points[point_ids[name]] = _point(panel, point_ids[name], *coordinate)
        edges = (("left", "top"), ("top", "right"), ("right", "bottom"), ("bottom", "left"))
        created = list(point_ids.values())
        for index, pair in enumerate(edges):
            curve_id = f"{prefix}.edge_{index}"
            document.curves[curve_id] = PatternCurve(
                curve_id, panel, "LINE", tuple(point_ids[name] for name in pair),
                f"GUSSET_EDGE_{index}", "SEWN"
            )
            created.append(curve_id)
        symmetry_id = f"{prefix}.symmetry"
        vertical_id = f"{prefix}.vertical"
        document.constraints[symmetry_id] = PatternConstraint(
            symmetry_id, "SYMMETRY_X", (point_ids["left"], point_ids["right"]),
            "HARD", 1.0e-10
        )
        document.constraints[vertical_id] = PatternConstraint(
            vertical_id, "VERTICAL", (point_ids["top"], point_ids["bottom"]),
            "HARD", 1.0e-10
        )
        created.extend((symmetry_id, vertical_id))
        _record_feature(document, self.spec, created)


def command_for(spec: PatternFeatureSpec):
    return {
        "DART": DartCommand,
        "PLEAT": PleatCommand,
        "GATHER": GatherCommand,
        "GUSSET": GussetCommand,
    }[spec.feature_type](spec)
