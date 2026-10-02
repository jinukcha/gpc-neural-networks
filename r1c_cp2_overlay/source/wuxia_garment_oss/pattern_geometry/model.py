"""Exact 2D curve and component authority for R1C pattern products."""
from __future__ import annotations

from dataclasses import dataclass
import math

from wuxia_garment_oss.pattern_components.model import canonical_sha256


_CURVE_POINT_COUNT = {"LINE": 2, "QUADRATIC_BEZIER": 3, "CUBIC_BEZIER": 4}


@dataclass(frozen=True)
class Point2:
    x: float
    y: float

    def validate(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("pattern point must be finite")

    def to_list(self) -> list[float]:
        self.validate()
        return [self.x, self.y]


@dataclass(frozen=True)
class CurveSegment:
    segment_id: str
    kind: str
    points: tuple[Point2, ...]

    def validate(self) -> None:
        if not self.segment_id or self.kind not in _CURVE_POINT_COUNT:
            raise ValueError(f"invalid curve identity or kind: {self.segment_id}")
        if len(self.points) != _CURVE_POINT_COUNT[self.kind]:
            raise ValueError(f"wrong control-point count for {self.segment_id}")
        for point in self.points:
            point.validate()
        if self.points[0] == self.points[-1]:
            raise ValueError(f"zero-span curve: {self.segment_id}")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "points_m": [point.to_list() for point in self.points],
        }


@dataclass(frozen=True)
class BoundaryGeometry:
    boundary_id: str
    semantic_role: str
    orientation: str
    disposition: str
    segment_ids: tuple[str, ...]

    def validate(self) -> None:
        if not self.boundary_id or not self.semantic_role or not self.segment_ids:
            raise ValueError("boundary geometry requires identity, role, and segments")
        if self.orientation not in {"FORWARD", "REVERSE"}:
            raise ValueError(f"invalid boundary orientation: {self.orientation}")
        if self.disposition not in {"OPEN", "SEWN", "FINISHED", "FOLD", "CUT_INTERNAL"}:
            raise ValueError(f"invalid boundary disposition: {self.disposition}")
        if len(self.segment_ids) != len(set(self.segment_ids)):
            raise ValueError(f"duplicate segment in boundary {self.boundary_id}")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "boundary_id": self.boundary_id,
            "semantic_role": self.semantic_role,
            "orientation": self.orientation,
            "disposition": self.disposition,
            "segment_ids": list(self.segment_ids),
        }


@dataclass(frozen=True)
class NotchPlacement:
    notch_id: str
    semantic_role: str
    boundary_id: str
    arc_length_m: float
    normalized_arc: float
    point: Point2

    def validate(self) -> None:
        if not self.notch_id or not self.semantic_role or not self.boundary_id:
            raise ValueError("notch identity, role, and boundary are required")
        if not math.isfinite(self.arc_length_m) or self.arc_length_m < 0.0:
            raise ValueError(f"invalid notch arc length: {self.notch_id}")
        if not 0.0 <= self.normalized_arc <= 1.0:
            raise ValueError(f"invalid normalized notch position: {self.notch_id}")
        self.point.validate()

    def to_dict(self) -> dict:
        self.validate()
        return {
            "notch_id": self.notch_id,
            "semantic_role": self.semantic_role,
            "boundary_id": self.boundary_id,
            "arc_length_m": self.arc_length_m,
            "normalized_arc": self.normalized_arc,
            "point_m": self.point.to_list(),
        }


@dataclass(frozen=True)
class ComponentGeometryAuthority:
    instance_id: str
    component_id: str
    component_revision: int
    parameter_set_sha256: str
    mirror_source_instance_id: str | None
    segments: tuple[CurveSegment, ...]
    boundaries: tuple[BoundaryGeometry, ...]
    notches: tuple[NotchPlacement, ...]
    landmarks: dict[str, Point2]
    internal_segment_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.instance_id or not self.component_id or self.component_revision < 1:
            raise ValueError("component geometry identity and revision are required")
        if len(self.parameter_set_sha256) != 64:
            raise ValueError("component geometry requires resolved parameter identity")
        segment_ids = [item.segment_id for item in self.segments]
        boundary_ids = [item.boundary_id for item in self.boundaries]
        if len(segment_ids) != len(set(segment_ids)) or len(boundary_ids) != len(set(boundary_ids)):
            raise ValueError(f"duplicate segment or boundary in {self.instance_id}")
        for segment in self.segments:
            segment.validate()
        known_segments = set(segment_ids)
        for boundary in self.boundaries:
            boundary.validate()
            if not set(boundary.segment_ids).issubset(known_segments):
                raise ValueError(f"unknown segment in {self.instance_id}.{boundary.boundary_id}")
        if not set(self.internal_segment_ids).issubset(known_segments):
            raise ValueError(f"unknown internal segment in {self.instance_id}")
        known_boundaries = set(boundary_ids)
        for notch in self.notches:
            notch.validate()
            if notch.boundary_id not in known_boundaries:
                raise ValueError(f"notch on unknown boundary: {notch.notch_id}")
        for landmark_id, point in self.landmarks.items():
            if not landmark_id:
                raise ValueError("empty landmark identity")
            point.validate()

    def segment_map(self) -> dict[str, CurveSegment]:
        return {item.segment_id: item for item in self.segments}

    def boundary(self, boundary_id: str) -> BoundaryGeometry:
        for boundary in self.boundaries:
            if boundary.boundary_id == boundary_id:
                return boundary
        raise KeyError(f"unknown geometry boundary: {self.instance_id}.{boundary_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "ComponentGeometryAuthority/1",
            "instance_id": self.instance_id,
            "component_id": self.component_id,
            "component_revision": self.component_revision,
            "parameter_set_sha256": self.parameter_set_sha256,
            "mirror_source_instance_id": self.mirror_source_instance_id,
            "segments": [item.to_dict() for item in self.segments],
            "boundaries": [item.to_dict() for item in self.boundaries],
            "notches": [item.to_dict() for item in self.notches],
            "landmarks": {key: value.to_list() for key, value in sorted(self.landmarks.items())},
            "internal_segment_ids": list(self.internal_segment_ids),
            "triangulation_executed": False,
            "simulation_executed": False,
            "three_dimensional_primitive_authority": False,
        }
        payload["geometry_sha256"] = canonical_sha256(payload)
        return payload
