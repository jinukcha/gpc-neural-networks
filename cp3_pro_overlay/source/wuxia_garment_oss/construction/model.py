"""Construction contracts owned above meshing and below PatternDocument."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..pattern_cad.document.model import PatternDocument, canonical_sha256


SEAM_TYPES = {
    "PLAIN_SEAM",
    "GATHERED_SEAM",
    "GUSSET_INSERTION",
    "WAIST_JOIN",
}
EDGE_FINISH_TYPES = {"BIAS_BINDING", "BOUND_FACING", "DOUBLE_TURN_HEM"}
LAYER_TYPES = {"LINING", "INTERFACING"}


@dataclass(frozen=True)
class BoundaryInterval:
    panel_id: str
    curve_id: str
    start_fraction: float = 0.0
    end_fraction: float = 1.0

    def validate(self, document: PatternDocument) -> None:
        if self.curve_id not in document.curves:
            raise ValueError(f"unknown construction curve: {self.curve_id}")
        curve = document.curves[self.curve_id]
        if curve.panel_id != self.panel_id:
            raise ValueError(f"construction curve owner mismatch: {self.curve_id}")
        if not 0.0 <= self.start_fraction < self.end_fraction <= 1.0:
            raise ValueError(f"invalid boundary interval: {self.curve_id}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NotchMatch:
    pair_id: str
    seam_id: str
    role: str
    side_a_notch_id: str
    side_b_notch_id: str
    side_a_fraction: float
    side_b_fraction: float

    def validate(self) -> None:
        if not self.pair_id or not self.seam_id or not self.role:
            raise ValueError("notch match identity is required")
        if not 0.0 <= self.side_a_fraction <= 1.0:
            raise ValueError(f"invalid notch fraction: {self.pair_id}: A")
        if not 0.0 <= self.side_b_fraction <= 1.0:
            raise ValueError(f"invalid notch fraction: {self.pair_id}: B")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SeamSpecV2:
    seam_id: str
    seam_type: str
    side_a: BoundaryInterval
    side_b: BoundaryInterval
    allowance_a_m: float
    allowance_b_m: float
    stitch_class: str
    ease_ratio: float = 1.0
    gather_ratio: float = 1.0
    notch_pair_ids: tuple[str, ...] = ()
    fold_direction: str = "NONE"
    topstitch_offset_m: float = 0.0
    turn_of_cloth_m: float = 0.0

    def validate(self, document: PatternDocument) -> None:
        if not self.seam_id or self.seam_type not in SEAM_TYPES:
            raise ValueError(f"invalid seam identity/type: {self.seam_id}")
        self.side_a.validate(document)
        self.side_b.validate(document)
        if not 0.0 <= self.allowance_a_m <= 0.05:
            raise ValueError(f"invalid seam allowance A: {self.seam_id}")
        if not 0.0 <= self.allowance_b_m <= 0.05:
            raise ValueError(f"invalid seam allowance B: {self.seam_id}")
        if not 0.5 <= self.ease_ratio <= 2.5:
            raise ValueError(f"invalid ease ratio: {self.seam_id}")
        if not 1.0 <= self.gather_ratio <= 3.0:
            raise ValueError(f"invalid gather ratio: {self.seam_id}")
        if self.seam_type == "GATHERED_SEAM" and self.gather_ratio <= 1.0:
            raise ValueError(f"gathered seam requires ratio > 1: {self.seam_id}")
        if self.topstitch_offset_m < 0.0 or self.turn_of_cloth_m < 0.0:
            raise ValueError(f"negative construction offset: {self.seam_id}")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["notch_pair_ids"] = list(self.notch_pair_ids)
        return payload


@dataclass(frozen=True)
class EdgeFinishSpec:
    finish_id: str
    boundary: BoundaryInterval
    finish_type: str
    allowance_m: float
    facing_id: str | None = None
    topstitch_offset_m: float = 0.0

    def validate(self, document: PatternDocument) -> None:
        if not self.finish_id or self.finish_type not in EDGE_FINISH_TYPES:
            raise ValueError(f"invalid edge finish: {self.finish_id}")
        self.boundary.validate(document)
        curve = document.curves[self.boundary.curve_id]
        if curve.disposition != "OPEN":
            raise ValueError(f"edge finish requires OPEN boundary: {self.finish_id}")
        if not 0.0 <= self.allowance_m <= 0.06:
            raise ValueError(f"invalid edge allowance: {self.finish_id}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClosureSpec:
    closure_id: str
    closure_type: str
    owner_panel_id: str
    start_point_id: str
    length_m: float
    direction_xy: tuple[float, float]
    component_ids: tuple[str, ...]
    facing_id: str | None = None

    def validate(self, document: PatternDocument) -> None:
        if self.owner_panel_id not in document.panel_ids:
            raise ValueError(f"unknown closure panel: {self.closure_id}")
        if self.start_point_id not in document.points:
            raise ValueError(f"unknown closure start point: {self.closure_id}")
        if not 0.0 < self.length_m <= 0.5:
            raise ValueError(f"invalid closure length: {self.closure_id}")
        if len(self.direction_xy) != 2 or sum(v * v for v in self.direction_xy) <= 0.0:
            raise ValueError(f"invalid closure direction: {self.closure_id}")
        if not self.component_ids:
            raise ValueError(f"closure components required: {self.closure_id}")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["direction_xy"] = list(self.direction_xy)
        payload["component_ids"] = list(self.component_ids)
        return payload


@dataclass(frozen=True)
class FacingSpec:
    facing_id: str
    owner_panel_id: str
    source_curve_ids: tuple[str, ...]
    depth_m: float
    allowance_m: float
    turn_of_cloth_m: float

    def validate(self, document: PatternDocument) -> None:
        if self.owner_panel_id not in document.panel_ids:
            raise ValueError(f"unknown facing panel: {self.facing_id}")
        for curve_id in self.source_curve_ids:
            if curve_id not in document.curves:
                raise ValueError(f"unknown facing curve: {self.facing_id}: {curve_id}")
        if not 0.005 <= self.depth_m <= 0.2:
            raise ValueError(f"invalid facing depth: {self.facing_id}")
        if not 0.0 <= self.allowance_m <= 0.03:
            raise ValueError(f"invalid facing allowance: {self.facing_id}")
        if not 0.0 <= self.turn_of_cloth_m <= 0.01:
            raise ValueError(f"invalid turn-of-cloth: {self.facing_id}")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["source_curve_ids"] = list(self.source_curve_ids)
        return payload


@dataclass(frozen=True)
class LayerPieceSpec:
    piece_id: str
    layer_type: str
    source_owner_id: str
    ease_m: float = 0.0
    hem_reduction_m: float = 0.0
    bonded: bool = False

    def validate(self, document: PatternDocument) -> None:
        if self.layer_type not in LAYER_TYPES:
            raise ValueError(f"unsupported layer type: {self.piece_id}")
        valid_owner = self.source_owner_id in document.panel_ids or self.source_owner_id in document.curves
        if not valid_owner:
            raise ValueError(f"unknown layer owner: {self.piece_id}")
        if self.ease_m < 0.0 or self.hem_reduction_m < 0.0:
            raise ValueError(f"negative layer adjustment: {self.piece_id}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TurnOfClothSpec:
    turn_id: str
    owner_id: str
    allowance_m: float
    direction: str

    def validate(self) -> None:
        if not self.turn_id or not self.owner_id:
            raise ValueError("turn-of-cloth identity is required")
        if not 0.0 <= self.allowance_m <= 0.01:
            raise ValueError(f"invalid turn-of-cloth allowance: {self.turn_id}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AssemblyOperation:
    operation_id: str
    operation_type: str
    owner_ids: tuple[str, ...]
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["owner_ids"] = list(self.owner_ids)
        payload["depends_on"] = list(self.depends_on)
        return payload


@dataclass(frozen=True)
class ConstructionGraph:
    graph_id: str
    operations: tuple[AssemblyOperation, ...]

    def validate(self) -> None:
        if not self.graph_id or not self.operations:
            raise ValueError("construction graph identity and operations are required")
        ids = [item.operation_id for item in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate assembly operation IDs")
        known = set(ids)
        for operation in self.operations:
            missing = sorted(set(operation.depends_on) - known)
            if missing:
                raise ValueError(f"unknown assembly dependency: {operation.operation_id}: {missing}")
            if operation.operation_id in operation.depends_on:
                raise ValueError(f"self-dependent operation: {operation.operation_id}")

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "ConstructionGraph/1",
            "graph_id": self.graph_id,
            "operations": [item.to_dict() for item in self.operations],
        }
        payload["graph_sha256"] = canonical_sha256(payload)
        return payload
