from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Any, Mapping, Sequence

FIXTURE_SCHEMA = "royal-capital.fortification.tower-family-fixture/1"
PLAN_SCHEMA = "royal-capital.fortification.tower-plan/1"
PARTS_SCHEMA = "royal-capital.fortification.tower-semantic-parts/1"
SOCKETS_SCHEMA = "royal-capital.fortification.tower-sockets/1"
BOUNDS_SCHEMA = "royal-capital.fortification.tower-bounds-attachment/1"
FOUNDATION_SCHEMA = "royal-capital.fortification.tower-foundation-interface/1"
TESSELLATION_SCHEMA = "royal-capital.fortification.tower-fixed-tessellation/1"
MESH_SCHEMA = "royal-capital.fortification.tower-indexed-mesh/1"
RECEIPT_SCHEMA = "royal-capital.fortification.tower-provider-receipt/1"
RESULT_SCHEMA = "royal-capital.fortification.tower-result/1"
FAILURE_SCHEMA = "royal-capital.fortification.tower-failure/1"

PROJECT_FRAME = "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"
EXPECTED_RUNTIME = {
    "build123d_version": "0.13.1.dev12+ge22d34dae",
    "ocp_version": "8.0.1.0.0",
    "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466",
    "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f",
}
PART_ORDER = ("foundation", "tower_body", "tower_crown")
FAMILY_ORDER = ("ROUND", "SQUARE", "POLYGONAL")
SOCKET_ORDER = (
    "span_in", "span_out", "wall_walk_in", "wall_walk_out",
    "foundation_in", "foundation_out", "foundation_center",
    "foundation_outside", "foundation_inside", "roof_socket",
)


class TowerFailureCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    FAMILY_UNSUPPORTED = "FAMILY_UNSUPPORTED"
    DIMENSION_OUT_OF_DOMAIN = "DIMENSION_OUT_OF_DOMAIN"
    SIDE_COUNT_OUT_OF_DOMAIN = "SIDE_COUNT_OUT_OF_DOMAIN"
    APPROXIMATION_ERROR_EXCEEDED = "APPROXIMATION_ERROR_EXCEEDED"
    ATTACHMENT_INTERSECTION_MISSING = "ATTACHMENT_INTERSECTION_MISSING"
    BODY_PROJECTION_OUT_OF_DOMAIN = "BODY_PROJECTION_OUT_OF_DOMAIN"
    RUNTIME_MISMATCH = "RUNTIME_MISMATCH"
    GEOMETRY_BUDGET_EXCEEDED = "GEOMETRY_BUDGET_EXCEEDED"
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    GEOMETRY_EVIDENCE_MISMATCH = "GEOMETRY_EVIDENCE_MISMATCH"
    PUBLISH_ABORTED = "PUBLISH_ABORTED"


class TowerError(ValueError):
    def __init__(self, code: TowerFailureCode | str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = message


@dataclass(frozen=True)
class PartSpec:
    part_id: str
    semantic_role: str
    y_min: float
    y_max: float
    expansion_m: float
    material_slot: str


@dataclass(frozen=True)
class TowerSpec:
    raw: dict[str, Any]
    tower_id: str
    family: str
    center_x: float
    center_z: float
    outer_width_m: float
    height_m: float
    foundation_depth_m: float
    side_count: int
    wall_centerline_z_m: float
    wall_outside_face_z_m: float
    wall_walk_y_m: float
    max_body_projection_m: float
    loops: dict[str, tuple[tuple[float, float], ...]]
    bounds: dict[str, dict[str, list[float]]]
    areas: dict[str, float]
    attachment_x: tuple[float, float]
    body_projection_m: float
    foundation_projection_m: float
    approximation_error_m: float
    route: str


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TowerError(TowerFailureCode.INVALID_REQUEST, f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TowerError(TowerFailureCode.INVALID_REQUEST, f"{field} must be finite")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TowerError(TowerFailureCode.INVALID_REQUEST, f"{field} must be an integer")
    return value


def _strict(value: Mapping[str, Any], required: set[str]) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise TowerError(TowerFailureCode.INVALID_REQUEST, f"key mismatch missing={sorted(missing)} extra={sorted(extra)}")


def _polygon_area(loop: Sequence[tuple[float, float]]) -> float:
    return abs(0.5 * sum(loop[i][0] * loop[(i + 1) % len(loop)][1] - loop[(i + 1) % len(loop)][0] * loop[i][1] for i in range(len(loop))))


def _regular_loop(center_x: float, center_z: float, radius: float, sides: int) -> tuple[tuple[float, float], ...]:
    # The local profile plane axes are project +X and project -Z. Build the loop
    # counter-clockwise in that local plane, then return project (X,Z) points.
    center_v = -center_z
    local = [
        (center_x + radius * math.cos(2.0 * math.pi * i / sides), center_v + radius * math.sin(2.0 * math.pi * i / sides))
        for i in range(sides)
    ]
    return tuple((round(u, 12), round(-v, 12)) for u, v in local)


def _square_loop(center_x: float, center_z: float, half: float) -> tuple[tuple[float, float], ...]:
    cv = -center_z
    local = ((center_x - half, cv - half), (center_x + half, cv - half), (center_x + half, cv + half), (center_x - half, cv + half))
    return tuple((round(u, 12), round(-v, 12)) for u, v in local)


def footprint_loop(family: str, center_x: float, center_z: float, outer_width_m: float, side_count: int, expansion_m: float) -> tuple[tuple[float, float], ...]:
    half = outer_width_m / 2.0 + expansion_m
    if family == "SQUARE":
        return _square_loop(center_x, center_z, half)
    return _regular_loop(center_x, center_z, half, side_count)


def _bounds(loop: Sequence[tuple[float, float]], y_min: float, y_max: float) -> dict[str, list[float]]:
    return {
        "min": [round(min(p[0] for p in loop), 9), round(y_min, 9), round(min(p[1] for p in loop), 9)],
        "max": [round(max(p[0] for p in loop), 9), round(y_max, 9), round(max(p[1] for p in loop), 9)],
    }


def _line_intersections(loop: Sequence[tuple[float, float]], z_value: float, eps: float) -> tuple[float, float]:
    values: list[float] = []
    for index, a in enumerate(loop):
        b = loop[(index + 1) % len(loop)]
        dz = b[1] - a[1]
        if abs(dz) <= eps:
            continue
        t = (z_value - a[1]) / dz
        if -eps <= t <= 1.0 + eps:
            x = a[0] + t * (b[0] - a[0])
            if not any(abs(x - old) <= eps for old in values):
                values.append(x)
    values.sort()
    if len(values) != 2:
        raise TowerError(TowerFailureCode.ATTACHMENT_INTERSECTION_MISSING, f"wall centerline z={z_value} intersects footprint at {values}")
    return (round(values[0], 9), round(values[1], 9))


def _part_specs(height_m: float, foundation_depth_m: float) -> tuple[PartSpec, ...]:
    return (
        PartSpec("foundation", "FOUNDATION", -foundation_depth_m, 0.0, 1.0, "stone_foundation"),
        PartSpec("tower_body", "TOWER_BODY", 0.0, height_m, 0.0, "stone_tower"),
        PartSpec("tower_crown", "TOWER_CROWN_PLATFORM", height_m, height_m + 0.6, 0.5, "stone_crown"),
    )


def validate_fixture(value: Mapping[str, Any]) -> TowerSpec:
    if not isinstance(value, Mapping):
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "fixture must be an object")
    required = {"schema", "tower_id", "family", "center_m", "outer_width_m", "height_m", "foundation_depth_m", "side_count", "wall_interface", "tolerances", "budget", "runtime"}
    _strict(value, required)
    if value["schema"] != FIXTURE_SCHEMA:
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "unsupported fixture schema")
    tower_id = value["tower_id"]
    if not isinstance(tower_id, str) or not tower_id.strip():
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "tower_id is required")
    family = str(value["family"])
    if family not in FAMILY_ORDER:
        raise TowerError(TowerFailureCode.FAMILY_UNSUPPORTED, f"unsupported family {family}")
    center = value["center_m"]
    if not isinstance(center, Sequence) or isinstance(center, (str, bytes)) or len(center) != 3:
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "center_m must have three values")
    center_x, center_y, center_z = (_number(v, f"center_m[{i}]") for i, v in enumerate(center))
    if abs(center_y) > 1e-9:
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "CP1 tower base elevation must be zero")
    width = _number(value["outer_width_m"], "outer_width_m")
    height = _number(value["height_m"], "height_m")
    depth = _number(value["foundation_depth_m"], "foundation_depth_m")
    sides = _integer(value["side_count"], "side_count")
    if not 8.0 <= width <= 28.0 or not 8.0 <= height <= 18.0 or not 1.0 <= depth <= 8.0:
        raise TowerError(TowerFailureCode.DIMENSION_OUT_OF_DOMAIN, f"dimensions outside domain width={width} height={height} foundation={depth}")

    wall = value["wall_interface"]
    if not isinstance(wall, Mapping):
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "wall_interface must be an object")
    _strict(wall, {"centerline_z_m", "outside_face_z_m", "wall_walk_y_m", "max_body_projection_m"})
    wall_z = _number(wall["centerline_z_m"], "wall_interface.centerline_z_m")
    outside_z = _number(wall["outside_face_z_m"], "wall_interface.outside_face_z_m")
    walk_y = _number(wall["wall_walk_y_m"], "wall_interface.wall_walk_y_m")
    max_projection = _number(wall["max_body_projection_m"], "wall_interface.max_body_projection_m")
    if max_projection > 10.0 or max_projection < 1.0:
        raise TowerError(TowerFailureCode.DIMENSION_OUT_OF_DOMAIN, "max body projection must be in [1,10]")

    tolerances = value["tolerances"]
    if not isinstance(tolerances, Mapping):
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "tolerances must be an object")
    _strict(tolerances, {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad", "mesh_round_digits"})
    if _number(tolerances["tessellation_linear_m"], "tessellation_linear_m") != 0.05 or _number(tolerances["tessellation_angular_rad"], "tessellation_angular_rad") != 0.1 or _integer(tolerances["mesh_round_digits"], "mesh_round_digits") != 9:
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "fixed tessellation contract mismatch")

    budget = value["budget"]
    if not isinstance(budget, Mapping):
        raise TowerError(TowerFailureCode.INVALID_REQUEST, "budget must be an object")
    _strict(budget, {"max_profile_points", "max_vertices", "max_triangles", "max_artifact_bytes", "max_parts"})
    for key in budget:
        _integer(budget[key], f"budget.{key}")
    if budget["max_parts"] < len(PART_ORDER):
        raise TowerError(TowerFailureCode.GEOMETRY_BUDGET_EXCEEDED, "part budget is below required tower parts")

    runtime = value["runtime"]
    if not isinstance(runtime, Mapping) or dict(runtime) != EXPECTED_RUNTIME:
        raise TowerError(TowerFailureCode.RUNTIME_MISMATCH, f"runtime mismatch {runtime}")

    if family == "ROUND":
        if sides != 32:
            raise TowerError(TowerFailureCode.SIDE_COUNT_OUT_OF_DOMAIN, "ROUND CP1 requires 32 sides")
        approximation_error = (width / 2.0) * (1.0 - math.cos(math.pi / sides))
        if approximation_error > float(tolerances["tessellation_linear_m"]):
            raise TowerError(TowerFailureCode.APPROXIMATION_ERROR_EXCEEDED, f"round approximation error {approximation_error}")
        route = "POLYGONAL_APPROXIMATION_32"
    elif family == "SQUARE":
        if sides != 4:
            raise TowerError(TowerFailureCode.SIDE_COUNT_OUT_OF_DOMAIN, "SQUARE requires four sides")
        approximation_error = 0.0
        route = "PROFILE_EXTRUSION_SQUARE"
    else:
        if not 6 <= sides <= 12:
            raise TowerError(TowerFailureCode.SIDE_COUNT_OUT_OF_DOMAIN, "POLYGONAL side_count must be in [6,12]")
        approximation_error = 0.0
        route = f"PROFILE_EXTRUSION_REGULAR_{sides}"

    loops: dict[str, tuple[tuple[float, float], ...]] = {}
    bounds: dict[str, dict[str, list[float]]] = {}
    areas: dict[str, float] = {}
    for part in _part_specs(height, depth):
        loop = footprint_loop(family, center_x, center_z, width, sides, part.expansion_m)
        loops[part.part_id] = loop
        bounds[part.part_id] = _bounds(loop, part.y_min, part.y_max)
        areas[part.part_id] = _polygon_area(loop)
        if len(loop) > budget["max_profile_points"]:
            raise TowerError(TowerFailureCode.GEOMETRY_BUDGET_EXCEEDED, f"{part.part_id} profile has {len(loop)} points")

    attachment = _line_intersections(loops["tower_body"], wall_z, max(1e-9, float(tolerances["linear_m"]) * 2.0))
    body_min_z = bounds["tower_body"]["min"][2]
    foundation_min_z = bounds["foundation"]["min"][2]
    body_projection = outside_z - body_min_z
    foundation_projection = outside_z - foundation_min_z
    if not 1.0 <= body_projection <= max_projection:
        raise TowerError(TowerFailureCode.BODY_PROJECTION_OUT_OF_DOMAIN, f"body projection {body_projection} outside [1,{max_projection}]")

    raw = dict(value)
    raw["center_m"] = [center_x, center_y, center_z]
    return TowerSpec(
        raw, tower_id, family, center_x, center_z, width, height, depth, sides,
        wall_z, outside_z, walk_y, max_projection,
        loops, bounds, areas, attachment, round(body_projection, 9), round(foundation_projection, 9),
        round(approximation_error, 12), route,
    )


def part_specs(spec: TowerSpec) -> tuple[PartSpec, ...]:
    return _part_specs(spec.height_m, spec.foundation_depth_m)


def provider_request(spec: TowerSpec, part: PartSpec) -> dict[str, Any]:
    local_loop = [[round(x, 12), round(-z, 12)] for x, z in spec.loops[part.part_id]]
    return {
        "schema": "royal-capital.fortification.cad-provider-request/1",
        "request_id": f"cad-request/{spec.tower_id}/{part.part_id}@1",
        "units": "METER",
        "frame": PROJECT_FRAME,
        "runtime": dict(EXPECTED_RUNTIME),
        "tolerances": {key: spec.raw["tolerances"][key] for key in ("linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad")},
        "budget": {key: spec.raw["budget"][key] for key in ("max_profile_points", "max_vertices", "max_triangles", "max_artifact_bytes")},
        "operation": {
            "kind": "PROFILE_EXTRUSION",
            "plane": {"origin_m": [0.0, part.y_min, 0.0], "x_axis": [1.0, 0.0, 0.0], "y_axis": [0.0, 0.0, -1.0]},
            "outer_loop_m": local_loop,
            "distance_m": part.y_max - part.y_min,
        },
        "outputs": {"indexed_mesh": True, "stored_copies": ["STEP", "BREP"]},
    }


def socket_plan(spec: TowerSpec) -> list[dict[str, Any]]:
    left, right = spec.attachment_x
    frame = {"x_axis": [1.0, 0.0, 0.0], "y_axis": [0.0, 1.0, 0.0], "z_axis": [0.0, 0.0, 1.0], "determinant": 1.0}
    foundation = spec.bounds["foundation"]
    crown_y = spec.height_m + 0.6
    rows = (
        ("span_in", "SPAN_ATTACHMENT", [left, 6.0, spec.wall_centerline_z_m], True),
        ("span_out", "SPAN_ATTACHMENT", [right, 6.0, spec.wall_centerline_z_m], True),
        ("wall_walk_in", "WALL_WALK_ATTACHMENT", [left, spec.wall_walk_y_m, spec.wall_centerline_z_m], True),
        ("wall_walk_out", "WALL_WALK_ATTACHMENT", [right, spec.wall_walk_y_m, spec.wall_centerline_z_m], True),
        ("foundation_in", "FOUNDATION_ATTACHMENT", [left, -spec.foundation_depth_m / 2.0, spec.wall_centerline_z_m], True),
        ("foundation_out", "FOUNDATION_ATTACHMENT", [right, -spec.foundation_depth_m / 2.0, spec.wall_centerline_z_m], True),
        ("foundation_center", "FOUNDATION_BEARING", [spec.center_x, -spec.foundation_depth_m, spec.center_z], True),
        ("foundation_outside", "FOUNDATION_OUTSIDE", [spec.center_x, -spec.foundation_depth_m, foundation["min"][2]], True),
        ("foundation_inside", "FOUNDATION_INSIDE", [spec.center_x, -spec.foundation_depth_m, foundation["max"][2]], True),
        ("roof_socket", "ROOF_ATTACHMENT", [spec.center_x, crown_y, spec.center_z], False),
    )
    return [{"socket_id": sid, "role": role, "position_m": [round(float(x), 9) for x in position], "frame": frame, "required": required} for sid, role, position, required in rows]


def fixture_examples() -> list[dict[str, Any]]:
    common = {
        "schema": FIXTURE_SCHEMA,
        "center_m": [0.0, 0.0, -4.0],
        "height_m": 16.0,
        "foundation_depth_m": 3.0,
        "wall_interface": {"centerline_z_m": 0.0, "outside_face_z_m": -3.0, "wall_walk_y_m": 12.3, "max_body_projection_m": 10.0},
        "tolerances": {"linear_m": 0.000001, "angular_rad": 0.000001, "tessellation_linear_m": 0.05, "tessellation_angular_rad": 0.1, "mesh_round_digits": 9},
        "budget": {"max_profile_points": 64, "max_vertices": 500000, "max_triangles": 500000, "max_artifact_bytes": 100000000, "max_parts": 8},
        "runtime": dict(EXPECTED_RUNTIME),
    }
    return [
        {**common, "tower_id": "fortification/caelmere/pilot/tower-round-001", "family": "ROUND", "outer_width_m": 17.0, "side_count": 32},
        {**common, "tower_id": "fortification/caelmere/pilot/tower-square-001", "family": "SQUARE", "outer_width_m": 16.0, "side_count": 4},
        {**common, "tower_id": "fortification/caelmere/pilot/tower-polygonal-001", "family": "POLYGONAL", "outer_width_m": 18.0, "side_count": 8},
    ]
