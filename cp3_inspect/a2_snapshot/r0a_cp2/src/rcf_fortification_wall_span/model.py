from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

SPAN_SCHEMA = "royal-capital.fortification.straight-wall-span/1"
PLAN_SCHEMA = "royal-capital.fortification.wall-span-plan/1"
PARTS_SCHEMA = "royal-capital.fortification.semantic-parts/1"
SOCKETS_SCHEMA = "royal-capital.fortification.socket-plan/1"
TESSELLATION_SCHEMA = "royal-capital.fortification.fixed-tessellation/1"
COMBINED_MESH_SCHEMA = "royal-capital.fortification.wall-span-indexed-mesh/1"
RECEIPT_SCHEMA = "royal-capital.fortification.wall-span-provider-receipt/1"
RESULT_SCHEMA = "royal-capital.fortification.wall-span-result/1"
PART_ORDER = ("foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet")
SOCKET_ORDER = (
    "span_start", "span_end", "wall_walk_start", "wall_walk_end",
    "foundation_start", "foundation_end", "tower_start", "tower_end",
    "utility_inside_01", "utility_inside_02",
)

@dataclass(frozen=True)
class PartSpec:
    part_id: str
    semantic_role: str
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    material_slot: str

    @property
    def volume_factor(self) -> float:
        return (self.y_max - self.y_min) * (self.z_max - self.z_min)

PART_SPECS = (
    PartSpec("foundation", "FOUNDATION", -2.0, 0.0, -4.0, 4.0, "stone_foundation"),
    PartSpec("wall_body", "WALL_BODY", 0.0, 12.0, -3.0, 3.0, "stone_wall"),
    PartSpec("wall_walk", "WALL_WALK", 12.0, 12.6, -3.2, 3.2, "stone_walk"),
    PartSpec("inner_parapet", "INNER_PARAPET", 12.6, 14.4, 2.4, 3.2, "stone_parapet"),
    PartSpec("outer_parapet", "OUTER_PARAPET", 12.6, 14.4, -3.2, -2.4, "stone_parapet"),
)


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "span_id", "length_m", "frame", "inside_side", "outside_side", "tolerances", "budget", "runtime"}
    if set(value) != required:
        raise ValueError(f"fixture keys mismatch missing={sorted(required-set(value))} extra={sorted(set(value)-required)}")
    if value["schema"] != SPAN_SCHEMA:
        raise ValueError("unsupported wall-span schema")
    if not isinstance(value["span_id"], str) or not value["span_id"].strip():
        raise ValueError("span_id required")
    length = float(value["length_m"])
    if not math.isfinite(length) or length <= 0 or length > 200:
        raise ValueError("length_m outside (0,200]")
    if value["frame"] != "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD":
        raise ValueError("project frame mismatch")
    if value["inside_side"] != "POSITIVE_Z" or value["outside_side"] != "NEGATIVE_Z":
        raise ValueError("pilot inside/outside convention mismatch")
    t = value["tolerances"]
    if set(t) != {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad", "mesh_round_digits"}:
        raise ValueError("tolerance keys mismatch")
    if float(t["tessellation_linear_m"]) != 0.05 or float(t["tessellation_angular_rad"]) != 0.1 or int(t["mesh_round_digits"]) != 9:
        raise ValueError("CP2 fixed tessellation contract mismatch")
    return dict(value)


def provider_request(fixture: Mapping[str, Any], part: PartSpec) -> dict[str, Any]:
    length = float(fixture["length_m"])
    loop = [[part.y_min, part.z_min], [part.y_max, part.z_min], [part.y_max, part.z_max], [part.y_min, part.z_max]]
    return {
        "schema": "royal-capital.fortification.cad-provider-request/1",
        "request_id": f"cad-request/{fixture['span_id']}/{part.part_id}@1",
        "units": "METER",
        "frame": fixture["frame"],
        "runtime": dict(fixture["runtime"]),
        "tolerances": {k: fixture["tolerances"][k] for k in ("linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad")},
        "budget": dict(fixture["budget"]),
        "operation": {
            "kind": "PROFILE_EXTRUSION",
            "plane": {"origin_m": [0.0, 0.0, 0.0], "x_axis": [0.0, 1.0, 0.0], "y_axis": [0.0, 0.0, 1.0]},
            "outer_loop_m": loop,
            "distance_m": length,
        },
        "outputs": {"indexed_mesh": True, "stored_copies": ["STEP", "BREP"]},
    }


def expected_bounds(length: float, part: PartSpec) -> dict[str, list[float]]:
    return {"min": [0.0, part.y_min, part.z_min], "max": [length, part.y_max, part.z_max]}


def socket_plan(length: float) -> list[dict[str, Any]]:
    frame = {"x_axis": [1.0, 0.0, 0.0], "y_axis": [0.0, 1.0, 0.0], "z_axis": [0.0, 0.0, 1.0]}
    rows = [
        ("span_start", "SPAN_JOIN", [0.0, 6.0, 0.0]), ("span_end", "SPAN_JOIN", [length, 6.0, 0.0]),
        ("wall_walk_start", "WALL_WALK_CONTINUATION", [0.0, 12.3, 0.0]), ("wall_walk_end", "WALL_WALK_CONTINUATION", [length, 12.3, 0.0]),
        ("foundation_start", "FOUNDATION_INTERFACE", [0.0, -1.0, 0.0]), ("foundation_end", "FOUNDATION_INTERFACE", [length, -1.0, 0.0]),
        ("tower_start", "TOWER_JOIN", [0.0, 6.0, 0.0]), ("tower_end", "TOWER_JOIN", [length, 6.0, 0.0]),
        ("utility_inside_01", "UTILITY_INSIDE", [length / 3.0, 6.0, 2.75]), ("utility_inside_02", "UTILITY_INSIDE", [2.0 * length / 3.0, 6.0, 2.75]),
    ]
    return [{"socket_id": sid, "role": role, "position_m": [round(float(v), 9) for v in pos], "frame": frame, "required": role != "UTILITY_INSIDE"} for sid, role, pos in rows]
