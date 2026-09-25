from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

FIXTURE_SCHEMA = "royal-capital.fortification.join-fixture/1"
PLAN_SCHEMA = "royal-capital.fortification.join-plan/1"
FRAME_SCHEMA = "royal-capital.fortification.join-local-frames/1"
ALIGNMENT_SCHEMA = "royal-capital.fortification.join-socket-alignment/1"
OVERLAP_SCHEMA = "royal-capital.fortification.join-bounded-overlap/1"
MESH_SCHEMA = "royal-capital.fortification.join-indexed-mesh/1"
PARTS_SCHEMA = "royal-capital.fortification.join-semantic-parts/1"
STORED_SCHEMA = "royal-capital.fortification.join-stored-copies/1"
RECEIPT_SCHEMA = "royal-capital.fortification.join-provider-receipt/1"
RESULT_SCHEMA = "royal-capital.fortification.join-result/1"
PROJECT_FRAME = "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"
UP = (0.0, 1.0, 0.0)
PART_ORDER = ("foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet")

EXPECTED_RUNTIME = {
    "build123d_version": "0.13.1.dev12+ge22d34dae",
    "ocp_version": "8.0.1.0.0",
    "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466",
    "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f",
}


class JoinFamily(StrEnum):
    MITER = "MITER"
    BEVEL = "BEVEL"
    PROFILE_TRANSITION = "PROFILE_TRANSITION"


class JoinFailure(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_FRAME = "INVALID_FRAME"
    SOCKET_GEOMETRY_MISMATCH = "SOCKET_GEOMETRY_MISMATCH"
    TURN_OUT_OF_DOMAIN = "TURN_OUT_OF_DOMAIN"
    PROFILE_TRANSITION_REQUIRED = "PROFILE_TRANSITION_REQUIRED"
    OVERLAP_BUDGET_EXCEEDED = "OVERLAP_BUDGET_EXCEEDED"
    GEOMETRY_BUDGET_EXCEEDED = "GEOMETRY_BUDGET_EXCEEDED"
    RUNTIME_MISMATCH = "RUNTIME_MISMATCH"
    PROVIDER_EXECUTION_FAILED = "PROVIDER_EXECUTION_FAILED"
    STORED_COPY_FAILED = "STORED_COPY_FAILED"
    PUBLISH_ABORTED = "PUBLISH_ABORTED"


class JoinError(ValueError):
    def __init__(self, code: JoinFailure | str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = message


@dataclass(frozen=True)
class Band:
    part_id: str
    semantic_role: str
    y_min: float
    y_max: float
    offset_min: float
    offset_max: float
    material_slot: str


@dataclass(frozen=True)
class WallProfile:
    profile_id: str
    bands: tuple[Band, ...]


STANDARD = WallProfile(
    "STANDARD_R0B",
    (
        Band("foundation", "FOUNDATION", -2.0, 0.0, -4.0, 4.0, "stone_foundation"),
        Band("wall_body", "WALL_BODY", 0.0, 12.0, -3.0, 3.0, "stone_wall"),
        Band("wall_walk", "WALL_WALK", 12.0, 12.6, -3.2, 3.2, "stone_walk"),
        Band("inner_parapet", "INNER_PARAPET", 12.6, 14.4, 2.4, 3.2, "stone_parapet"),
        Band("outer_parapet", "OUTER_PARAPET", 12.6, 14.4, -3.2, -2.4, "stone_parapet"),
    ),
)
REINFORCED = WallProfile(
    "REINFORCED_R0B",
    (
        Band("foundation", "FOUNDATION", -2.5, 0.0, -5.0, 5.0, "stone_foundation_reinforced"),
        Band("wall_body", "WALL_BODY", 0.0, 13.0, -3.5, 3.5, "stone_wall_reinforced"),
        Band("wall_walk", "WALL_WALK", 13.0, 13.7, -3.7, 3.7, "stone_walk_reinforced"),
        Band("inner_parapet", "INNER_PARAPET", 13.7, 15.7, 2.7, 3.7, "stone_parapet_reinforced"),
        Band("outer_parapet", "OUTER_PARAPET", 13.7, 15.7, -3.7, -2.7, "stone_parapet_reinforced"),
    ),
)
PROFILES = {p.profile_id: p for p in (STANDARD, REINFORCED)}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_ref(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _number(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise JoinError(JoinFailure.INVALID_REQUEST, f"{name} must be numeric")
    out = float(v)
    if not math.isfinite(out):
        raise JoinError(JoinFailure.INVALID_REQUEST, f"{name} must be finite")
    return out


def vec3(v: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(v, Sequence) or isinstance(v, (str, bytes)) or len(v) != 3:
        raise JoinError(JoinFailure.INVALID_REQUEST, f"{name} must be a 3-vector")
    return tuple(_number(x, f"{name}[{i}]") for i, x in enumerate(v))  # type: ignore[return-value]


def add(a, b): return tuple(a[i] + b[i] for i in range(3))
def sub(a, b): return tuple(a[i] - b[i] for i in range(3))
def mul(a, s): return tuple(a[i] * s for i in range(3))
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a): return math.sqrt(dot(a, a))
def normalize(a, name="vector"):
    n = norm(a)
    if n <= 1e-12: raise JoinError(JoinFailure.INVALID_FRAME, f"{name} is zero length")
    return tuple(x / n for x in a)
def distance(a, b): return norm(sub(a, b))
def rounded(v, digits=9): return [round(float(x), digits) for x in v]


def frame_from_tangent(position, tangent) -> dict[str, Any]:
    t = normalize(tangent, "tangent")
    if abs(t[1]) > 1e-9:
        raise JoinError(JoinFailure.INVALID_FRAME, "R0B-CP2 supports horizontal tangent only")
    inside = normalize(cross(t, UP), "inside")
    outside = mul(inside, -1.0)
    determinant = dot(cross(t, UP), inside)
    return {
        "position_m": rounded(position), "tangent": rounded(t), "up": list(UP),
        "inside": rounded(inside), "outside": rounded(outside),
        "orientation_determinant": round(determinant, 12),
    }


def validate_frame(value: Mapping[str, Any], name: str, tol: float) -> dict[str, Any]:
    required = {"position_m", "tangent", "up", "inside", "outside"}
    if set(value) != required:
        raise JoinError(JoinFailure.INVALID_REQUEST, f"{name} keys mismatch")
    p = vec3(value["position_m"], f"{name}.position_m")
    t = normalize(vec3(value["tangent"], f"{name}.tangent"), f"{name}.tangent")
    u = normalize(vec3(value["up"], f"{name}.up"), f"{name}.up")
    i = normalize(vec3(value["inside"], f"{name}.inside"), f"{name}.inside")
    o = normalize(vec3(value["outside"], f"{name}.outside"), f"{name}.outside")
    if distance(u, UP) > tol or abs(t[1]) > tol:
        raise JoinError(JoinFailure.INVALID_FRAME, f"{name} is not horizontal with fixed +Y up")
    expected_i = normalize(cross(t, u), f"{name}.expected_inside")
    if distance(i, expected_i) > tol or distance(o, mul(i, -1.0)) > tol:
        raise JoinError(JoinFailure.INVALID_FRAME, f"{name} inside/outside basis mismatch")
    if max(abs(dot(t,u)), abs(dot(t,i)), abs(dot(u,i)), abs(norm(t)-1), abs(norm(u)-1), abs(norm(i)-1)) > tol:
        raise JoinError(JoinFailure.INVALID_FRAME, f"{name} is not orthonormal")
    det = dot(cross(t, u), i)
    if abs(det - 1.0) > tol:
        raise JoinError(JoinFailure.INVALID_FRAME, f"{name} determinant={det}")
    return frame_from_tangent(p, t)


def signed_turn_deg(t0, t1) -> float:
    a, b = normalize(t0), normalize(t1)
    c = cross(a, b)
    return math.degrees(math.atan2(dot(c, UP), dot(a, b)))


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "join_id", "join_family", "frame", "anchor_m", "incoming_socket", "outgoing_socket", "incoming_profile", "outgoing_profile", "join_depth_m", "bevel_setback_m", "max_overlap_m", "tolerances", "budget", "runtime"}
    if set(value) != required:
        raise JoinError(JoinFailure.INVALID_REQUEST, f"fixture keys mismatch missing={sorted(required-set(value))} extra={sorted(set(value)-required)}")
    if value["schema"] != FIXTURE_SCHEMA or value["frame"] != PROJECT_FRAME:
        raise JoinError(JoinFailure.INVALID_REQUEST, "fixture schema or project frame mismatch")
    join_id = value["join_id"]
    if not isinstance(join_id, str) or not join_id.strip():
        raise JoinError(JoinFailure.INVALID_REQUEST, "join_id required")
    try: family = JoinFamily(str(value["join_family"]))
    except ValueError as exc: raise JoinError(JoinFailure.INVALID_REQUEST, "unsupported join family") from exc
    t = value["tolerances"]
    if set(t) != {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad", "mesh_round_digits"}:
        raise JoinError(JoinFailure.INVALID_REQUEST, "tolerance keys mismatch")
    linear = _number(t["linear_m"], "tolerances.linear_m")
    angular = _number(t["angular_rad"], "tolerances.angular_rad")
    if not (1e-9 <= linear <= 1e-3 and 1e-9 <= angular <= 1e-3):
        raise JoinError(JoinFailure.INVALID_REQUEST, "model tolerance outside R0B contract")
    if _number(t["tessellation_linear_m"], "tessellation_linear_m") != 0.05 or _number(t["tessellation_angular_rad"], "tessellation_angular_rad") != 0.1 or int(t["mesh_round_digits"]) != 9:
        raise JoinError(JoinFailure.INVALID_REQUEST, "fixed tessellation mismatch")
    anchor = vec3(value["anchor_m"], "anchor_m")
    incoming = validate_frame(value["incoming_socket"], "incoming_socket", max(linear, angular))
    outgoing = validate_frame(value["outgoing_socket"], "outgoing_socket", max(linear, angular))
    in_profile = str(value["incoming_profile"]); out_profile = str(value["outgoing_profile"])
    if in_profile not in PROFILES or out_profile not in PROFILES:
        raise JoinError(JoinFailure.INVALID_REQUEST, "unknown wall profile")
    depth = _number(value["join_depth_m"], "join_depth_m")
    setback = _number(value["bevel_setback_m"], "bevel_setback_m")
    overlap = _number(value["max_overlap_m"], "max_overlap_m")
    if not (0.25 <= depth <= 12.0) or not (0.0 <= setback < depth) or not (0.0 <= overlap <= 0.10):
        raise JoinError(JoinFailure.INVALID_REQUEST, "join depth/setback/overlap outside bounded domain")
    turn = abs(signed_turn_deg(incoming["tangent"], outgoing["tangent"]))
    if family is JoinFamily.MITER and not (0.1 <= turn <= 15.0 + 1e-9):
        raise JoinError(JoinFailure.TURN_OUT_OF_DOMAIN, f"miter turn {turn} outside [0.1,15]")
    if family is JoinFamily.BEVEL and not (15.0 < turn <= 35.0 + 1e-9 and 0.1 <= setback < depth):
        raise JoinError(JoinFailure.TURN_OUT_OF_DOMAIN, f"bevel turn/setback invalid turn={turn} setback={setback}")
    if family is JoinFamily.PROFILE_TRANSITION:
        if turn > 1e-6: raise JoinError(JoinFailure.TURN_OUT_OF_DOMAIN, "profile transition must be straight")
        if in_profile == out_profile: raise JoinError(JoinFailure.PROFILE_TRANSITION_REQUIRED, "profile transition requires distinct profiles")
    in_pos, out_pos = incoming["position_m"], outgoing["position_m"]
    in_anchor = add(in_pos, mul(incoming["tangent"], depth))
    if family is JoinFamily.PROFILE_TRANSITION:
        expected_out = add(in_pos, mul(incoming["tangent"], depth))
        if distance(anchor, in_pos) > linear or distance(out_pos, expected_out) > linear:
            raise JoinError(JoinFailure.SOCKET_GEOMETRY_MISMATCH, "transition sockets do not match anchor/length")
    else:
        out_anchor = sub(out_pos, mul(outgoing["tangent"], depth))
        if distance(in_anchor, anchor) > linear or distance(out_anchor, anchor) > linear:
            raise JoinError(JoinFailure.SOCKET_GEOMETRY_MISMATCH, "join sockets do not meet anchor at join_depth")
    budget = value["budget"]
    if set(budget) != {"max_components", "max_vertices", "max_triangles", "max_artifact_bytes"}:
        raise JoinError(JoinFailure.INVALID_REQUEST, "budget keys mismatch")
    for key in budget:
        if isinstance(budget[key], bool) or not isinstance(budget[key], int) or budget[key] <= 0:
            raise JoinError(JoinFailure.INVALID_REQUEST, f"budget.{key} must be positive integer")
    runtime = dict(value["runtime"])
    if runtime != EXPECTED_RUNTIME:
        raise JoinError(JoinFailure.RUNTIME_MISMATCH, f"runtime mismatch {runtime}")
    return {
        **dict(value), "join_family": family.value, "anchor_m": list(anchor),
        "incoming_socket": incoming, "outgoing_socket": outgoing,
        "turn_angle_deg": round(turn, 9),
    }


def profile(profile_id: str) -> WallProfile:
    return PROFILES[profile_id]


def band_map(profile_id: str) -> dict[str, Band]:
    return {b.part_id: b for b in profile(profile_id).bands}
