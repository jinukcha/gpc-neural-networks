from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

JOIN_SCHEMA = "royal-capital.fortification.wall-join/1"
JOIN_PLAN_SCHEMA = "royal-capital.fortification.wall-join-plan/1"
JOIN_RESULT_SCHEMA = "royal-capital.fortification.wall-join-result/1"
JOIN_RECEIPT_SCHEMA = "royal-capital.fortification.wall-join-receipt/1"
JOIN_MESH_SCHEMA = "royal-capital.fortification.wall-join-indexed-mesh/1"
JOIN_PARTS_SCHEMA = "royal-capital.fortification.wall-join-semantic-parts/1"
JOIN_SOCKETS_SCHEMA = "royal-capital.fortification.wall-join-sockets/1"
SOCKET_ALIGNMENT_SCHEMA = "royal-capital.fortification.socket-alignment/1"
OVERLAP_SCHEMA = "royal-capital.fortification.bounded-overlap/1"
STORED_SCHEMA = "royal-capital.fortification.wall-join-stored-copies/1"
TESSELLATION_SCHEMA = "royal-capital.fortification.wall-join-tessellation/1"
PROJECT_FRAME = "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"
PART_ORDER = ("foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet")
FAMILIES = ("MITER", "BEVEL", "PROFILE_TRANSITION")
ALIGNMENT_LEVELS = (
    ("span", "SPAN_JOIN", 6.0),
    ("wall_walk", "WALL_WALK_CONTINUATION", 12.3),
    ("foundation", "FOUNDATION_INTERFACE", -1.0),
)
EXPECTED_RUNTIME = {
    "build123d_version": "0.13.1.dev12+ge22d34dae",
    "ocp_version": "8.0.1.0.0",
    "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466",
    "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f",
}


class JoinContractError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_ref(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _f(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JoinContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise JoinContractError(f"{name} must be finite")
    return result


def _v3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise JoinContractError(f"{name} must contain exactly three numbers")
    return tuple(_f(item, f"{name}[{index}]") for index, item in enumerate(value))


def _round(value: float, digits: int = 9) -> float:
    result = round(float(value), digits)
    return 0.0 if result == -0.0 else result


def _rv(value: Sequence[float], digits: int = 9) -> list[float]:
    return [_round(item, digits) for item in value]


def add(a, b): return tuple(a[index] + b[index] for index in range(3))
def sub(a, b): return tuple(a[index] - b[index] for index in range(3))
def mul(a, scalar): return tuple(a[index] * scalar for index in range(3))
def dot(a, b): return sum(a[index] * b[index] for index in range(3))
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def length(vector): return math.sqrt(dot(vector, vector))


def normalize(vector):
    magnitude = length(vector)
    if magnitude <= 1e-15:
        raise JoinContractError("zero-length vector")
    return tuple(item / magnitude for item in vector)


def angle_rad(a, b) -> float:
    na, nb = normalize(a), normalize(b)
    return math.acos(max(-1.0, min(1.0, dot(na, nb))))


def signed_turn_deg(incoming, outgoing) -> float:
    incoming = normalize(incoming)
    outgoing = normalize(outgoing)
    if abs(incoming[1]) > 1e-9 or abs(outgoing[1]) > 1e-9:
        raise JoinContractError("graded/vertical join deferred to R0B-CP3")
    unsigned = math.degrees(angle_rad(incoming, outgoing))
    sign = 1.0 if cross(incoming, outgoing)[1] < 0.0 else -1.0
    # In +Y-up, +X to +Z is a left turn under the project -Z-forward convention.
    return _round(sign * unsigned, 9)


def frame_from_tangent(origin, tangent, digits: int = 9) -> dict[str, Any]:
    tangent = normalize(tangent)
    up = (0.0, 1.0, 0.0)
    if abs(dot(tangent, up)) > 1e-9:
        raise JoinContractError("non-horizontal tangent unsupported")
    inside = normalize(cross(tangent, up))
    outside = mul(inside, -1.0)
    determinant = dot(cross(tangent, up), inside)
    if abs(determinant - 1.0) > 1e-9:
        raise JoinContractError("local frame determinant mismatch")
    return {
        "origin_m": _rv(origin, digits),
        "tangent": _rv(tangent, digits),
        "up": [0.0, 1.0, 0.0],
        "inside": _rv(inside, digits),
        "outside": _rv(outside, digits),
        "orientation_determinant": 1.0,
    }


@dataclass(frozen=True)
class ProfilePart:
    part_id: str
    semantic_role: str
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    material_slot: str

    @property
    def area_m2(self) -> float:
        return (self.y_max - self.y_min) * (self.z_max - self.z_min)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "semantic_role": self.semantic_role,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "material_slot": self.material_slot,
        }


def standard_profile(profile_id: str = "profile/caelmere/standard@1") -> dict[str, Any]:
    parts = (
        ProfilePart("foundation", "FOUNDATION", -2.0, 0.0, -4.0, 4.0, "stone_foundation"),
        ProfilePart("wall_body", "WALL_BODY", 0.0, 12.0, -3.0, 3.0, "stone_wall"),
        ProfilePart("wall_walk", "WALL_WALK", 12.0, 12.6, -3.2, 3.2, "stone_walk"),
        ProfilePart("inner_parapet", "INNER_PARAPET", 12.6, 14.4, 2.4, 3.2, "stone_parapet"),
        ProfilePart("outer_parapet", "OUTER_PARAPET", 12.6, 14.4, -3.2, -2.4, "stone_parapet"),
    )
    return {"profile_id": profile_id, "parts": [part.to_dict() for part in parts]}


def reinforced_profile(profile_id: str = "profile/caelmere/reinforced@1") -> dict[str, Any]:
    parts = (
        ProfilePart("foundation", "FOUNDATION", -2.0, 0.0, -5.0, 5.0, "stone_foundation"),
        ProfilePart("wall_body", "WALL_BODY", 0.0, 12.0, -3.6, 3.6, "stone_wall"),
        ProfilePart("wall_walk", "WALL_WALK", 12.0, 12.6, -3.8, 3.8, "stone_walk"),
        ProfilePart("inner_parapet", "INNER_PARAPET", 12.6, 14.4, 3.0, 3.8, "stone_parapet"),
        ProfilePart("outer_parapet", "OUTER_PARAPET", 12.6, 14.4, -3.8, -3.0, "stone_parapet"),
    )
    return {"profile_id": profile_id, "parts": [part.to_dict() for part in parts]}


def _profile(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"profile_id", "parts"}:
        raise JoinContractError(f"{name} keys mismatch")
    if not isinstance(value["profile_id"], str) or not value["profile_id"].strip():
        raise JoinContractError(f"{name}.profile_id required")
    rows = value["parts"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != len(PART_ORDER):
        raise JoinContractError(f"{name}.parts must contain five parts")
    parsed = []
    for index, row in enumerate(rows):
        required = {"part_id", "semantic_role", "y_min", "y_max", "z_min", "z_max", "material_slot"}
        if not isinstance(row, Mapping) or set(row) != required:
            raise JoinContractError(f"{name}.parts[{index}] keys mismatch")
        part_id = str(row["part_id"])
        if part_id != PART_ORDER[index]:
            raise JoinContractError(f"{name} part order mismatch at {index}: {part_id}")
        y_min, y_max = _f(row["y_min"], f"{name}.{part_id}.y_min"), _f(row["y_max"], f"{name}.{part_id}.y_max")
        z_min, z_max = _f(row["z_min"], f"{name}.{part_id}.z_min"), _f(row["z_max"], f"{name}.{part_id}.z_max")
        if y_max <= y_min or z_max <= z_min:
            raise JoinContractError(f"{name}.{part_id} has non-positive section")
        parsed.append({
            "part_id": part_id,
            "semantic_role": str(row["semantic_role"]),
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "material_slot": str(row["material_slot"]),
        })
    return {"profile_id": str(value["profile_id"]), "parts": parsed}


def profile_part(profile: Mapping[str, Any], part_id: str) -> dict[str, Any]:
    for row in profile["parts"]:
        if row["part_id"] == part_id:
            return dict(row)
    raise JoinContractError(f"profile missing part {part_id}")


def interpolate_part(a: Mapping[str, Any], b: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    alpha = float(alpha)
    return {
        "part_id": a["part_id"],
        "semantic_role": a["semantic_role"],
        "y_min": (1.0-alpha)*float(a["y_min"]) + alpha*float(b["y_min"]),
        "y_max": (1.0-alpha)*float(a["y_max"]) + alpha*float(b["y_max"]),
        "z_min": (1.0-alpha)*float(a["z_min"]) + alpha*float(b["z_min"]),
        "z_max": (1.0-alpha)*float(a["z_max"]) + alpha*float(b["z_max"]),
        "material_slot": a["material_slot"] if alpha < 0.5 else b["material_slot"],
    }


def _source_end(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    required = {"span_id", "span_length_m", "endpoint", "anchor_m", "frame", "sockets"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise JoinContractError(f"{name} keys mismatch")
    if value["endpoint"] not in {"START", "END"}:
        raise JoinContractError(f"{name}.endpoint invalid")
    anchor = _v3(value["anchor_m"], f"{name}.anchor_m")
    frame = value["frame"]
    required_frame = {"tangent", "up", "inside", "outside", "orientation_determinant"}
    if not isinstance(frame, Mapping) or set(frame) != required_frame:
        raise JoinContractError(f"{name}.frame keys mismatch")
    tangent = normalize(_v3(frame["tangent"], f"{name}.frame.tangent"))
    up = normalize(_v3(frame["up"], f"{name}.frame.up"))
    inside = normalize(_v3(frame["inside"], f"{name}.frame.inside"))
    outside = normalize(_v3(frame["outside"], f"{name}.frame.outside"))
    if abs(dot(tangent, up)) > 1e-9 or abs(dot(tangent, inside)) > 1e-9 or abs(dot(up, inside)) > 1e-9:
        raise JoinContractError(f"{name}.frame not orthogonal")
    if max(abs(inside[index] + outside[index]) for index in range(3)) > 1e-9:
        raise JoinContractError(f"{name}.outside must equal -inside")
    if abs(dot(cross(tangent, up), inside) - 1.0) > 1e-9 or abs(float(frame["orientation_determinant"]) - 1.0) > 1e-9:
        raise JoinContractError(f"{name}.frame handedness mismatch")
    sockets = value["sockets"]
    expected = {level for level, _, _ in ALIGNMENT_LEVELS}
    if not isinstance(sockets, Mapping) or set(sockets) != expected:
        raise JoinContractError(f"{name}.sockets keys mismatch")
    parsed_sockets = {}
    for level, role, y in ALIGNMENT_LEVELS:
        row = sockets[level]
        required_socket = {"socket_id", "role", "position_m", "frame"}
        if not isinstance(row, Mapping) or set(row) != required_socket:
            raise JoinContractError(f"{name}.sockets.{level} keys mismatch")
        position = _v3(row["position_m"], f"{name}.sockets.{level}.position_m")
        expected_position = add(anchor, mul(up, y))
        if length(sub(position, expected_position)) > 1e-8:
            raise JoinContractError(f"{name}.sockets.{level} position mismatch")
        if row["role"] != role:
            raise JoinContractError(f"{name}.sockets.{level} role mismatch")
        socket_frame = row["frame"]
        if socket_frame != frame:
            raise JoinContractError(f"{name}.sockets.{level} frame mismatch")
        parsed_sockets[level] = {
            "socket_id": str(row["socket_id"]),
            "role": str(row["role"]),
            "position_m": _rv(position),
            "frame": {
                "tangent": _rv(tangent), "up": _rv(up), "inside": _rv(inside), "outside": _rv(outside),
                "orientation_determinant": 1.0,
            },
        }
    span_length = _f(value["span_length_m"], f"{name}.span_length_m")
    if not (10.0 <= span_length <= 60.0):
        raise JoinContractError(f"{name}.span_length_m outside [10,60]")
    return {
        "span_id": str(value["span_id"]),
        "span_length_m": span_length,
        "endpoint": value["endpoint"],
        "anchor_m": _rv(anchor),
        "frame": {
            "tangent": _rv(tangent), "up": _rv(up), "inside": _rv(inside), "outside": _rv(outside),
            "orientation_determinant": 1.0,
        },
        "sockets": parsed_sockets,
    }


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "join_id", "family", "frame", "incoming", "outgoing", "incoming_profile", "outgoing_profile", "overlap", "tolerances", "budget", "runtime"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise JoinContractError(f"fixture keys mismatch missing={sorted(required-set(value))} extra={sorted(set(value)-required)}")
    if value["schema"] != JOIN_SCHEMA:
        raise JoinContractError("unsupported join schema")
    if not isinstance(value["join_id"], str) or not value["join_id"].strip():
        raise JoinContractError("join_id required")
    family = str(value["family"])
    if family not in FAMILIES:
        raise JoinContractError(f"unsupported join family {family!r}")
    if value["frame"] != PROJECT_FRAME:
        raise JoinContractError("project frame mismatch")
    if value["runtime"] != EXPECTED_RUNTIME:
        raise JoinContractError(f"runtime mismatch: {value['runtime']}")
    incoming = _source_end(value["incoming"], "incoming")
    outgoing = _source_end(value["outgoing"], "outgoing")
    if incoming["endpoint"] != "END" or outgoing["endpoint"] != "START":
        raise JoinContractError("join requires incoming END and outgoing START")
    if length(sub(incoming["anchor_m"], outgoing["anchor_m"])) > 1e-8:
        raise JoinContractError("source socket anchors do not coincide")
    turn = signed_turn_deg(incoming["frame"]["tangent"], outgoing["frame"]["tangent"])
    abs_turn = abs(turn)
    if abs_turn >= 179.0:
        raise JoinContractError("opposed tangents unsupported")
    if family == "MITER" and not (1.0 <= abs_turn <= 15.0):
        raise JoinContractError("MITER turn must be within [1,15] degrees")
    if family == "BEVEL" and not (15.0 < abs_turn <= 35.0):
        raise JoinContractError("BEVEL turn must be within (15,35] degrees")
    if family == "PROFILE_TRANSITION" and abs_turn > 5.0:
        raise JoinContractError("PROFILE_TRANSITION turn must be <=5 degrees")
    incoming_profile = _profile(value["incoming_profile"], "incoming_profile")
    outgoing_profile = _profile(value["outgoing_profile"], "outgoing_profile")
    if family in {"MITER", "BEVEL"} and incoming_profile != outgoing_profile:
        raise JoinContractError(f"{family} requires identical profiles")
    if family == "PROFILE_TRANSITION" and incoming_profile == outgoing_profile:
        raise JoinContractError("PROFILE_TRANSITION requires distinct profiles")
    overlap = value["overlap"]
    required_overlap = {"incoming_m", "outgoing_m", "max_m", "bevel_setback_m"}
    if not isinstance(overlap, Mapping) or set(overlap) != required_overlap:
        raise JoinContractError("overlap keys mismatch")
    incoming_overlap = _f(overlap["incoming_m"], "overlap.incoming_m")
    outgoing_overlap = _f(overlap["outgoing_m"], "overlap.outgoing_m")
    maximum = _f(overlap["max_m"], "overlap.max_m")
    setback = _f(overlap["bevel_setback_m"], "overlap.bevel_setback_m")
    if not (0.25 <= incoming_overlap <= maximum <= 12.0) or not (0.25 <= outgoing_overlap <= maximum):
        raise JoinContractError("overlap outside bounded domain")
    if incoming_overlap > incoming["span_length_m"] * 0.5 or outgoing_overlap > outgoing["span_length_m"] * 0.5:
        raise JoinContractError("overlap exceeds half of source span")
    if family == "BEVEL":
        if not (0.25 <= setback < min(incoming_overlap, outgoing_overlap)):
            raise JoinContractError("BEVEL setback outside overlap")
    elif abs(setback) > 1e-12:
        raise JoinContractError("bevel_setback_m must be zero outside BEVEL")
    tolerances = value["tolerances"]
    required_tolerances = {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad", "mesh_round_digits", "socket_position_m", "socket_angle_rad"}
    if not isinstance(tolerances, Mapping) or set(tolerances) != required_tolerances:
        raise JoinContractError("tolerance keys mismatch")
    if not (1e-9 <= _f(tolerances["linear_m"], "linear_m") <= 1e-3): raise JoinContractError("linear tolerance out of domain")
    if not (1e-9 <= _f(tolerances["angular_rad"], "angular_rad") <= 1e-3): raise JoinContractError("angular tolerance out of domain")
    if not (1e-3 <= _f(tolerances["tessellation_linear_m"], "tessellation_linear_m") <= 0.1): raise JoinContractError("tessellation linear out of domain")
    if not (0.01 <= _f(tolerances["tessellation_angular_rad"], "tessellation_angular_rad") <= 0.3): raise JoinContractError("tessellation angular out of domain")
    if int(tolerances["mesh_round_digits"]) != 9: raise JoinContractError("mesh_round_digits must be 9")
    if not (1e-9 <= _f(tolerances["socket_position_m"], "socket_position_m") <= 1e-3): raise JoinContractError("socket position tolerance out of domain")
    if not (1e-9 <= _f(tolerances["socket_angle_rad"], "socket_angle_rad") <= 1e-3): raise JoinContractError("socket angle tolerance out of domain")
    budget = value["budget"]
    required_budget = {"max_vertices", "max_triangles", "max_artifact_bytes", "max_sections"}
    if not isinstance(budget, Mapping) or set(budget) != required_budget:
        raise JoinContractError("budget keys mismatch")
    for key in required_budget:
        if not isinstance(budget[key], int) or budget[key] <= 0:
            raise JoinContractError(f"budget.{key} must be positive integer")
    required_sections = {"MITER": 3, "BEVEL": 4, "PROFILE_TRANSITION": 3}[family]
    if budget["max_sections"] < required_sections:
        raise JoinContractError("required sections exceed join budget")
    result = json.loads(json.dumps(value))
    result["incoming"] = incoming
    result["outgoing"] = outgoing
    result["incoming_profile"] = incoming_profile
    result["outgoing_profile"] = outgoing_profile
    result["turn_angle_deg"] = turn
    result["required_sections"] = required_sections
    return result


def _section(origin, tangent, profile: Mapping[str, Any], section_id: str) -> dict[str, Any]:
    frame = frame_from_tangent(origin, tangent)
    return {"section_id": section_id, **frame, "profile": profile}


def section_plan(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    incoming = spec["incoming"]
    outgoing = spec["outgoing"]
    anchor = tuple(float(item) for item in incoming["anchor_m"])
    ti = tuple(float(item) for item in incoming["frame"]["tangent"])
    to = tuple(float(item) for item in outgoing["frame"]["tangent"])
    oi = float(spec["overlap"]["incoming_m"])
    oo = float(spec["overlap"]["outgoing_m"])
    incoming_profile = spec["incoming_profile"]
    outgoing_profile = spec["outgoing_profile"]
    family = spec["family"]
    if family == "MITER":
        bisector = normalize(add(ti, to))
        half = math.radians(abs(float(spec["turn_angle_deg"])) * 0.5)
        scale = 1.0 / math.cos(half)
        middle_profile = json.loads(json.dumps(incoming_profile))
        middle_profile["profile_id"] = incoming_profile["profile_id"] + "/miter-expanded"
        for row in middle_profile["parts"]:
            row["z_min"] = float(row["z_min"]) * scale
            row["z_max"] = float(row["z_max"]) * scale
        return [
            _section(sub(anchor, mul(ti, oi)), ti, incoming_profile, "incoming_overlap"),
            _section(anchor, bisector, middle_profile, "miter_apex"),
            _section(add(anchor, mul(to, oo)), to, outgoing_profile, "outgoing_overlap"),
        ]
    if family == "BEVEL":
        setback = float(spec["overlap"]["bevel_setback_m"])
        return [
            _section(sub(anchor, mul(ti, oi)), ti, incoming_profile, "incoming_overlap"),
            _section(sub(anchor, mul(ti, setback)), ti, incoming_profile, "incoming_bevel_cut"),
            _section(add(anchor, mul(to, setback)), to, outgoing_profile, "outgoing_bevel_cut"),
            _section(add(anchor, mul(to, oo)), to, outgoing_profile, "outgoing_overlap"),
        ]
    middle = {"profile_id": incoming_profile["profile_id"] + "->" + outgoing_profile["profile_id"] + "/mid", "parts": []}
    for part_id in PART_ORDER:
        middle["parts"].append(interpolate_part(profile_part(incoming_profile, part_id), profile_part(outgoing_profile, part_id), 0.5))
    tangent = normalize(add(ti, to))
    return [
        _section(sub(anchor, mul(ti, oi)), ti, incoming_profile, "incoming_overlap"),
        _section(anchor, tangent, middle, "transition_mid"),
        _section(add(anchor, mul(to, oo)), to, outgoing_profile, "outgoing_overlap"),
    ]


def interface_sockets(spec: Mapping[str, Any]) -> dict[str, Any]:
    sockets = []
    for side in ("incoming", "outgoing"):
        source = spec[side]
        endpoint = source["endpoint"].lower()
        for level, role, _ in ALIGNMENT_LEVELS:
            src = source["sockets"][level]
            sockets.append({
                "socket_id": f"{side}_{level}",
                "role": role,
                "source_span_id": source["span_id"],
                "source_socket_id": src["socket_id"],
                "source_endpoint": source["endpoint"],
                "position_m": src["position_m"],
                "frame": src["frame"],
                "required": True,
                "endpoint": endpoint,
            })
    return {"schema": JOIN_SOCKETS_SCHEMA, "join_id": spec["join_id"], "sockets": sockets}


def socket_alignment(spec: Mapping[str, Any], sockets_doc: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    by_id = {row["socket_id"]: row for row in sockets_doc["sockets"]}
    position_tolerance = float(spec["tolerances"]["socket_position_m"])
    angle_tolerance = float(spec["tolerances"]["socket_angle_rad"])
    for side in ("incoming", "outgoing"):
        source = spec[side]
        for level, _, _ in ALIGNMENT_LEVELS:
            src = source["sockets"][level]
            join = by_id[f"{side}_{level}"]
            position_error = length(sub(src["position_m"], join["position_m"]))
            tangent_error = angle_rad(src["frame"]["tangent"], join["frame"]["tangent"])
            up_error = angle_rad(src["frame"]["up"], join["frame"]["up"])
            inside_error = angle_rad(src["frame"]["inside"], join["frame"]["inside"])
            passed = position_error <= position_tolerance and max(tangent_error, up_error, inside_error) <= angle_tolerance
            rows.append({
                "alignment_id": f"alignment/{side}/{level}",
                "source_span_id": source["span_id"],
                "source_socket_id": src["socket_id"],
                "join_socket_id": join["socket_id"],
                "position_error_m": _round(position_error, 12),
                "tangent_angle_error_rad": _round(tangent_error, 12),
                "up_angle_error_rad": _round(up_error, 12),
                "inside_angle_error_rad": _round(inside_error, 12),
                "exact_numeric_equal": src["position_m"] == join["position_m"] and src["frame"] == join["frame"],
                "status": "PASS" if passed else "FAIL",
            })
    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    return {
        "schema": SOCKET_ALIGNMENT_SCHEMA,
        "join_id": spec["join_id"],
        "status": status,
        "position_tolerance_m": position_tolerance,
        "angle_tolerance_rad": angle_tolerance,
        "alignment_count": len(rows),
        "alignments": rows,
    }


def bounded_overlap(spec: Mapping[str, Any], sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    incoming_overlap = float(spec["overlap"]["incoming_m"])
    outgoing_overlap = float(spec["overlap"]["outgoing_m"])
    maximum = float(spec["overlap"]["max_m"])
    incoming_ratio = incoming_overlap / float(spec["incoming"]["span_length_m"])
    outgoing_ratio = outgoing_overlap / float(spec["outgoing"]["span_length_m"])
    section_ids = [str(row["section_id"]) for row in sections]
    return {
        "schema": OVERLAP_SCHEMA,
        "join_id": spec["join_id"],
        "family": spec["family"],
        "policy": "CENTERLINE_SECTION_INTERVAL_WITH_CONSERVATIVE_PROFILE_ENVELOPE",
        "incoming": {
            "overlap_m": incoming_overlap,
            "max_m": maximum,
            "source_span_length_m": float(spec["incoming"]["span_length_m"]),
            "ratio": _round(incoming_ratio, 12),
            "interval_m": [-incoming_overlap, 0.0],
            "status": "PASS" if incoming_overlap <= maximum and incoming_ratio <= 0.5 else "FAIL",
        },
        "outgoing": {
            "overlap_m": outgoing_overlap,
            "max_m": maximum,
            "source_span_length_m": float(spec["outgoing"]["span_length_m"]),
            "ratio": _round(outgoing_ratio, 12),
            "interval_m": [0.0, outgoing_overlap],
            "status": "PASS" if outgoing_overlap <= maximum and outgoing_ratio <= 0.5 else "FAIL",
        },
        "bevel_setback_m": float(spec["overlap"]["bevel_setback_m"]),
        "section_ids": section_ids,
        "required_section_count": int(spec["required_sections"]),
        "observed_section_count": len(sections),
        "status": "PASS" if len(sections) == int(spec["required_sections"]) and incoming_overlap <= maximum and outgoing_overlap <= maximum and incoming_ratio <= 0.5 and outgoing_ratio <= 0.5 else "FAIL",
    }
