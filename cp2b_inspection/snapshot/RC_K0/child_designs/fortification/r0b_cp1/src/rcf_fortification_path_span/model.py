from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

PATH_SPAN_SCHEMA = "royal-capital.fortification.path-wall-span/1"
CENTERLINE_SCHEMA = "royal-capital.fortification.canonical-centerline/1"
LOCAL_FRAMES_SCHEMA = "royal-capital.fortification.local-frames/1"
PLAN_SCHEMA = "royal-capital.fortification.path-wall-span-plan/1"
RESULT_SCHEMA = "royal-capital.fortification.path-wall-span-result/1"
RECEIPT_SCHEMA = "royal-capital.fortification.path-wall-span-receipt/1"
COMBINED_MESH_SCHEMA = "royal-capital.fortification.path-wall-span-indexed-mesh/1"
PARTS_SCHEMA = "royal-capital.fortification.path-wall-span-semantic-parts/1"
SOCKETS_SCHEMA = "royal-capital.fortification.path-wall-span-sockets/1"
STORED_SCHEMA = "royal-capital.fortification.path-wall-span-stored-copies/1"
TESSELLATION_SCHEMA = "royal-capital.fortification.path-wall-span-tessellation/1"
PROJECT_FRAME = "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"
PART_ORDER = ("foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet")
SOCKET_ORDER = (
    "span_start", "span_end", "wall_walk_start", "wall_walk_end",
    "foundation_start", "foundation_end", "tower_start", "tower_end",
    "utility_inside_01", "utility_inside_02",
)

EXPECTED_RUNTIME = {
    "build123d_version": "0.13.1.dev12+ge22d34dae",
    "ocp_version": "8.0.1.0.0",
    "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466",
    "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f",
}


class PathContractError(ValueError):
    pass


def _f(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PathContractError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise PathContractError(f"{name} must be finite")
    return out


def _v3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise PathContractError(f"{name} must contain exactly three numbers")
    return tuple(_f(v, f"{name}[{i}]") for i, v in enumerate(value))


def _round(v: float, digits: int = 9) -> float:
    out = round(float(v), digits)
    return 0.0 if out == -0.0 else out


def _rv(v: Sequence[float], digits: int = 9) -> list[float]:
    return [_round(x, digits) for x in v]


def add(a, b): return tuple(a[i] + b[i] for i in range(3))
def sub(a, b): return tuple(a[i] - b[i] for i in range(3))
def mul(a, s): return tuple(a[i] * s for i in range(3))
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def length(v): return math.sqrt(dot(v, v))
def normalize(v):
    n = length(v)
    if n <= 1e-15: raise PathContractError("zero-length vector")
    return tuple(x / n for x in v)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_ref(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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
    def area_m2(self) -> float:
        return (self.y_max - self.y_min) * (self.z_max - self.z_min)

    @property
    def average_z_m(self) -> float:
        return (self.z_min + self.z_max) * 0.5


PART_SPECS = (
    PartSpec("foundation", "FOUNDATION", -2.0, 0.0, -4.0, 4.0, "stone_foundation"),
    PartSpec("wall_body", "WALL_BODY", 0.0, 12.0, -3.0, 3.0, "stone_wall"),
    PartSpec("wall_walk", "WALL_WALK", 12.0, 12.6, -3.2, 3.2, "stone_walk"),
    PartSpec("inner_parapet", "INNER_PARAPET", 12.6, 14.4, 2.4, 3.2, "stone_parapet"),
    PartSpec("outer_parapet", "OUTER_PARAPET", 12.6, 14.4, -3.2, -2.4, "stone_parapet"),
)


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "span_id", "frame", "inside_side", "outside_side", "centerline", "sampling", "tolerances", "budget", "runtime"}
    if set(value) != required:
        raise PathContractError(f"fixture keys mismatch missing={sorted(required-set(value))} extra={sorted(set(value)-required)}")
    if value["schema"] != PATH_SPAN_SCHEMA:
        raise PathContractError("unsupported path-span schema")
    if not isinstance(value["span_id"], str) or not value["span_id"].strip():
        raise PathContractError("span_id required")
    if value["frame"] != PROJECT_FRAME:
        raise PathContractError("project frame mismatch")
    if value["inside_side"] != "LEFT_OF_TRAVEL" or value["outside_side"] != "RIGHT_OF_TRAVEL":
        raise PathContractError("CP1 requires LEFT_OF_TRAVEL inside and RIGHT_OF_TRAVEL outside")
    runtime = value["runtime"]
    if runtime != EXPECTED_RUNTIME:
        raise PathContractError(f"runtime mismatch: {runtime}")
    sampling = value["sampling"]
    if set(sampling) != {"curve_sample_tolerance_m", "max_turn_deg", "max_segments", "round_digits"}:
        raise PathContractError("sampling keys mismatch")
    tol = _f(sampling["curve_sample_tolerance_m"], "sampling.curve_sample_tolerance_m")
    turn = _f(sampling["max_turn_deg"], "sampling.max_turn_deg")
    max_segments = sampling["max_segments"]
    digits = sampling["round_digits"]
    if not (0.01 <= tol <= 0.25): raise PathContractError("curve sample tolerance outside [0.01,0.25]")
    if not (1.0 <= turn <= 35.0): raise PathContractError("max turn outside [1,35]")
    if not isinstance(max_segments, int) or not (1 <= max_segments <= 4096): raise PathContractError("max_segments outside [1,4096]")
    if not isinstance(digits, int) or not (6 <= digits <= 12): raise PathContractError("round_digits outside [6,12]")
    t = value["tolerances"]
    if set(t) != {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad", "mesh_round_digits"}:
        raise PathContractError("tolerance keys mismatch")
    if not (1e-9 <= _f(t["linear_m"], "linear_m") <= 1e-3): raise PathContractError("linear tolerance out of domain")
    if not (1e-9 <= _f(t["angular_rad"], "angular_rad") <= 1e-3): raise PathContractError("angular tolerance out of domain")
    if not (1e-3 <= _f(t["tessellation_linear_m"], "tessellation_linear_m") <= 0.1): raise PathContractError("tessellation linear out of domain")
    if not (0.01 <= _f(t["tessellation_angular_rad"], "tessellation_angular_rad") <= 0.3): raise PathContractError("tessellation angular out of domain")
    if int(t["mesh_round_digits"]) != 9: raise PathContractError("mesh_round_digits must be 9")
    b = value["budget"]
    if set(b) != {"max_vertices", "max_triangles", "max_artifact_bytes", "max_sections"}:
        raise PathContractError("budget keys mismatch")
    for key in b:
        if not isinstance(b[key], int) or b[key] <= 0: raise PathContractError(f"budget.{key} must be positive integer")
    c = value["centerline"]
    if not isinstance(c, Mapping) or "kind" not in c:
        raise PathContractError("centerline object required")
    kind = c["kind"]
    if kind == "LINE":
        if set(c) != {"kind", "start_m", "end_m"}: raise PathContractError("LINE keys mismatch")
        start, end = _v3(c["start_m"], "centerline.start_m"), _v3(c["end_m"], "centerline.end_m")
        if abs(start[1] - end[1]) > t["linear_m"]: raise PathContractError("vertical/graded centerline deferred to R0B-CP3")
        d = sub(end, start)
        if length(d) <= t["linear_m"]: raise PathContractError("zero-length span")
        if length(d) > 60.0: raise PathContractError("span exceeds max 60m")
    elif kind == "CIRCULAR_ARC":
        expected = {"kind", "center_m", "radius_m", "start_angle_deg", "sweep_angle_deg", "elevation_m"}
        if set(c) != expected: raise PathContractError("CIRCULAR_ARC keys mismatch")
        center = _v3(c["center_m"], "centerline.center_m")
        radius = _f(c["radius_m"], "centerline.radius_m")
        sweep = _f(c["sweep_angle_deg"], "centerline.sweep_angle_deg")
        elevation = _f(c["elevation_m"], "centerline.elevation_m")
        _f(c["start_angle_deg"], "centerline.start_angle_deg")
        if abs(center[1] - elevation) > t["linear_m"]: raise PathContractError("arc center/elevation mismatch")
        if radius <= 4.0 + t["linear_m"]: raise PathContractError("arc radius must exceed maximum profile offset")
        if not (0.0 < abs(sweep) <= 35.0): raise PathContractError("arc sweep outside (0,35]")
        if math.radians(abs(sweep)) * radius > 60.0: raise PathContractError("arc length exceeds max 60m")
    else:
        raise PathContractError(f"unsupported centerline kind {kind!r}")
    return json.loads(json.dumps(value))


def _frame(point, tangent, distance_m, u, digits):
    tangent = normalize(tangent)
    up = (0.0, 1.0, 0.0)
    if abs(dot(tangent, up)) > 1e-9: raise PathContractError("non-horizontal tangent unsupported")
    inside = normalize(cross(tangent, up))
    outside = mul(inside, -1.0)
    determinant = dot(cross(tangent, up), inside)
    if abs(determinant - 1.0) > 1e-9: raise PathContractError("local frame determinant mismatch")
    return {
        "u": _round(u, digits), "distance_m": _round(distance_m, digits),
        "origin_m": _rv(point, digits), "tangent": _rv(tangent, digits), "up": [0.0,1.0,0.0],
        "inside": _rv(inside, digits), "outside": _rv(outside, digits),
        "orientation_determinant": 1.0,
    }


def _arc_point(center, radius, angle, elevation):
    return (center[0] + radius * math.cos(angle), elevation, center[2] + radius * math.sin(angle))


def _arc_tangent(angle, sign):
    return normalize((-math.sin(angle) * sign, 0.0, math.cos(angle) * sign))


def canonicalize_centerline(fixture: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = validate_fixture(fixture)
    c, sampling = spec["centerline"], spec["sampling"]
    digits = int(sampling["round_digits"])
    samples: list[dict[str, Any]] = []
    exact_bounds_points: list[tuple[float,float,float]] = []
    if c["kind"] == "LINE":
        start, end = _v3(c["start_m"], "start"), _v3(c["end_m"], "end")
        tangent = normalize(sub(end, start)); total = length(sub(end, start))
        samples = [_frame(start, tangent, 0.0, 0.0, digits), _frame(end, tangent, total, 1.0, digits)]
        exact_bounds_points = [start, end]
        segment_count = 1; radius = None; sweep_deg = 0.0; chord_error = 0.0
    else:
        center = _v3(c["center_m"], "center"); radius = float(c["radius_m"])
        start_angle = math.radians(float(c["start_angle_deg"])); sweep = math.radians(float(c["sweep_angle_deg"])); sign = 1.0 if sweep > 0 else -1.0
        chord_tol = float(sampling["curve_sample_tolerance_m"])
        ratio = max(-1.0, min(1.0, 1.0 - chord_tol / radius))
        chord_angle = 2.0 * math.acos(ratio)
        max_angle = min(math.radians(float(sampling["max_turn_deg"])), chord_angle if chord_angle > 1e-12 else math.radians(float(sampling["max_turn_deg"])))
        segment_count = max(1, math.ceil(abs(sweep) / max_angle))
        if segment_count > int(sampling["max_segments"]): raise PathContractError("required sample segments exceed sampling budget")
        if segment_count + 1 > int(spec["budget"]["max_sections"]): raise PathContractError("required sections exceed geometry budget")
        total = abs(sweep) * radius
        for i in range(segment_count + 1):
            u = i / segment_count; angle = start_angle + sweep * u
            samples.append(_frame(_arc_point(center, radius, angle, float(c["elevation_m"])), _arc_tangent(angle, sign), total*u, u, digits))
        candidates = [start_angle, start_angle + sweep]
        low, high = sorted((start_angle, start_angle+sweep))
        k0, k1 = math.floor(low/(math.pi/2))-1, math.ceil(high/(math.pi/2))+1
        for k in range(k0, k1+1):
            a = k*(math.pi/2)
            if low-1e-12 <= a <= high+1e-12: candidates.append(a)
        exact_bounds_points = [_arc_point(center, radius, a, float(c["elevation_m"])) for a in candidates]
        delta = abs(sweep) / segment_count
        chord_error = radius * (1.0 - math.cos(delta*0.5))
        sweep_deg = float(c["sweep_angle_deg"])
    bounds = {"min": [_round(min(p[i] for p in exact_bounds_points), digits) for i in range(3)], "max": [_round(max(p[i] for p in exact_bounds_points), digits) for i in range(3)]}
    polyline_length = sum(length(sub(samples[i+1]["origin_m"], samples[i]["origin_m"])) for i in range(len(samples)-1))
    centerline = {
        "schema": CENTERLINE_SCHEMA, "span_id": spec["span_id"], "frame": spec["frame"], "kind": c["kind"],
        "inside_side": spec["inside_side"], "outside_side": spec["outside_side"], "length_m": _round(total, digits),
        "polyline_length_m": _round(polyline_length, digits), "segment_count": segment_count,
        "curve_sample_tolerance_m": float(sampling["curve_sample_tolerance_m"]), "max_turn_deg": float(sampling["max_turn_deg"]),
        "maximum_observed_chord_error_m": _round(chord_error, digits), "bounds_m": bounds,
        "arc_radius_m": radius, "arc_sweep_deg": sweep_deg, "samples": samples,
    }
    centerline["canonical_digest"] = sha256_ref(canonical_json_bytes({k:v for k,v in centerline.items() if k != "canonical_digest"}))
    frames = {
        "schema": LOCAL_FRAMES_SCHEMA, "span_id": spec["span_id"], "frame": spec["frame"],
        "basis_order": ["tangent", "up", "inside"], "outside_relation": "outside=-inside",
        "orientation_determinant": 1.0, "frames": samples,
        "canonical_centerline_digest": centerline["canonical_digest"],
    }
    return centerline, frames


def evaluate_frame(centerline: Mapping[str, Any], u: float) -> dict[str, Any]:
    u = max(0.0, min(1.0, float(u)))
    samples = centerline["samples"]
    if centerline["kind"] == "LINE":
        a, b = samples[0], samples[-1]
        point = [a["origin_m"][i] + u*(b["origin_m"][i]-a["origin_m"][i]) for i in range(3)]
        result = dict(a); result["u"] = _round(u); result["distance_m"] = _round(u*centerline["length_m"]); result["origin_m"] = _rv(point)
        return result
    scaled = u * (len(samples)-1); i = min(len(samples)-2, int(math.floor(scaled))); alpha = scaled-i
    a,b=samples[i],samples[i+1]
    point=[a["origin_m"][j]+alpha*(b["origin_m"][j]-a["origin_m"][j]) for j in range(3)]
    tangent=normalize(tuple(a["tangent"][j]+alpha*(b["tangent"][j]-a["tangent"][j]) for j in range(3)))
    return _frame(point,tangent,u*centerline["length_m"],u,9)


def socket_plan(centerline: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("span_start", "SPAN_JOIN", 0.0, 6.0, 0.0, True), ("span_end", "SPAN_JOIN", 1.0, 6.0, 0.0, True),
        ("wall_walk_start", "WALL_WALK_CONTINUATION", 0.0, 12.3, 0.0, True), ("wall_walk_end", "WALL_WALK_CONTINUATION", 1.0, 12.3, 0.0, True),
        ("foundation_start", "FOUNDATION_INTERFACE", 0.0, -1.0, 0.0, True), ("foundation_end", "FOUNDATION_INTERFACE", 1.0, -1.0, 0.0, True),
        ("tower_start", "TOWER_JOIN", 0.0, 6.0, 0.0, True), ("tower_end", "TOWER_JOIN", 1.0, 6.0, 0.0, True),
        ("utility_inside_01", "UTILITY_INSIDE", 1/3, 6.0, 2.75, False), ("utility_inside_02", "UTILITY_INSIDE", 2/3, 6.0, 2.75, False),
    ]
    out=[]
    for sid,role,u,y,z,required in rows:
        f=evaluate_frame(centerline,u); p=f["origin_m"]; up=f["up"]; inside=f["inside"]
        position=[p[i]+y*up[i]+z*inside[i] for i in range(3)]
        out.append({"socket_id":sid,"role":role,"u":_round(u),"position_m":_rv(position),"frame":{"tangent":f["tangent"],"up":f["up"],"inside":f["inside"],"outside":f["outside"],"orientation_determinant":1.0},"required":required})
    return out


def theoretical_volume_m3(centerline: Mapping[str, Any], part: PartSpec) -> float:
    length_m = float(centerline["length_m"])
    if centerline["kind"] == "LINE": return part.area_m2 * length_m
    radius = float(centerline["arc_radius_m"]); theta = math.radians(abs(float(centerline["arc_sweep_deg"])))
    turn_sign = 1.0 if float(centerline["arc_sweep_deg"]) > 0.0 else -1.0
    return (part.y_max-part.y_min) * (part.z_max-part.z_min) * (radius-turn_sign*part.average_z_m) * theta
