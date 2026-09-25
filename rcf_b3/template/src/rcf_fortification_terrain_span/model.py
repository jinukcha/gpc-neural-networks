from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

FIXTURE_SCHEMA = "royal-capital.fortification.terrain-span-fixture/1"
PLAN_SCHEMA = "royal-capital.fortification.terrain-span-plan/1"
FOUNDATION_SCHEMA = "royal-capital.fortification.foundation-interface/1"
CONTACT_SCHEMA = "royal-capital.fortification.terrain-contact-evidence/1"
GAP_SCHEMA = "royal-capital.fortification.terrain-gap-evidence/1"
GRADE_SCHEMA = "royal-capital.fortification.terrain-grade-evidence/1"
MESH_SCHEMA = "royal-capital.fortification.terrain-span-indexed-mesh/1"
PARTS_SCHEMA = "royal-capital.fortification.terrain-span-semantic-parts/1"
SOCKETS_SCHEMA = "royal-capital.fortification.terrain-span-sockets/1"
STORED_SCHEMA = "royal-capital.fortification.terrain-span-stored-copies/1"
RECEIPT_SCHEMA = "royal-capital.fortification.terrain-span-receipt/1"
RESULT_SCHEMA = "royal-capital.fortification.terrain-span-result/1"
PROJECT_FRAME = "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"
PART_ORDER = ("foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet")
LATERAL_ORDER = ("outside", "center", "inside")

EXPECTED_RUNTIME = {
    "build123d_version": "0.13.1.dev12+ge22d34dae",
    "ocp_version": "8.0.1.0.0",
    "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466",
    "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f",
}


class TerrainSpanFamily(StrEnum):
    TERRAIN_STEPPED = "TERRAIN_STEPPED"
    RETAINING = "RETAINING"


class TerrainFailure(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_FRAME = "INVALID_FRAME"
    GRADE_OUT_OF_DOMAIN = "GRADE_OUT_OF_DOMAIN"
    STEP_OUT_OF_DOMAIN = "STEP_OUT_OF_DOMAIN"
    FOUNDATION_DEPTH_OUT_OF_DOMAIN = "FOUNDATION_DEPTH_OUT_OF_DOMAIN"
    FOUNDATION_CONTACT_UNSATISFIED = "FOUNDATION_CONTACT_UNSATISFIED"
    UNSUPPORTED_GAP_EXCEEDED = "UNSUPPORTED_GAP_EXCEEDED"
    TERRAIN_PENETRATION_EXCEEDED = "TERRAIN_PENETRATION_EXCEEDED"
    RETAINING_HEIGHT_EXCEEDED = "RETAINING_HEIGHT_EXCEEDED"
    TERRAIN_SAMPLE_ORDER_INVALID = "TERRAIN_SAMPLE_ORDER_INVALID"
    GEOMETRY_BUDGET_EXCEEDED = "GEOMETRY_BUDGET_EXCEEDED"
    RUNTIME_MISMATCH = "RUNTIME_MISMATCH"
    PROVIDER_EXECUTION_FAILED = "PROVIDER_EXECUTION_FAILED"
    PUBLISH_ABORTED = "PUBLISH_ABORTED"


class TerrainSpanError(ValueError):
    def __init__(self, code: TerrainFailure | str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = message


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_ref(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, f"{name} must be finite")
    return result


def vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, f"{name} must contain three numbers")
    return tuple(number(item, f"{name}[{index}]") for index, item in enumerate(value))


def add(a, b): return tuple(a[i] + b[i] for i in range(3))
def sub(a, b): return tuple(a[i] - b[i] for i in range(3))
def mul(a, scalar): return tuple(a[i] * scalar for i in range(3))
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(value): return math.sqrt(dot(value, value))
def normalize(value):
    length = norm(value)
    if length <= 1e-12:
        raise TerrainSpanError(TerrainFailure.INVALID_FRAME, "zero-length vector")
    return tuple(component / length for component in value)


def rounded(value: float, digits: int = 9) -> float:
    result = round(float(value), digits)
    return 0.0 if result == -0.0 else result


def rounded_vec(value: Sequence[float], digits: int = 9) -> list[float]:
    return [rounded(component, digits) for component in value]


@dataclass(frozen=True)
class Band:
    part_id: str
    semantic_role: str
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    material_slot: str


STANDARD_BANDS = (
    Band("foundation", "FOUNDATION", -2.0, 0.0, -4.0, 4.0, "stone_foundation"),
    Band("wall_body", "WALL_BODY", 0.0, 12.0, -3.0, 3.0, "stone_wall"),
    Band("wall_walk", "WALL_WALK", 12.0, 12.6, -3.2, 3.2, "stone_walk"),
    Band("inner_parapet", "INNER_PARAPET", 12.6, 14.4, 2.4, 3.2, "stone_parapet"),
    Band("outer_parapet", "OUTER_PARAPET", 12.6, 14.4, -3.2, -2.4, "stone_parapet"),
)


def _strict(value: Mapping[str, Any], required: set[str], name: str) -> None:
    if set(value) != required:
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, f"{name} keys mismatch missing={sorted(required-set(value))} extra={sorted(set(value)-required)}")


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "span_id", "span_family", "frame", "inside_side", "outside_side", "plan_centerline", "terrain", "foundation", "sampling", "tolerances", "budget", "runtime"}
    _strict(value, required, "fixture")
    if value["schema"] != FIXTURE_SCHEMA or value["frame"] != PROJECT_FRAME:
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "schema or project frame mismatch")
    if not isinstance(value["span_id"], str) or not value["span_id"].strip():
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "span_id required")
    try:
        family = TerrainSpanFamily(str(value["span_family"]))
    except ValueError as exc:
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "unsupported terrain span family") from exc
    if value["inside_side"] != "LEFT_OF_TRAVEL" or value["outside_side"] != "RIGHT_OF_TRAVEL":
        raise TerrainSpanError(TerrainFailure.INVALID_FRAME, "inside/outside convention mismatch")
    if value["runtime"] != EXPECTED_RUNTIME:
        raise TerrainSpanError(TerrainFailure.RUNTIME_MISMATCH, f"runtime mismatch {value['runtime']}")

    centerline = value["plan_centerline"]
    if not isinstance(centerline, Mapping):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "plan_centerline required")
    _strict(centerline, {"kind", "start_m", "end_m"}, "plan_centerline")
    if centerline["kind"] != "LINE":
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "R0B-CP3 pilot supports canonical LINE plan centerline")
    start = vector3(centerline["start_m"], "plan_centerline.start_m")
    end = vector3(centerline["end_m"], "plan_centerline.end_m")
    if abs(start[1] - end[1]) > 1e-9:
        raise TerrainSpanError(TerrainFailure.INVALID_FRAME, "plan centerline must remain horizontal; terrain is a separate field")
    delta = sub(end, start)
    horizontal_length = math.hypot(delta[0], delta[2])
    if horizontal_length <= 1e-6 or horizontal_length > 60.0:
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "plan centerline length outside (0,60]")

    sampling = value["sampling"]
    _strict(sampling, {"curve_sample_tolerance_m", "max_turn_deg", "max_segments", "round_digits"}, "sampling")
    if not (0.01 <= number(sampling["curve_sample_tolerance_m"], "sampling.curve_sample_tolerance_m") <= 0.25):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "curve sample tolerance outside domain")
    if not (1.0 <= number(sampling["max_turn_deg"], "sampling.max_turn_deg") <= 35.0):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "max turn outside domain")
    if not isinstance(sampling["max_segments"], int) or not (1 <= sampling["max_segments"] <= 4096):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "max_segments outside domain")
    if not isinstance(sampling["round_digits"], int) or not (6 <= sampling["round_digits"] <= 12):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "round_digits outside domain")

    tolerances = value["tolerances"]
    _strict(tolerances, {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad", "mesh_round_digits"}, "tolerances")
    if not (1e-9 <= number(tolerances["linear_m"], "linear_m") <= 1e-3):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "linear tolerance outside domain")
    if not (1e-9 <= number(tolerances["angular_rad"], "angular_rad") <= 1e-3):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "angular tolerance outside domain")
    if number(tolerances["tessellation_linear_m"], "tessellation_linear_m") != 0.05 or number(tolerances["tessellation_angular_rad"], "tessellation_angular_rad") != 0.1 or int(tolerances["mesh_round_digits"]) != 9:
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "fixed tessellation contract mismatch")

    terrain = value["terrain"]
    _strict(terrain, {"interpolation", "samples"}, "terrain")
    if terrain["interpolation"] != "LINEAR_STATION_PROFILE":
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "unsupported terrain interpolation")
    samples = terrain["samples"]
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or len(samples) < 2:
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "at least two terrain samples required")
    normalized_samples = []
    previous = None
    for index, row in enumerate(samples):
        if not isinstance(row, Mapping):
            raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, f"terrain.samples[{index}] must be object")
        _strict(row, {"station_m", "outside_elevation_m", "center_elevation_m", "inside_elevation_m"}, f"terrain.samples[{index}]")
        station = number(row["station_m"], f"terrain.samples[{index}].station_m")
        if previous is not None and station <= previous:
            raise TerrainSpanError(TerrainFailure.TERRAIN_SAMPLE_ORDER_INVALID, "terrain stations must be strictly increasing")
        previous = station
        normalized_samples.append({
            "station_m": station,
            "outside_elevation_m": number(row["outside_elevation_m"], "outside_elevation_m"),
            "center_elevation_m": number(row["center_elevation_m"], "center_elevation_m"),
            "inside_elevation_m": number(row["inside_elevation_m"], "inside_elevation_m"),
        })
    linear_tolerance = float(tolerances["linear_m"])
    if abs(normalized_samples[0]["station_m"]) > linear_tolerance or abs(normalized_samples[-1]["station_m"] - horizontal_length) > linear_tolerance:
        raise TerrainSpanError(TerrainFailure.TERRAIN_SAMPLE_ORDER_INVALID, "terrain samples must cover the exact centerline interval")

    foundation = value["foundation"]
    _strict(foundation, {"depth_m", "step_height_m", "max_step_height_m", "wall_base_clearance_m", "fixed_base_elevation_m", "min_embed_m", "max_embed_m", "max_gap_m", "max_penetration_m", "retaining_side", "toe_extension_m", "heel_extension_m", "max_retained_height_m"}, "foundation")
    depth = number(foundation["depth_m"], "foundation.depth_m")
    if not (1.0 <= depth <= 8.0):
        raise TerrainSpanError(TerrainFailure.FOUNDATION_DEPTH_OUT_OF_DOMAIN, "foundation depth outside [1,8]")
    step_height = number(foundation["step_height_m"], "foundation.step_height_m")
    max_step = number(foundation["max_step_height_m"], "foundation.max_step_height_m")
    if not (0.25 <= step_height <= 2.5 and step_height <= max_step <= 2.5):
        raise TerrainSpanError(TerrainFailure.STEP_OUT_OF_DOMAIN, "step height contract outside domain")
    clearance = number(foundation["wall_base_clearance_m"], "foundation.wall_base_clearance_m")
    fixed_base = foundation["fixed_base_elevation_m"]
    if fixed_base is not None:
        fixed_base = number(fixed_base, "foundation.fixed_base_elevation_m")
    min_embed = number(foundation["min_embed_m"], "foundation.min_embed_m")
    max_embed = number(foundation["max_embed_m"], "foundation.max_embed_m")
    max_gap = number(foundation["max_gap_m"], "foundation.max_gap_m")
    max_penetration = number(foundation["max_penetration_m"], "foundation.max_penetration_m")
    if not (0.0 <= clearance <= 2.5 and 0.0 <= min_embed < max_embed <= 8.0 and 0.0 <= max_gap <= 0.5 and 0.0 <= max_penetration <= 0.5):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "foundation clearance/embed/gap/penetration contract invalid")
    retaining_side = str(foundation["retaining_side"])
    toe = number(foundation["toe_extension_m"], "foundation.toe_extension_m")
    heel = number(foundation["heel_extension_m"], "foundation.heel_extension_m")
    max_retained = number(foundation["max_retained_height_m"], "foundation.max_retained_height_m")
    if not (0.0 <= toe <= 4.0 and 0.0 <= heel <= 6.0 and 0.0 <= max_retained <= 8.0):
        raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "retaining extension/height contract invalid")
    if family is TerrainSpanFamily.TERRAIN_STEPPED:
        if fixed_base is not None or retaining_side != "NONE" or toe != 0.0 or heel != 0.0:
            raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "terrain-stepped fixture must derive steps and use no retaining extension")
    else:
        if fixed_base is None or retaining_side not in {"INSIDE", "OUTSIDE"}:
            raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, "retaining fixture requires fixed base and retaining side")

    budget = value["budget"]
    _strict(budget, {"max_segments", "max_components", "max_vertices", "max_triangles", "max_artifact_bytes"}, "budget")
    for key, item in budget.items():
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, f"budget.{key} must be positive integer")

    result = json.loads(json.dumps(value))
    result["span_family"] = family.value
    result["terrain"]["samples"] = normalized_samples
    result["foundation"]["fixed_base_elevation_m"] = fixed_base
    return result


def _ceiling_step(value: float, step: float) -> float:
    return math.ceil((value - 1e-12) / step) * step


def _station_point(start: Sequence[float], tangent: Sequence[float], station: float, elevation: float) -> tuple[float, float, float]:
    return (start[0] + tangent[0] * station, elevation, start[2] + tangent[2] * station)


def _segment_for_station(segments: Sequence[Mapping[str, Any]], station: float, total: float) -> Mapping[str, Any]:
    if abs(station - total) <= 1e-9:
        return segments[-1]
    for segment in segments:
        if float(segment["station_start_m"]) - 1e-9 <= station < float(segment["station_end_m"]) - 1e-9:
            return segment
    raise TerrainSpanError(TerrainFailure.INVALID_REQUEST, f"station {station} not covered by foundation segments")


def derive_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = validate_fixture(value)
    start = vector3(spec["plan_centerline"]["start_m"], "start")
    end = vector3(spec["plan_centerline"]["end_m"], "end")
    direction = sub(end, start)
    total = math.hypot(direction[0], direction[2])
    tangent = normalize((direction[0], 0.0, direction[2]))
    up = (0.0, 1.0, 0.0)
    inside = normalize(cross(tangent, up))
    outside = mul(inside, -1.0)
    determinant = dot(cross(tangent, up), inside)
    if abs(determinant - 1.0) > 1e-9:
        raise TerrainSpanError(TerrainFailure.INVALID_FRAME, "local frame determinant mismatch")
    samples = spec["terrain"]["samples"]
    max_grade = 0.0
    grade_segments = []
    for index in range(len(samples) - 1):
        a, b = samples[index], samples[index + 1]
        run = b["station_m"] - a["station_m"]
        rise = b["center_elevation_m"] - a["center_elevation_m"]
        grade = abs(rise) / run
        max_grade = max(max_grade, grade)
        grade_segments.append({"segment_index": index, "station_start_m": rounded(a["station_m"]), "station_end_m": rounded(b["station_m"]), "run_m": rounded(run), "rise_m": rounded(rise), "absolute_grade": rounded(grade, 12), "limit": 0.12, "status": "PASS" if grade <= 0.12 + 1e-12 else "FAIL"})
    if max_grade > 0.12 + 1e-12:
        raise TerrainSpanError(TerrainFailure.GRADE_OUT_OF_DOMAIN, f"maximum centerline terrain grade {max_grade} exceeds 0.12")

    foundation = spec["foundation"]
    segments = []
    if spec["span_family"] == TerrainSpanFamily.TERRAIN_STEPPED.value:
        for index in range(len(samples) - 1):
            a, b = samples[index], samples[index + 1]
            terrain_max = max(a[f"{side}_elevation_m"] for side in LATERAL_ORDER)
            terrain_max = max(terrain_max, *(b[f"{side}_elevation_m"] for side in LATERAL_ORDER))
            base = _ceiling_step(terrain_max + foundation["wall_base_clearance_m"], foundation["step_height_m"])
            segments.append({"segment_id": f"step-{index:03d}", "segment_index": index, "station_start_m": rounded(a["station_m"]), "station_end_m": rounded(b["station_m"]), "length_m": rounded(b["station_m"] - a["station_m"]), "base_elevation_m": rounded(base), "foundation_bottom_elevation_m": rounded(base - foundation["depth_m"]), "interface_class": "EMBEDDED_TERRAIN_STEP"})
        for previous, current in zip(segments, segments[1:]):
            delta = abs(current["base_elevation_m"] - previous["base_elevation_m"])
            if delta > foundation["max_step_height_m"] + 1e-9:
                raise TerrainSpanError(TerrainFailure.STEP_OUT_OF_DOMAIN, f"foundation step delta {delta} exceeds {foundation['max_step_height_m']}")
    else:
        base = float(foundation["fixed_base_elevation_m"])
        segments.append({"segment_id": "retaining-000", "segment_index": 0, "station_start_m": 0.0, "station_end_m": rounded(total), "length_m": rounded(total), "base_elevation_m": rounded(base), "foundation_bottom_elevation_m": rounded(base - foundation["depth_m"]), "interface_class": "RETAINING_EMBEDDED"})
    if len(segments) > spec["budget"]["max_segments"]:
        raise TerrainSpanError(TerrainFailure.GEOMETRY_BUDGET_EXCEEDED, "derived terrain segments exceed budget")
    if len(segments) * len(PART_ORDER) > spec["budget"]["max_components"]:
        raise TerrainSpanError(TerrainFailure.GEOMETRY_BUDGET_EXCEEDED, "derived component count exceeds budget")

    contacts = []
    max_gap = 0.0
    max_penetration = 0.0
    min_embed = math.inf
    max_embed = -math.inf
    max_retained_height = 0.0
    for sample_index, sample in enumerate(samples):
        segment = _segment_for_station(segments, float(sample["station_m"]), total)
        bottom = float(segment["foundation_bottom_elevation_m"])
        top = float(segment["base_elevation_m"])
        retained_height = abs(float(sample["inside_elevation_m"]) - float(sample["outside_elevation_m"]))
        max_retained_height = max(max_retained_height, retained_height)
        for side in LATERAL_ORDER:
            terrain_elevation = float(sample[f"{side}_elevation_m"])
            gap = max(0.0, bottom - terrain_elevation)
            penetration = max(0.0, terrain_elevation - top)
            embed = terrain_elevation - bottom
            max_gap = max(max_gap, gap)
            max_penetration = max(max_penetration, penetration)
            min_embed = min(min_embed, embed)
            max_embed = max(max_embed, embed)
            status = "PASS" if gap <= foundation["max_gap_m"] + 1e-12 and penetration <= foundation["max_penetration_m"] + 1e-12 and foundation["min_embed_m"] - 1e-12 <= embed <= foundation["max_embed_m"] + 1e-12 else "FAIL"
            contacts.append({"contact_id": f"contact-{sample_index:03d}-{side}", "sample_index": sample_index, "segment_id": segment["segment_id"], "station_m": rounded(sample["station_m"]), "lateral_role": side.upper(), "terrain_elevation_m": rounded(terrain_elevation), "foundation_bottom_elevation_m": rounded(bottom), "foundation_top_elevation_m": rounded(top), "embed_depth_m": rounded(embed), "unsupported_gap_m": rounded(gap), "terrain_penetration_m": rounded(penetration), "status": status})
    if max_gap > foundation["max_gap_m"] + 1e-12:
        raise TerrainSpanError(TerrainFailure.UNSUPPORTED_GAP_EXCEEDED, f"unsupported gap {max_gap} exceeds {foundation['max_gap_m']}")
    if max_penetration > foundation["max_penetration_m"] + 1e-12:
        raise TerrainSpanError(TerrainFailure.TERRAIN_PENETRATION_EXCEEDED, f"terrain penetration {max_penetration} exceeds {foundation['max_penetration_m']}")
    if min_embed < foundation["min_embed_m"] - 1e-12 or max_embed > foundation["max_embed_m"] + 1e-12 or any(row["status"] != "PASS" for row in contacts):
        raise TerrainSpanError(TerrainFailure.FOUNDATION_CONTACT_UNSATISFIED, f"embed range [{min_embed},{max_embed}] outside [{foundation['min_embed_m']},{foundation['max_embed_m']}]")
    if spec["span_family"] == TerrainSpanFamily.RETAINING.value and max_retained_height > foundation["max_retained_height_m"] + 1e-12:
        raise TerrainSpanError(TerrainFailure.RETAINING_HEIGHT_EXCEEDED, f"retained height {max_retained_height} exceeds {foundation['max_retained_height_m']}")

    toe = float(foundation["toe_extension_m"])
    heel = float(foundation["heel_extension_m"])
    retaining_side = foundation["retaining_side"]
    foundation_z_min, foundation_z_max = -4.0, 4.0
    if retaining_side == "INSIDE":
        foundation_z_min -= toe
        foundation_z_max += heel
    elif retaining_side == "OUTSIDE":
        foundation_z_min -= heel
        foundation_z_max += toe

    frames = []
    for index, segment in enumerate(segments):
        station0, station1 = float(segment["station_start_m"]), float(segment["station_end_m"])
        base = float(segment["base_elevation_m"])
        frames.append({"segment_id": segment["segment_id"], "segment_index": index, "start": {"position_m": rounded_vec(_station_point(start, tangent, station0, base)), "tangent": rounded_vec(tangent), "up": [0.0, 1.0, 0.0], "inside": rounded_vec(inside), "outside": rounded_vec(outside), "orientation_determinant": 1.0}, "end": {"position_m": rounded_vec(_station_point(start, tangent, station1, base)), "tangent": rounded_vec(tangent), "up": [0.0, 1.0, 0.0], "inside": rounded_vec(inside), "outside": rounded_vec(outside), "orientation_determinant": 1.0}})

    canonical_centerline = {"schema": "royal-capital.fortification.canonical-plan-centerline/1", "kind": "LINE", "start_m": rounded_vec(start), "end_m": rounded_vec(end), "length_m": rounded(total), "tangent": rounded_vec(tangent), "inside": rounded_vec(inside), "outside": rounded_vec(outside), "orientation_determinant": 1.0, "canonical_digest": ""}
    canonical_centerline["canonical_digest"] = sha256_ref(canonical_bytes({key: value for key, value in canonical_centerline.items() if key != "canonical_digest"}))
    grade_doc = {"schema": GRADE_SCHEMA, "span_id": spec["span_id"], "maximum_allowed_grade": 0.12, "maximum_observed_grade": rounded(max_grade, 12), "segments": grade_segments, "status": "PASS"}
    contact_doc = {"schema": CONTACT_SCHEMA, "span_id": spec["span_id"], "sample_count": len(contacts), "contacts": contacts, "summary": {"minimum_embed_depth_m": rounded(min_embed), "maximum_embed_depth_m": rounded(max_embed), "maximum_unsupported_gap_m": rounded(max_gap), "maximum_terrain_penetration_m": rounded(max_penetration), "maximum_retained_height_m": rounded(max_retained_height), "status": "PASS"}}
    gap_doc = {"schema": GAP_SCHEMA, "span_id": spec["span_id"], "maximum_allowed_gap_m": foundation["max_gap_m"], "maximum_observed_gap_m": rounded(max_gap), "maximum_allowed_penetration_m": foundation["max_penetration_m"], "maximum_observed_penetration_m": rounded(max_penetration), "status": "PASS"}
    foundation_doc = {"schema": FOUNDATION_SCHEMA, "span_id": spec["span_id"], "span_family": spec["span_family"], "terrain_mutation_requested": False, "foundation_depth_m": foundation["depth_m"], "foundation_offsets_m": {"outside": rounded(foundation_z_min), "inside": rounded(foundation_z_max)}, "retaining_side": retaining_side, "toe_extension_m": toe, "heel_extension_m": heel, "segments": segments, "contact_evidence_ref": "contact-evidence.json", "gap_evidence_ref": "gap-evidence.json", "grade_evidence_ref": "grade-evidence.json", "status": "PASS"}
    contract = {"spec": spec, "canonical_centerline": canonical_centerline, "segment_frames": {"schema": "royal-capital.fortification.terrain-segment-local-frames/1", "span_id": spec["span_id"], "frames": frames}, "foundation_interface": foundation_doc, "contact_evidence": contact_doc, "gap_evidence": gap_doc, "grade_evidence": grade_doc, "foundation_z_min": foundation_z_min, "foundation_z_max": foundation_z_max, "tangent": tangent, "inside": inside, "outside": outside, "start": start, "total_length_m": total}
    return contract


def bands_for_contract(contract: Mapping[str, Any]) -> tuple[Band, ...]:
    depth = float(contract["spec"]["foundation"]["depth_m"])
    result = []
    for band in STANDARD_BANDS:
        if band.part_id == "foundation":
            result.append(Band(band.part_id, band.semantic_role, -depth, 0.0, float(contract["foundation_z_min"]), float(contract["foundation_z_max"]), "stone_foundation_retaining" if contract["spec"]["span_family"] == TerrainSpanFamily.RETAINING.value else band.material_slot))
        else:
            result.append(band)
    return tuple(result)


def provider_request(contract: Mapping[str, Any], segment: Mapping[str, Any], frame: Mapping[str, Any], band: Band) -> dict[str, Any]:
    base_position = frame["start"]["position_m"]
    return {
        "schema": "royal-capital.fortification.cad-provider-request/1",
        "request_id": f"cad-request/{contract['spec']['span_id']}/{segment['segment_id']}/{band.part_id}@1",
        "units": "METER",
        "frame": PROJECT_FRAME,
        "runtime": dict(contract["spec"]["runtime"]),
        "tolerances": {key: contract["spec"]["tolerances"][key] for key in ("linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad")},
        "budget": {"max_profile_points": 16, "max_vertices": contract["spec"]["budget"]["max_vertices"], "max_triangles": contract["spec"]["budget"]["max_triangles"], "max_artifact_bytes": contract["spec"]["budget"]["max_artifact_bytes"]},
        "operation": {"kind": "PROFILE_EXTRUSION", "plane": {"origin_m": base_position, "x_axis": [0.0, 1.0, 0.0], "y_axis": frame["start"]["inside"]}, "outer_loop_m": [[band.y_min, band.z_min], [band.y_max, band.z_min], [band.y_max, band.z_max], [band.y_min, band.z_max]], "distance_m": segment["length_m"]},
        "outputs": {"indexed_mesh": True, "stored_copies": ["STEP", "BREP"]},
    }


def socket_plan(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = contract["segment_frames"]["frames"]
    first, last = frames[0]["start"], frames[-1]["end"]
    foundation_depth = float(contract["spec"]["foundation"]["depth_m"])
    def shifted(frame: Mapping[str, Any], y: float) -> list[float]:
        return rounded_vec(add(frame["position_m"], (0.0, y, 0.0)))
    rows = [
        ("span_start", "SPAN_JOIN", first, 6.0), ("span_end", "SPAN_JOIN", last, 6.0),
        ("wall_walk_start", "WALL_WALK_CONTINUATION", first, 12.3), ("wall_walk_end", "WALL_WALK_CONTINUATION", last, 12.3),
        ("foundation_start", "FOUNDATION_INTERFACE", first, -foundation_depth * 0.5), ("foundation_end", "FOUNDATION_INTERFACE", last, -foundation_depth * 0.5),
        ("tower_start", "TOWER_JOIN", first, 6.0), ("tower_end", "TOWER_JOIN", last, 6.0),
        ("utility_inside_01", "UTILITY_INSIDE", frames[0]["start"], 6.0), ("utility_inside_02", "UTILITY_INSIDE", frames[-1]["end"], 6.0),
    ]
    result = []
    for socket_id, role, frame, y in rows:
        result.append({"socket_id": socket_id, "role": role, "position_m": shifted(frame, y), "frame": {"tangent": frame["tangent"], "up": frame["up"], "inside": frame["inside"], "outside": frame["outside"], "orientation_determinant": frame["orientation_determinant"]}, "required": role != "UTILITY_INSIDE"})
    return result
