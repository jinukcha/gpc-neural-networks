from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from rcf_fortification_path_span.model import (
    EXPECTED_RUNTIME,
    PART_ORDER,
    PART_SPECS,
    PROJECT_FRAME,
)

TERRAIN_SPAN_SCHEMA = "royal-capital.fortification.terrain-wall-span/1"
TERRAIN_PROFILE_SCHEMA = "royal-capital.fortification.canonical-terrain-profile/1"
PLAN_SCHEMA = "royal-capital.fortification.terrain-wall-span-plan/1"
FOUNDATION_INTERFACE_SCHEMA = "royal-capital.fortification.foundation-interface/1"
CONTACT_EVIDENCE_SCHEMA = "royal-capital.fortification.contact-gap-evidence/1"
GRADE_EVIDENCE_SCHEMA = "royal-capital.fortification.grade-evidence/1"
SOCKETS_SCHEMA = "royal-capital.fortification.terrain-span-sockets/1"
PARTS_SCHEMA = "royal-capital.fortification.terrain-span-semantic-parts/1"
MESH_SCHEMA = "royal-capital.fortification.terrain-span-indexed-mesh/1"
STORED_SCHEMA = "royal-capital.fortification.terrain-span-stored-copies/1"
TESSELLATION_SCHEMA = "royal-capital.fortification.terrain-span-tessellation/1"
RECEIPT_SCHEMA = "royal-capital.fortification.terrain-span-receipt/1"
RESULT_SCHEMA = "royal-capital.fortification.terrain-span-result/1"


class TerrainFailureCode(StrEnum):
    INVALID_TERRAIN_PROFILE = "INVALID_TERRAIN_PROFILE"
    STALE_INPUT_REFERENCE = "STALE_INPUT_REFERENCE"
    GRADE_OUT_OF_DOMAIN = "GRADE_OUT_OF_DOMAIN"
    FOUNDATION_GAP_OUT_OF_DOMAIN = "FOUNDATION_GAP_OUT_OF_DOMAIN"
    FOUNDATION_EMBEDMENT_OUT_OF_DOMAIN = "FOUNDATION_EMBEDMENT_OUT_OF_DOMAIN"
    RETAINING_HEIGHT_OUT_OF_DOMAIN = "RETAINING_HEIGHT_OUT_OF_DOMAIN"
    GEOMETRY_BUDGET_EXCEEDED = "GEOMETRY_BUDGET_EXCEEDED"
    CAD_RUNTIME_VERSION_MISMATCH = "CAD_RUNTIME_VERSION_MISMATCH"
    SOCKET_GEOMETRY_MISMATCH = "SOCKET_GEOMETRY_MISMATCH"


class TerrainContractError(ValueError):
    def __init__(self, code: TerrainFailureCode | str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = message


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_ref(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _f(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, f"{name} must be finite")
    return result


def _i(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, f"{name} must be an integer")
    return value


def _v3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, f"{name} must contain exactly three numbers")
    return tuple(_f(component, f"{name}[{index}]") for index, component in enumerate(value))


def _round(value: float, digits: int = 9) -> float:
    result = round(float(value), digits)
    return 0.0 if result == -0.0 else result


def _rv(value: Sequence[float], digits: int = 9) -> list[float]:
    return [_round(component, digits) for component in value]


def add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a[index]) + float(b[index]) for index in range(3))


def sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a[index]) - float(b[index]) for index in range(3))


def mul(a: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(a[index]) * float(scalar) for index in range(3))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def length(value: Sequence[float]) -> float:
    return math.sqrt(dot(value, value))


def normalize(value: Sequence[float]) -> tuple[float, float, float]:
    magnitude = length(value)
    if magnitude <= 1e-15:
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "zero-length vector")
    return tuple(float(component) / magnitude for component in value)


def _strict_keys(value: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise TerrainContractError(
            TerrainFailureCode.INVALID_TERRAIN_PROFILE,
            f"{name} keys mismatch missing={sorted(missing)} extra={sorted(extra)}",
        )


def axis_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    axis = spec["axis"]
    start = _v3(axis["start_m"], "axis.start_m")
    end = _v3(axis["end_m"], "axis.end_m")
    delta = sub(end, start)
    horizontal = (delta[0], 0.0, delta[2])
    axis_length = length(horizontal)
    if axis_length <= float(spec["tolerances"]["linear_m"]):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "terrain span axis is zero length")
    if abs(delta[1]) > float(spec["tolerances"]["linear_m"]):
        raise TerrainContractError(TerrainFailureCode.GRADE_OUT_OF_DOMAIN, "axis itself must remain horizontal; terrain elevation is carried by the foundation interface")
    if axis_length > 60.0:
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "terrain span exceeds 60m bounded module limit")
    tangent = normalize(horizontal)
    up = (0.0, 1.0, 0.0)
    inside = normalize(cross(tangent, up))
    outside = mul(inside, -1.0)
    determinant = dot(cross(tangent, up), inside)
    if abs(determinant - 1.0) > 1e-9:
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "axis frame determinant mismatch")
    return {
        "start_m": _rv(start),
        "end_m": _rv(end),
        "length_m": _round(axis_length),
        "tangent": _rv(tangent),
        "up": [0.0, 1.0, 0.0],
        "inside": _rv(inside),
        "outside": _rv(outside),
        "orientation_determinant": 1.0,
    }


def terrain_profile_payload(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": TERRAIN_PROFILE_SCHEMA,
        "family": spec["family"],
        "axis": spec["axis"],
        "terrain": spec["terrain"],
    }


def terrain_profile_digest(spec: Mapping[str, Any]) -> str:
    return sha256_ref(canonical_json_bytes(terrain_profile_payload(spec)))


def _validate_common(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "span_id", "family", "profile_id", "frame", "inside_side", "outside_side",
        "axis", "terrain_source", "terrain", "foundation", "grade", "tolerances", "budget", "runtime",
    }
    _strict_keys(value, required, "fixture")
    if value["schema"] != TERRAIN_SPAN_SCHEMA:
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "unsupported terrain-span schema")
    if not isinstance(value["span_id"], str) or not value["span_id"].strip():
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "span_id required")
    if not isinstance(value["profile_id"], str) or not value["profile_id"].strip():
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "profile_id required")
    if value["family"] not in {"TERRAIN_STEPPED", "RETAINING"}:
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "family must be TERRAIN_STEPPED or RETAINING")
    if value["frame"] != PROJECT_FRAME:
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "project frame mismatch")
    if value["inside_side"] != "LEFT_OF_TRAVEL" or value["outside_side"] != "RIGHT_OF_TRAVEL":
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "inside/outside convention mismatch")
    if value["runtime"] != EXPECTED_RUNTIME:
        raise TerrainContractError(TerrainFailureCode.CAD_RUNTIME_VERSION_MISMATCH, f"runtime mismatch: {value['runtime']}")

    axis = value["axis"]
    if not isinstance(axis, Mapping):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "axis must be an object")
    _strict_keys(axis, {"start_m", "end_m"}, "axis")

    source = value["terrain_source"]
    if not isinstance(source, Mapping):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "terrain_source must be an object")
    _strict_keys(source, {"source_id", "revision", "profile_digest"}, "terrain_source")
    if not all(isinstance(source[key], str) and source[key] for key in ("source_id", "revision", "profile_digest")):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "terrain_source fields must be non-empty strings")

    foundation = value["foundation"]
    if not isinstance(foundation, Mapping):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "foundation must be an object")
    _strict_keys(
        foundation,
        {"embedment_depth_m", "max_embedment_depth_m", "max_gap_m", "contact_sample_spacing_m", "terrain_contact_offset_m"},
        "foundation",
    )
    embedment = _f(foundation["embedment_depth_m"], "foundation.embedment_depth_m")
    maximum_embedment = _f(foundation["max_embedment_depth_m"], "foundation.max_embedment_depth_m")
    maximum_gap = _f(foundation["max_gap_m"], "foundation.max_gap_m")
    spacing = _f(foundation["contact_sample_spacing_m"], "foundation.contact_sample_spacing_m")
    contact_offset = _f(foundation["terrain_contact_offset_m"], "foundation.terrain_contact_offset_m")
    if not (1.0 <= embedment <= 8.0) or not (embedment <= maximum_embedment <= 8.0):
        raise TerrainContractError(TerrainFailureCode.FOUNDATION_EMBEDMENT_OUT_OF_DOMAIN, "foundation embedment outside [1,8] or exceeds maximum")
    if not (0.0 <= maximum_gap <= 0.25):
        raise TerrainContractError(TerrainFailureCode.FOUNDATION_GAP_OUT_OF_DOMAIN, "max_gap_m outside [0,0.25]")
    if not (0.25 <= spacing <= 10.0):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "contact_sample_spacing_m outside [0.25,10]")
    if abs(contact_offset) > maximum_gap:
        raise TerrainContractError(
            TerrainFailureCode.FOUNDATION_GAP_OUT_OF_DOMAIN,
            f"terrain contact datum offset {contact_offset} exceeds max gap {maximum_gap}",
        )

    grade = value["grade"]
    if not isinstance(grade, Mapping):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "grade must be an object")
    _strict_keys(grade, {"max_centerline_grade", "max_step_height_m"}, "grade")
    maximum_grade = _f(grade["max_centerline_grade"], "grade.max_centerline_grade")
    maximum_step = _f(grade["max_step_height_m"], "grade.max_step_height_m")
    if not (0.0 <= maximum_grade <= 0.12):
        raise TerrainContractError(TerrainFailureCode.GRADE_OUT_OF_DOMAIN, "max_centerline_grade outside [0,0.12]")
    if not (0.25 <= maximum_step <= 2.5):
        raise TerrainContractError(TerrainFailureCode.GRADE_OUT_OF_DOMAIN, "max_step_height_m outside [0.25,2.5]")

    tolerances = value["tolerances"]
    if not isinstance(tolerances, Mapping):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "tolerances must be an object")
    _strict_keys(tolerances, {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad", "mesh_round_digits"}, "tolerances")
    if not (1e-9 <= _f(tolerances["linear_m"], "tolerances.linear_m") <= 1e-3):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "linear tolerance outside admitted domain")
    if not (1e-9 <= _f(tolerances["angular_rad"], "tolerances.angular_rad") <= 1e-3):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "angular tolerance outside admitted domain")
    if not (1e-3 <= _f(tolerances["tessellation_linear_m"], "tolerances.tessellation_linear_m") <= 0.1):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "tessellation linear tolerance outside admitted domain")
    if not (0.01 <= _f(tolerances["tessellation_angular_rad"], "tolerances.tessellation_angular_rad") <= 0.3):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "tessellation angular tolerance outside admitted domain")
    if _i(tolerances["mesh_round_digits"], "tolerances.mesh_round_digits") != 9:
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "mesh_round_digits must be 9")

    budget = value["budget"]
    if not isinstance(budget, Mapping):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "budget must be an object")
    _strict_keys(budget, {"max_units", "max_vertices", "max_triangles", "max_artifact_bytes"}, "budget")
    for key in budget:
        if _i(budget[key], f"budget.{key}") <= 0:
            raise TerrainContractError(TerrainFailureCode.GEOMETRY_BUDGET_EXCEEDED, f"budget.{key} must be positive")

    spec = json.loads(json.dumps(value))
    axis_contract(spec)
    return spec


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = _validate_common(value)
    axis = axis_contract(spec)
    terrain = spec["terrain"]
    if not isinstance(terrain, Mapping):
        raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "terrain must be an object")

    if spec["family"] == "TERRAIN_STEPPED":
        _strict_keys(terrain, {"kind", "terraces"}, "terrain")
        if terrain["kind"] != "STEPPED_TERRACES":
            raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "stepped family requires STEPPED_TERRACES terrain")
        terraces = terrain["terraces"]
        if not isinstance(terraces, Sequence) or isinstance(terraces, (str, bytes)) or not terraces:
            raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "terraces must be a non-empty array")
        expected_start = 0.0
        elevations: list[float] = []
        for index, row in enumerate(terraces):
            if not isinstance(row, Mapping):
                raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, f"terraces[{index}] must be an object")
            _strict_keys(row, {"segment_id", "start_station_m", "end_station_m", "elevation_m"}, f"terraces[{index}]")
            if not isinstance(row["segment_id"], str) or not row["segment_id"].strip():
                raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, f"terraces[{index}].segment_id required")
            start = _f(row["start_station_m"], f"terraces[{index}].start_station_m")
            end = _f(row["end_station_m"], f"terraces[{index}].end_station_m")
            elevation = _f(row["elevation_m"], f"terraces[{index}].elevation_m")
            if abs(start - expected_start) > float(spec["tolerances"]["linear_m"]):
                raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, f"terrace coverage gap/overlap at {start}, expected {expected_start}")
            if end - start <= float(spec["tolerances"]["linear_m"]):
                raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "terrace interval must have positive length")
            expected_start = end
            elevations.append(elevation)
        if abs(expected_start - float(axis["length_m"])) > float(spec["tolerances"]["linear_m"]):
            raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "terraces do not cover the full span")
        step_heights = [abs(elevations[index + 1] - elevations[index]) for index in range(len(elevations) - 1)]
        if step_heights and max(step_heights) > float(spec["grade"]["max_step_height_m"]) + float(spec["tolerances"]["linear_m"]):
            raise TerrainContractError(TerrainFailureCode.GRADE_OUT_OF_DOMAIN, f"step height {max(step_heights)} exceeds contract")
        overall_grade = abs(elevations[-1] - elevations[0]) / float(axis["length_m"])
        if overall_grade > float(spec["grade"]["max_centerline_grade"]) + 1e-12:
            raise TerrainContractError(TerrainFailureCode.GRADE_OUT_OF_DOMAIN, f"overall grade {overall_grade} exceeds contract")
        required_units = len(terraces) * len(PART_ORDER)
    else:
        _strict_keys(
            terrain,
            {"kind", "outside_elevation_m", "inside_elevation_m", "bearing_elevation_m", "outside_toe_width_m", "inside_heel_width_m"},
            "terrain",
        )
        if terrain["kind"] != "RETAINING_CROSS_SLOPE":
            raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "retaining family requires RETAINING_CROSS_SLOPE terrain")
        outside_elevation = _f(terrain["outside_elevation_m"], "terrain.outside_elevation_m")
        inside_elevation = _f(terrain["inside_elevation_m"], "terrain.inside_elevation_m")
        bearing_elevation = _f(terrain["bearing_elevation_m"], "terrain.bearing_elevation_m")
        toe_width = _f(terrain["outside_toe_width_m"], "terrain.outside_toe_width_m")
        heel_width = _f(terrain["inside_heel_width_m"], "terrain.inside_heel_width_m")
        retained_height = inside_elevation - outside_elevation
        if not (0.25 <= retained_height <= 6.0):
            raise TerrainContractError(TerrainFailureCode.RETAINING_HEIGHT_OUT_OF_DOMAIN, f"retained height {retained_height} outside [0.25,6]")
        lower_embedment = outside_elevation - bearing_elevation
        upper_embedment = inside_elevation - bearing_elevation
        if abs(lower_embedment - float(spec["foundation"]["embedment_depth_m"])) > float(spec["tolerances"]["linear_m"]):
            raise TerrainContractError(TerrainFailureCode.FOUNDATION_EMBEDMENT_OUT_OF_DOMAIN, "bearing elevation does not match declared outside embedment")
        if upper_embedment > float(spec["foundation"]["max_embedment_depth_m"]) + float(spec["tolerances"]["linear_m"]):
            raise TerrainContractError(TerrainFailureCode.FOUNDATION_EMBEDMENT_OUT_OF_DOMAIN, "inside embedment exceeds maximum")
        if not (4.0 <= toe_width <= 8.0 and 3.0 <= heel_width <= 8.0):
            raise TerrainContractError(TerrainFailureCode.INVALID_TERRAIN_PROFILE, "retaining toe/heel widths outside admitted domain")
        required_units = len(PART_ORDER)

    if required_units > int(spec["budget"]["max_units"]):
        raise TerrainContractError(TerrainFailureCode.GEOMETRY_BUDGET_EXCEEDED, f"required units {required_units} exceed budget")
    observed_digest = terrain_profile_digest(spec)
    if spec["terrain_source"]["profile_digest"] != observed_digest:
        raise TerrainContractError(
            TerrainFailureCode.STALE_INPUT_REFERENCE,
            f"terrain profile digest mismatch expected={spec['terrain_source']['profile_digest']} observed={observed_digest}",
        )
    return spec


def _point_on_axis(axis: Mapping[str, Any], station_m: float, elevation_m: float) -> tuple[float, float, float]:
    start = axis["start_m"]
    tangent = axis["tangent"]
    up = axis["up"]
    return tuple(float(start[index]) + float(station_m) * float(tangent[index]) + float(elevation_m) * float(up[index]) for index in range(3))


def _standard_loop(part: Any) -> list[list[float]]:
    return [
        [float(part.y_min), float(part.z_min)],
        [float(part.y_max), float(part.z_min)],
        [float(part.y_max), float(part.z_max)],
        [float(part.y_min), float(part.z_max)],
    ]


def construction_units(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = validate_fixture(spec)
    axis = axis_contract(spec)
    units: list[dict[str, Any]] = []
    if spec["family"] == "TERRAIN_STEPPED":
        for segment_index, terrace in enumerate(spec["terrain"]["terraces"]):
            start_station = float(terrace["start_station_m"])
            end_station = float(terrace["end_station_m"])
            elevation = float(terrace["elevation_m"])
            origin = _point_on_axis(axis, start_station, elevation)
            for part in PART_SPECS:
                units.append({
                    "unit_id": f"unit/{terrace['segment_id']}/{part.part_id}",
                    "segment_id": terrace["segment_id"],
                    "segment_index": segment_index,
                    "part_id": part.part_id,
                    "semantic_role": part.semantic_role,
                    "material_slot": part.material_slot,
                    "start_station_m": _round(start_station),
                    "end_station_m": _round(end_station),
                    "terrain_elevation_m": _round(elevation),
                    "plane_origin_m": _rv(origin),
                    "plane_x_axis": axis["up"],
                    "plane_y_axis": axis["inside"],
                    "outer_loop_m": _standard_loop(part),
                    "distance_m": _round(end_station - start_station),
                    "profile_area_m2": _round(part.area_m2),
                })
    else:
        terrain = spec["terrain"]
        outside_elevation = float(terrain["outside_elevation_m"])
        inside_elevation = float(terrain["inside_elevation_m"])
        bearing = float(terrain["bearing_elevation_m"])
        toe = float(terrain["outside_toe_width_m"])
        heel = float(terrain["inside_heel_width_m"])
        foundation_loop = [
            [bearing - outside_elevation, -toe],
            [0.0, -toe],
            [0.0, -4.0],
            [inside_elevation - outside_elevation, -4.0],
            [inside_elevation - outside_elevation, heel],
            [bearing - outside_elevation, heel],
        ]
        for part in PART_SPECS:
            if part.part_id == "foundation":
                elevation = outside_elevation
                loop = foundation_loop
                area = abs(0.5 * sum(
                    loop[index][0] * loop[(index + 1) % len(loop)][1] - loop[(index + 1) % len(loop)][0] * loop[index][1]
                    for index in range(len(loop))
                ))
            else:
                elevation = inside_elevation
                loop = _standard_loop(part)
                area = part.area_m2
            origin = _point_on_axis(axis, 0.0, elevation)
            units.append({
                "unit_id": f"unit/retaining/{part.part_id}",
                "segment_id": "retaining_00",
                "segment_index": 0,
                "part_id": part.part_id,
                "semantic_role": part.semantic_role,
                "material_slot": part.material_slot,
                "start_station_m": 0.0,
                "end_station_m": axis["length_m"],
                "terrain_elevation_m": _round(elevation),
                "plane_origin_m": _rv(origin),
                "plane_x_axis": axis["up"],
                "plane_y_axis": axis["inside"],
                "outer_loop_m": [[_round(point[0]), _round(point[1])] for point in loop],
                "distance_m": axis["length_m"],
                "profile_area_m2": _round(area),
            })
    return units


def provider_request(spec: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "royal-capital.fortification.cad-provider-request/1",
        "request_id": f"cad-request/{spec['span_id']}/{unit['segment_id']}/{unit['part_id']}@1",
        "units": "METER",
        "frame": spec["frame"],
        "runtime": dict(spec["runtime"]),
        "tolerances": {key: spec["tolerances"][key] for key in ("linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad")},
        "budget": {
            "max_profile_points": 64,
            "max_vertices": int(spec["budget"]["max_vertices"]),
            "max_triangles": int(spec["budget"]["max_triangles"]),
            "max_artifact_bytes": int(spec["budget"]["max_artifact_bytes"]),
        },
        "operation": {
            "kind": "PROFILE_EXTRUSION",
            "plane": {
                "origin_m": list(unit["plane_origin_m"]),
                "x_axis": list(unit["plane_x_axis"]),
                "y_axis": list(unit["plane_y_axis"]),
            },
            "outer_loop_m": [list(point) for point in unit["outer_loop_m"]],
            "distance_m": float(unit["distance_m"]),
        },
        "outputs": {"indexed_mesh": True, "stored_copies": ["STEP", "BREP"]},
    }


def canonical_terrain_profile(spec: Mapping[str, Any]) -> dict[str, Any]:
    spec = validate_fixture(spec)
    axis = axis_contract(spec)
    payload = terrain_profile_payload(spec)
    document = {
        **payload,
        "span_id": spec["span_id"],
        "source_id": spec["terrain_source"]["source_id"],
        "source_revision": spec["terrain_source"]["revision"],
        "profile_digest": terrain_profile_digest(spec),
        "axis_contract": axis,
    }
    return document


def foundation_interface(spec: Mapping[str, Any]) -> dict[str, Any]:
    spec = validate_fixture(spec)
    axis = axis_contract(spec)
    offset = float(spec["foundation"]["terrain_contact_offset_m"])
    interfaces: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    if spec["family"] == "TERRAIN_STEPPED":
        terraces = spec["terrain"]["terraces"]
        for index, terrace in enumerate(terraces):
            length_m = float(terrace["end_station_m"]) - float(terrace["start_station_m"])
            elevation = float(terrace["elevation_m"])
            interfaces.append({
                "interface_id": f"foundation-interface/{terrace['segment_id']}",
                "segment_id": terrace["segment_id"],
                "kind": "TERRACE_EMBEDDED_STRIP",
                "station_interval_m": [_round(float(terrace["start_station_m"])), _round(float(terrace["end_station_m"]))],
                "terrain_elevation_m": _round(elevation + offset),
                "foundation_top_elevation_m": _round(elevation),
                "foundation_base_elevation_m": _round(elevation - float(spec["foundation"]["embedment_depth_m"])),
                "embedment_depth_m": _round(float(spec["foundation"]["embedment_depth_m"])),
                "contact_width_m": 8.0,
                "contact_area_m2": _round(length_m * 8.0),
                "unsupported_gap_m": _round(abs(offset)),
                "terrain_mutation": False,
            })
            if index + 1 < len(terraces):
                next_terrace = terraces[index + 1]
                step_height = float(next_terrace["elevation_m"]) - elevation
                transitions.append({
                    "transition_id": f"foundation-step/{index:02d}",
                    "station_m": _round(float(terrace["end_station_m"])),
                    "lower_segment_id": terrace["segment_id"] if step_height >= 0 else next_terrace["segment_id"],
                    "upper_segment_id": next_terrace["segment_id"] if step_height >= 0 else terrace["segment_id"],
                    "signed_step_height_m": _round(step_height),
                    "absolute_step_height_m": _round(abs(step_height)),
                    "wall_walk_transition_required": abs(step_height) > float(spec["tolerances"]["linear_m"]),
                })
        maximum_embedment = float(spec["foundation"]["embedment_depth_m"])
        retained_height = 0.0
    else:
        terrain = spec["terrain"]
        outside = float(terrain["outside_elevation_m"])
        inside = float(terrain["inside_elevation_m"])
        bearing = float(terrain["bearing_elevation_m"])
        toe = float(terrain["outside_toe_width_m"])
        heel = float(terrain["inside_heel_width_m"])
        length_m = float(axis["length_m"])
        interfaces.extend([
            {
                "interface_id": "foundation-interface/base-bearing",
                "segment_id": "retaining_00",
                "kind": "BASE_BEARING_STRIP",
                "station_interval_m": [0.0, _round(length_m)],
                "terrain_elevation_m": _round(bearing + offset),
                "foundation_top_elevation_m": _round(bearing),
                "foundation_base_elevation_m": _round(bearing),
                "embedment_depth_m": 0.0,
                "contact_width_m": _round(toe + heel),
                "contact_area_m2": _round(length_m * (toe + heel)),
                "unsupported_gap_m": _round(abs(offset)),
                "terrain_mutation": False,
            },
            {
                "interface_id": "foundation-interface/outside-toe",
                "segment_id": "retaining_00",
                "kind": "OUTSIDE_TERRAIN_SEAM",
                "station_interval_m": [0.0, _round(length_m)],
                "terrain_elevation_m": _round(outside + offset),
                "foundation_top_elevation_m": _round(outside),
                "foundation_base_elevation_m": _round(bearing),
                "embedment_depth_m": _round(outside - bearing),
                "contact_width_m": _round(max(0.0, toe - 4.0)),
                "contact_area_m2": _round(length_m * max(0.0, toe - 4.0)),
                "unsupported_gap_m": _round(abs(offset)),
                "terrain_mutation": False,
            },
            {
                "interface_id": "foundation-interface/inside-heel",
                "segment_id": "retaining_00",
                "kind": "INSIDE_TERRAIN_SEAM",
                "station_interval_m": [0.0, _round(length_m)],
                "terrain_elevation_m": _round(inside + offset),
                "foundation_top_elevation_m": _round(inside),
                "foundation_base_elevation_m": _round(bearing),
                "embedment_depth_m": _round(inside - bearing),
                "contact_width_m": _round(max(0.0, heel - 3.0)),
                "contact_area_m2": _round(length_m * max(0.0, heel - 3.0)),
                "unsupported_gap_m": _round(abs(offset)),
                "terrain_mutation": False,
            },
        ])
        maximum_embedment = inside - bearing
        retained_height = inside - outside

    maximum_gap = max(float(row["unsupported_gap_m"]) for row in interfaces)
    total_contact_area = sum(float(row["contact_area_m2"]) for row in interfaces)
    status = "PASS" if maximum_gap <= float(spec["foundation"]["max_gap_m"]) + 1e-12 and maximum_embedment <= float(spec["foundation"]["max_embedment_depth_m"]) + 1e-12 else "FAIL"
    return {
        "schema": FOUNDATION_INTERFACE_SCHEMA,
        "span_id": spec["span_id"],
        "family": spec["family"],
        "frame": spec["frame"],
        "axis_contract": axis,
        "terrain_source": dict(spec["terrain_source"]),
        "terrain_profile_digest": terrain_profile_digest(spec),
        "interfaces": interfaces,
        "transitions": transitions,
        "summary": {
            "status": status,
            "interface_count": len(interfaces),
            "transition_count": len(transitions),
            "total_contact_area_m2": _round(total_contact_area),
            "maximum_unsupported_gap_m": _round(maximum_gap),
            "maximum_embedment_depth_m": _round(maximum_embedment),
            "retained_height_m": _round(retained_height),
            "terrain_mutation": False,
        },
    }


def _sample_stations(start: float, end: float, spacing: float) -> list[float]:
    count = max(1, math.ceil((end - start) / spacing))
    return [_round(start + (end - start) * index / count) for index in range(count + 1)]


def contact_evidence(spec: Mapping[str, Any], interface: Mapping[str, Any]) -> dict[str, Any]:
    spec = validate_fixture(spec)
    spacing = float(spec["foundation"]["contact_sample_spacing_m"])
    samples: list[dict[str, Any]] = []
    for row in interface["interfaces"]:
        start, end = map(float, row["station_interval_m"])
        for station in _sample_stations(start, end, spacing):
            samples.append({
                "sample_id": f"contact/{row['interface_id'].split('/')[-1]}/{station:.6f}",
                "interface_id": row["interface_id"],
                "station_m": _round(station),
                "terrain_elevation_m": row["terrain_elevation_m"],
                "foundation_contact_elevation_m": row["foundation_top_elevation_m"],
                "gap_m": _round(abs(float(row["terrain_elevation_m"]) - float(row["foundation_top_elevation_m"]))),
                "contact": abs(float(row["terrain_elevation_m"]) - float(row["foundation_top_elevation_m"])) <= float(spec["foundation"]["max_gap_m"]) + 1e-12,
            })
    maximum_gap = max(float(sample["gap_m"]) for sample in samples)
    contact_count = sum(bool(sample["contact"]) for sample in samples)
    return {
        "schema": CONTACT_EVIDENCE_SCHEMA,
        "span_id": spec["span_id"],
        "family": spec["family"],
        "sampling_spacing_m": _round(spacing),
        "samples": samples,
        "summary": {
            "status": "PASS" if contact_count == len(samples) and maximum_gap <= float(spec["foundation"]["max_gap_m"]) + 1e-12 else "FAIL",
            "sample_count": len(samples),
            "contact_count": contact_count,
            "contact_ratio": _round(contact_count / len(samples)),
            "maximum_gap_m": _round(maximum_gap),
            "maximum_allowed_gap_m": _round(float(spec["foundation"]["max_gap_m"])),
        },
    }


def grade_evidence(spec: Mapping[str, Any]) -> dict[str, Any]:
    spec = validate_fixture(spec)
    axis = axis_contract(spec)
    if spec["family"] == "TERRAIN_STEPPED":
        terraces = spec["terrain"]["terraces"]
        elevations = [float(row["elevation_m"]) for row in terraces]
        step_rows = []
        for index in range(len(terraces) - 1):
            signed = elevations[index + 1] - elevations[index]
            step_rows.append({
                "transition_id": f"grade-step/{index:02d}",
                "station_m": _round(float(terraces[index]["end_station_m"])),
                "signed_step_height_m": _round(signed),
                "absolute_step_height_m": _round(abs(signed)),
                "within_limit": abs(signed) <= float(spec["grade"]["max_step_height_m"]) + 1e-12,
            })
        overall_grade = abs(elevations[-1] - elevations[0]) / float(axis["length_m"])
        maximum_step = max((float(row["absolute_step_height_m"]) for row in step_rows), default=0.0)
        maximum_platform_grade = 0.0
        retained_height = 0.0
        equivalent_cross_slope = 0.0
    else:
        terrain = spec["terrain"]
        retained_height = float(terrain["inside_elevation_m"]) - float(terrain["outside_elevation_m"])
        width = float(terrain["outside_toe_width_m"]) + float(terrain["inside_heel_width_m"])
        equivalent_cross_slope = retained_height / width
        overall_grade = 0.0
        maximum_step = 0.0
        maximum_platform_grade = 0.0
        step_rows = []
    status = "PASS" if overall_grade <= float(spec["grade"]["max_centerline_grade"]) + 1e-12 and maximum_step <= float(spec["grade"]["max_step_height_m"]) + 1e-12 else "FAIL"
    return {
        "schema": GRADE_EVIDENCE_SCHEMA,
        "span_id": spec["span_id"],
        "family": spec["family"],
        "axis_length_m": axis["length_m"],
        "overall_centerline_grade": _round(overall_grade, 12),
        "maximum_platform_grade": _round(maximum_platform_grade, 12),
        "maximum_allowed_centerline_grade": _round(float(spec["grade"]["max_centerline_grade"]), 12),
        "maximum_step_height_m": _round(maximum_step),
        "maximum_allowed_step_height_m": _round(float(spec["grade"]["max_step_height_m"])),
        "retained_height_m": _round(retained_height),
        "equivalent_cross_slope_ratio": _round(equivalent_cross_slope, 12),
        "steps": step_rows,
        "status": status,
    }


def socket_plan(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = validate_fixture(spec)
    axis = axis_contract(spec)
    up, inside, outside, tangent = axis["up"], axis["inside"], axis["outside"], axis["tangent"]
    if spec["family"] == "TERRAIN_STEPPED":
        terraces = spec["terrain"]["terraces"]
        start_elevation = float(terraces[0]["elevation_m"])
        end_elevation = float(terraces[-1]["elevation_m"])
    else:
        start_elevation = end_elevation = float(spec["terrain"]["inside_elevation_m"])

    def point(station: float, elevation: float, y: float, z: float) -> list[float]:
        origin = _point_on_axis(axis, station, elevation)
        return _rv(tuple(origin[index] + y * float(up[index]) + z * float(inside[index]) for index in range(3)))

    frame = {"tangent": tangent, "up": up, "inside": inside, "outside": outside, "orientation_determinant": 1.0}
    rows = [
        {"socket_id": "span_start", "role": "SPAN_JOIN", "position_m": point(0.0, start_elevation, 6.0, 0.0), "frame": frame, "required": True},
        {"socket_id": "span_end", "role": "SPAN_JOIN", "position_m": point(float(axis["length_m"]), end_elevation, 6.0, 0.0), "frame": frame, "required": True},
        {"socket_id": "wall_walk_start", "role": "WALL_WALK_CONTINUATION", "position_m": point(0.0, start_elevation, 12.3, 0.0), "frame": frame, "required": True},
        {"socket_id": "wall_walk_end", "role": "WALL_WALK_CONTINUATION", "position_m": point(float(axis["length_m"]), end_elevation, 12.3, 0.0), "frame": frame, "required": True},
        {"socket_id": "foundation_start", "role": "FOUNDATION_INTERFACE", "position_m": point(0.0, start_elevation, -1.0, 0.0), "frame": frame, "required": True},
        {"socket_id": "foundation_end", "role": "FOUNDATION_INTERFACE", "position_m": point(float(axis["length_m"]), end_elevation, -1.0, 0.0), "frame": frame, "required": True},
    ]
    if spec["family"] == "TERRAIN_STEPPED":
        terraces = spec["terrain"]["terraces"]
        for index in range(len(terraces) - 1):
            lower = terraces[index]
            upper = terraces[index + 1]
            station = float(lower["end_station_m"])
            rows.extend([
                {"socket_id": f"walk_step_{index:02d}_lower", "role": "STAIR_RAMP_SOCKET", "position_m": point(station, float(lower["elevation_m"]), 12.3, 0.0), "frame": frame, "required": True},
                {"socket_id": f"walk_step_{index:02d}_upper", "role": "STAIR_RAMP_SOCKET", "position_m": point(station, float(upper["elevation_m"]), 12.3, 0.0), "frame": frame, "required": True},
                {"socket_id": f"foundation_step_{index:02d}_lower", "role": "FOUNDATION_STEP_INTERFACE", "position_m": point(station, float(lower["elevation_m"]), -1.0, 0.0), "frame": frame, "required": True},
                {"socket_id": f"foundation_step_{index:02d}_upper", "role": "FOUNDATION_STEP_INTERFACE", "position_m": point(station, float(upper["elevation_m"]), -1.0, 0.0), "frame": frame, "required": True},
            ])
    else:
        terrain = spec["terrain"]
        midpoint = float(axis["length_m"]) * 0.5
        rows.extend([
            {"socket_id": "drainage_outside_01", "role": "DRAINAGE", "position_m": point(midpoint, float(terrain["outside_elevation_m"]), 0.0, -float(terrain["outside_toe_width_m"])), "frame": frame, "required": False},
            {"socket_id": "retaining_inside_01", "role": "RETAINING_INTERFACE", "position_m": point(midpoint, float(terrain["inside_elevation_m"]), 0.0, float(terrain["inside_heel_width_m"])), "frame": frame, "required": True},
        ])
    return rows
