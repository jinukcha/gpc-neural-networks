from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

MIXED_FIXTURE_SCHEMA = "royal-capital.fortification.mixed-span-fixture/1"
SEGMENTATION_SCHEMA = "royal-capital.fortification.mixed-span-segmentation/1"
ASSEMBLY_PLAN_SCHEMA = "royal-capital.fortification.mixed-span-assembly-plan/1"
TRANSFORM_SCHEMA = "royal-capital.fortification.mixed-span-instance-transforms/1"
SOCKET_ALIGNMENT_SCHEMA = "royal-capital.fortification.mixed-span-socket-alignment/1"
MESH_SCHEMA = "royal-capital.fortification.mixed-span-indexed-mesh/1"
MODULE_INDEX_SCHEMA = "royal-capital.fortification.mixed-span-module-index/1"
COVERAGE_SCHEMA = "royal-capital.fortification.mixed-span-source-coverage/1"
RESULT_SCHEMA = "royal-capital.fortification.mixed-span-result/1"
RECEIPT_SCHEMA = "royal-capital.fortification.mixed-span-receipt/1"
PROJECT_FRAME = "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"
MODULE_KINDS = ("PATH_SPAN", "JOIN", "TERRAIN_SPAN")
EXPECTED_RUNTIME = {
    "build123d_version": "0.13.1.dev12+ge22d34dae",
    "ocp_version": "8.0.1.0.0",
    "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466",
    "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f",
}


class MixedSpanContractError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_ref(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _f(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MixedSpanContractError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise MixedSpanContractError(f"{name} must be finite")
    return out


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MixedSpanContractError(f"{name} must be an integer")
    return value


def _v3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise MixedSpanContractError(f"{name} must contain exactly three numbers")
    return tuple(_f(item, f"{name}[{index}]") for index, item in enumerate(value))


def _round(value: float, digits: int = 9) -> float:
    out = round(float(value), digits)
    return 0.0 if out == -0.0 else out


def _rv(value: Sequence[float], digits: int = 9) -> list[float]:
    return [_round(item, digits) for item in value]


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
        raise MixedSpanContractError("zero-length vector")
    return tuple(float(item) / magnitude for item in value)


def heading_deg(tangent: Sequence[float]) -> float:
    tangent = normalize(tangent)
    if abs(tangent[1]) > 1e-8:
        raise MixedSpanContractError("only horizontal headings are supported in R0B-CP4")
    return _round(math.degrees(math.atan2(tangent[2], tangent[0])), 9)


def rotate_heading(value: Sequence[float], angle_deg: float) -> tuple[float, float, float]:
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = (float(value[0]), float(value[1]), float(value[2]))
    return (c * x - s * z, y, s * x + c * z)


def transform_point(value: Sequence[float], heading: float, translation: Sequence[float]) -> tuple[float, float, float]:
    return add(rotate_heading(value, heading), translation)


def transform_vector(value: Sequence[float], heading: float) -> tuple[float, float, float]:
    return rotate_heading(value, heading)


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = normalize(a), normalize(b)
    return math.acos(max(-1.0, min(1.0, dot(na, nb))))


def _safe_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MixedSpanContractError(f"{name} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or value.startswith("/"):
        raise MixedSpanContractError(f"{name} must stay within the fortification tree")
    return pure.as_posix()


def _station_entry(module: Mapping[str, Any]) -> float:
    station = module["station"]
    return float(station["start_m"] if module["module_kind"] != "JOIN" else station["anchor_m"])


def _station_exit(module: Mapping[str, Any]) -> float:
    station = module["station"]
    return float(station["end_m"] if module["module_kind"] != "JOIN" else station["anchor_m"])


def _module_sort_key(module: Mapping[str, Any]) -> tuple[float, int, str]:
    rank = 0 if module["module_kind"] != "JOIN" else 1
    return (_station_entry(module), rank, str(module["module_id"]))


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "assembly_id", "frame", "root_module_id", "root_transform",
        "modules", "connections", "segmentation", "tolerances", "budget", "runtime",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        missing = sorted(required - set(value)) if isinstance(value, Mapping) else sorted(required)
        extra = sorted(set(value) - required) if isinstance(value, Mapping) else []
        raise MixedSpanContractError(f"fixture keys mismatch missing={missing} extra={extra}")
    if value["schema"] != MIXED_FIXTURE_SCHEMA:
        raise MixedSpanContractError("unsupported mixed-span fixture schema")
    if not isinstance(value["assembly_id"], str) or not value["assembly_id"].strip():
        raise MixedSpanContractError("assembly_id required")
    if value["frame"] != PROJECT_FRAME:
        raise MixedSpanContractError("project frame mismatch")
    if value["runtime"] != EXPECTED_RUNTIME:
        raise MixedSpanContractError("runtime identity mismatch")

    tolerances = value["tolerances"]
    required_tolerances = {"linear_m", "angular_rad", "station_m", "mesh_round_digits"}
    if not isinstance(tolerances, Mapping) or set(tolerances) != required_tolerances:
        raise MixedSpanContractError("tolerance keys mismatch")
    linear = _f(tolerances["linear_m"], "tolerances.linear_m")
    angular = _f(tolerances["angular_rad"], "tolerances.angular_rad")
    station_tolerance = _f(tolerances["station_m"], "tolerances.station_m")
    digits = _integer(tolerances["mesh_round_digits"], "tolerances.mesh_round_digits")
    if not 1e-9 <= linear <= 1e-3:
        raise MixedSpanContractError("linear tolerance outside [1e-9,1e-3]")
    if not 1e-9 <= angular <= 1e-3:
        raise MixedSpanContractError("angular tolerance outside [1e-9,1e-3]")
    if not 1e-9 <= station_tolerance <= 1e-3:
        raise MixedSpanContractError("station tolerance outside [1e-9,1e-3]")
    if not 6 <= digits <= 12:
        raise MixedSpanContractError("mesh_round_digits outside [6,12]")

    segmentation = value["segmentation"]
    required_segmentation = {"max_span_length_m", "max_modules", "require_contiguous_span_intervals", "array_order_is_identity"}
    if not isinstance(segmentation, Mapping) or set(segmentation) != required_segmentation:
        raise MixedSpanContractError("segmentation keys mismatch")
    max_span_length = _f(segmentation["max_span_length_m"], "segmentation.max_span_length_m")
    max_modules = _integer(segmentation["max_modules"], "segmentation.max_modules")
    if not 10.0 <= max_span_length <= 60.0:
        raise MixedSpanContractError("max span length outside [10,60]")
    if not 1 <= max_modules <= 4096:
        raise MixedSpanContractError("max_modules outside [1,4096]")
    if segmentation["require_contiguous_span_intervals"] is not True:
        raise MixedSpanContractError("R0B-CP4 requires contiguous span intervals")
    if segmentation["array_order_is_identity"] is not False:
        raise MixedSpanContractError("array order cannot be identity")

    budget = value["budget"]
    required_budget = {"max_modules", "max_vertices", "max_triangles", "max_coverage_rows", "max_artifact_bytes"}
    if not isinstance(budget, Mapping) or set(budget) != required_budget:
        raise MixedSpanContractError("budget keys mismatch")
    parsed_budget = {key: _integer(budget[key], f"budget.{key}") for key in required_budget}
    if any(number <= 0 for number in parsed_budget.values()):
        raise MixedSpanContractError("all budget values must be positive")

    root_transform = value["root_transform"]
    if not isinstance(root_transform, Mapping) or set(root_transform) != {"translation_m", "heading_deg"}:
        raise MixedSpanContractError("root_transform keys mismatch")
    root_translation = _v3(root_transform["translation_m"], "root_transform.translation_m")
    root_heading = _f(root_transform["heading_deg"], "root_transform.heading_deg")

    raw_modules = value["modules"]
    if not isinstance(raw_modules, Sequence) or isinstance(raw_modules, (str, bytes)) or not raw_modules:
        raise MixedSpanContractError("modules must be a non-empty array")
    if len(raw_modules) > min(max_modules, parsed_budget["max_modules"]):
        raise MixedSpanContractError("module count exceeds budget")
    modules: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_modules):
        required_module = {
            "module_id", "module_kind", "family", "source_dir", "source_result_sha256",
            "source_mesh_sha256", "station", "input_socket", "output_socket", "required",
        }
        if not isinstance(raw, Mapping) or set(raw) != required_module:
            raise MixedSpanContractError(f"modules[{index}] keys mismatch")
        module_id = raw["module_id"]
        if not isinstance(module_id, str) or not module_id.strip() or module_id in ids:
            raise MixedSpanContractError(f"invalid or duplicate module_id {module_id!r}")
        ids.add(module_id)
        kind = raw["module_kind"]
        if kind not in MODULE_KINDS:
            raise MixedSpanContractError(f"unsupported module kind {kind!r}")
        family = raw["family"]
        if not isinstance(family, str) or not family.strip():
            raise MixedSpanContractError(f"modules[{index}].family required")
        source_dir = _safe_relative_path(raw["source_dir"], f"modules[{index}].source_dir")
        for digest_key in ("source_result_sha256", "source_mesh_sha256"):
            digest = raw[digest_key]
            if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
                raise MixedSpanContractError(f"modules[{index}].{digest_key} invalid")
        station = raw["station"]
        if not isinstance(station, Mapping):
            raise MixedSpanContractError(f"modules[{index}].station must be an object")
        if kind == "JOIN":
            if set(station) != {"anchor_m"}:
                raise MixedSpanContractError(f"join {module_id} station keys mismatch")
            anchor = _f(station["anchor_m"], f"modules[{index}].station.anchor_m")
            parsed_station = {"anchor_m": _round(anchor, digits)}
        else:
            if set(station) != {"start_m", "end_m"}:
                raise MixedSpanContractError(f"span {module_id} station keys mismatch")
            start = _f(station["start_m"], f"modules[{index}].station.start_m")
            end = _f(station["end_m"], f"modules[{index}].station.end_m")
            if end - start <= station_tolerance:
                raise MixedSpanContractError(f"span {module_id} has a non-positive station interval")
            if end - start > max_span_length + station_tolerance:
                raise MixedSpanContractError(f"span {module_id} exceeds max span length")
            parsed_station = {"start_m": _round(start, digits), "end_m": _round(end, digits)}
        for socket_key in ("input_socket", "output_socket"):
            if not isinstance(raw[socket_key], str) or not raw[socket_key].strip():
                raise MixedSpanContractError(f"modules[{index}].{socket_key} required")
        if raw["required"] is not True:
            raise MixedSpanContractError("all R0B closeout fixture modules are required")
        modules.append({
            "module_id": module_id,
            "module_kind": kind,
            "family": family,
            "source_dir": source_dir,
            "source_result_sha256": raw["source_result_sha256"],
            "source_mesh_sha256": raw["source_mesh_sha256"],
            "station": parsed_station,
            "input_socket": raw["input_socket"],
            "output_socket": raw["output_socket"],
            "required": True,
        })

    root_module_id = value["root_module_id"]
    if root_module_id not in ids:
        raise MixedSpanContractError("root_module_id does not identify a module")

    raw_connections = value["connections"]
    if not isinstance(raw_connections, Sequence) or isinstance(raw_connections, (str, bytes)):
        raise MixedSpanContractError("connections must be an array")
    if len(raw_connections) != len(modules) - 1:
        raise MixedSpanContractError("mixed fixture must form one chain with module_count-1 connections")
    connections: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    module_by_id = {module["module_id"]: module for module in modules}
    indegree = {module_id: 0 for module_id in ids}
    outdegree = {module_id: 0 for module_id in ids}
    outgoing_by_module: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(raw_connections):
        required_connection = {"from_module", "from_socket", "to_module", "to_socket"}
        if not isinstance(raw, Mapping) or set(raw) != required_connection:
            raise MixedSpanContractError(f"connections[{index}] keys mismatch")
        source = raw["from_module"]
        target = raw["to_module"]
        if source not in ids or target not in ids or source == target:
            raise MixedSpanContractError(f"connections[{index}] references invalid modules")
        edge = (source, target)
        if edge in seen_edges:
            raise MixedSpanContractError(f"duplicate connection {source}->{target}")
        seen_edges.add(edge)
        if raw["from_socket"] != module_by_id[source]["output_socket"]:
            raise MixedSpanContractError(f"connection {source}->{target} does not use declared output socket")
        if raw["to_socket"] != module_by_id[target]["input_socket"]:
            raise MixedSpanContractError(f"connection {source}->{target} does not use declared input socket")
        indegree[target] += 1
        outdegree[source] += 1
        if indegree[target] > 1 or outdegree[source] > 1:
            raise MixedSpanContractError("mixed fixture is not a single bounded chain")
        source_station = _station_exit(module_by_id[source])
        target_station = _station_entry(module_by_id[target])
        if abs(source_station - target_station) > station_tolerance:
            raise MixedSpanContractError(f"connection station mismatch {source}->{target}: {source_station} != {target_station}")
        row = {
            "from_module": source,
            "from_socket": raw["from_socket"],
            "to_module": target,
            "to_socket": raw["to_socket"],
        }
        connections.append(row)
        outgoing_by_module[source] = row

    if indegree[root_module_id] != 0:
        raise MixedSpanContractError("root module must have zero indegree")
    if [module_id for module_id, degree in indegree.items() if degree == 0] != [root_module_id]:
        roots = sorted(module_id for module_id, degree in indegree.items() if degree == 0)
        if roots != [root_module_id]:
            raise MixedSpanContractError(f"expected exactly one root {root_module_id}, observed {roots}")
    terminals = sorted(module_id for module_id, degree in outdegree.items() if degree == 0)
    if len(terminals) != 1:
        raise MixedSpanContractError(f"expected one terminal module, observed {terminals}")
    chain: list[str] = []
    current = root_module_id
    visited: set[str] = set()
    while True:
        if current in visited:
            raise MixedSpanContractError("connection cycle detected")
        visited.add(current)
        chain.append(current)
        if current not in outgoing_by_module:
            break
        current = outgoing_by_module[current]["to_module"]
    if visited != ids:
        raise MixedSpanContractError(f"connection chain does not cover all modules: missing={sorted(ids-visited)}")

    spans = sorted((module for module in modules if module["module_kind"] != "JOIN"), key=_module_sort_key)
    if not spans:
        raise MixedSpanContractError("mixed fixture requires at least one span")
    for index, span in enumerate(spans):
        if index and abs(float(spans[index - 1]["station"]["end_m"]) - float(span["station"]["start_m"])) > station_tolerance:
            raise MixedSpanContractError(f"span segmentation gap/overlap between {spans[index-1]['module_id']} and {span['module_id']}")
    span_boundaries = [float(spans[0]["station"]["start_m"])] + [float(span["station"]["end_m"]) for span in spans]
    joins = sorted((module for module in modules if module["module_kind"] == "JOIN"), key=_module_sort_key)
    for join in joins:
        anchor = float(join["station"]["anchor_m"])
        if not any(abs(anchor - boundary) <= station_tolerance for boundary in span_boundaries):
            raise MixedSpanContractError(f"join {join['module_id']} is not anchored to a canonical span boundary")

    canonical_modules = sorted(modules, key=lambda module: (chain.index(module["module_id"]), module["module_id"]))
    connection_by_source = {row["from_module"]: row for row in connections}
    canonical_connections = [connection_by_source[module_id] for module_id in chain[:-1]]
    normalized = {
        "schema": MIXED_FIXTURE_SCHEMA,
        "assembly_id": str(value["assembly_id"]),
        "frame": PROJECT_FRAME,
        "root_module_id": root_module_id,
        "root_transform": {"translation_m": _rv(root_translation, digits), "heading_deg": _round(root_heading, digits)},
        "modules": canonical_modules,
        "connections": canonical_connections,
        "segmentation": {
            "max_span_length_m": _round(max_span_length, digits),
            "max_modules": max_modules,
            "require_contiguous_span_intervals": True,
            "array_order_is_identity": False,
        },
        "tolerances": {
            "linear_m": linear,
            "angular_rad": angular,
            "station_m": station_tolerance,
            "mesh_round_digits": digits,
        },
        "budget": parsed_budget,
        "runtime": dict(EXPECTED_RUNTIME),
    }
    return normalized


def canonicalize_segmentation(value: Mapping[str, Any]) -> dict[str, Any]:
    fixture = validate_fixture(value)
    modules = fixture["modules"]
    chain = [module["module_id"] for module in modules]
    spans = [module for module in modules if module["module_kind"] != "JOIN"]
    joins = [module for module in modules if module["module_kind"] == "JOIN"]
    span_rows = []
    for module in spans:
        descriptor = {
            "module_id": module["module_id"],
            "module_kind": module["module_kind"],
            "family": module["family"],
            "station": module["station"],
            "source_result_sha256": module["source_result_sha256"],
            "source_mesh_sha256": module["source_mesh_sha256"],
        }
        span_rows.append({
            **descriptor,
            "length_m": _round(float(module["station"]["end_m"]) - float(module["station"]["start_m"])),
            "stable_segment_key": sha256_ref(canonical_json_bytes(descriptor)),
        })
    join_rows = []
    for module in joins:
        descriptor = {
            "module_id": module["module_id"],
            "family": module["family"],
            "anchor_m": module["station"]["anchor_m"],
            "source_result_sha256": module["source_result_sha256"],
            "source_mesh_sha256": module["source_mesh_sha256"],
        }
        join_rows.append({**descriptor, "stable_join_key": sha256_ref(canonical_json_bytes(descriptor))})
    boundaries = sorted({
        float(module["station"][key])
        for module in spans
        for key in ("start_m", "end_m")
    })
    plan = {
        "schema": SEGMENTATION_SCHEMA,
        "assembly_id": fixture["assembly_id"],
        "frame": fixture["frame"],
        "identity_policy": "EXPLICIT_MODULE_ID_PLUS_CANONICAL_SOURCE_INTERVAL_AND_DIGEST",
        "input_array_order_is_identity": False,
        "chain_order": chain,
        "span_segments": span_rows,
        "join_anchors": join_rows,
        "boundaries_m": [_round(boundary) for boundary in boundaries],
        "span_segment_count": len(span_rows),
        "join_count": len(join_rows),
        "module_count": len(modules),
        "total_station_length_m": _round(boundaries[-1] - boundaries[0]),
    }
    plan["segmentation_digest"] = sha256_ref(canonical_json_bytes({key: val for key, val in plan.items() if key != "segmentation_digest"}))
    return plan
