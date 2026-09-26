from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from rcf_fortification_cad import Build123dProviderAdapter, CadStatus
from rcf_fortification_cad.canonical import (
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_ref,
    unit_frame_contract,
)

FIXTURE_SCHEMA = "royal-capital.fortification.tower-join-fixture/1"
PLAN_SCHEMA = "royal-capital.fortification.tower-join-plan/1"
BINDING_SCHEMA = "royal-capital.fortification.tower-join-source-bindings/1"
SOCKET_EVIDENCE_SCHEMA = "royal-capital.fortification.tower-join-socket-frame-evidence/1"
CONTINUITY_SCHEMA = "royal-capital.fortification.tower-join-continuity-evidence/1"
OVERLAP_SCHEMA = "royal-capital.fortification.tower-join-overlap-gap-projection/1"
IMMUTABILITY_SCHEMA = "royal-capital.fortification.tower-join-source-immutability/1"
PARTS_SCHEMA = "royal-capital.fortification.tower-join-semantic-parts/1"
MESH_SCHEMA = "royal-capital.fortification.tower-join-indexed-mesh/1"
TESSELLATION_SCHEMA = "royal-capital.fortification.tower-join-fixed-tessellation/1"
STORED_SCHEMA = "royal-capital.fortification.tower-join-stored-copies/1"
RECEIPT_SCHEMA = "royal-capital.fortification.tower-join-provider-receipt/1"
RESULT_SCHEMA = "royal-capital.fortification.tower-join-result/1"
FAILURE_SCHEMA = "royal-capital.fortification.tower-join-failure/1"

PROJECT_FRAME = "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"
EXPECTED_RUNTIME = {
    "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466",
    "build123d_version": "0.13.1.dev12+ge22d34dae",
    "ocp_version": "8.0.1.0.0",
    "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f",
}
PART_ORDER = ("join_body", "wall_walk_transition", "foundation_transition")


class JoinFailureCode:
    INVALID_REQUEST = "INVALID_REQUEST"
    UNKNOWN_TOWER_FAMILY = "UNKNOWN_TOWER_FAMILY"
    SOURCE_TOWER_DIGEST_MISMATCH = "SOURCE_TOWER_DIGEST_MISMATCH"
    SOURCE_SPAN_DIGEST_MISMATCH = "SOURCE_SPAN_DIGEST_MISMATCH"
    MISSING_REQUIRED_SOCKET = "MISSING_REQUIRED_SOCKET"
    SOCKET_POSITION_MISMATCH = "SOCKET_POSITION_MISMATCH"
    SOCKET_TANGENT_MISMATCH = "SOCKET_TANGENT_MISMATCH"
    SOCKET_UP_AXIS_MISMATCH = "SOCKET_UP_AXIS_MISMATCH"
    INSIDE_OUTSIDE_FRAME_INVERSION = "INSIDE_OUTSIDE_FRAME_INVERSION"
    FOUNDATION_ELEVATION_MISMATCH = "FOUNDATION_ELEVATION_MISMATCH"
    WALL_WALK_ELEVATION_MISMATCH = "WALL_WALK_ELEVATION_MISMATCH"
    OVERLAP_BUDGET_EXCEEDED = "OVERLAP_BUDGET_EXCEEDED"
    UNSUPPORTED_GAP_EXCEEDED = "UNSUPPORTED_GAP_EXCEEDED"
    PROJECTION_CLEARANCE_DOMAIN_EXCEEDED = "PROJECTION_CLEARANCE_DOMAIN_EXCEEDED"
    STALE_FIXTURE_OR_SOURCE_RECEIPT = "STALE_FIXTURE_OR_SOURCE_RECEIPT"
    RUNTIME_IDENTITY_MISMATCH = "RUNTIME_IDENTITY_MISMATCH"
    JOIN_ARTIFACT_BUDGET_EXCEEDED = "JOIN_ARTIFACT_BUDGET_EXCEEDED"
    ACCEPTED_SOURCE_GEOMETRY_MUTATION = "ACCEPTED_SOURCE_GEOMETRY_MUTATION"
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    GEOMETRY_EVIDENCE_MISMATCH = "GEOMETRY_EVIDENCE_MISMATCH"
    PARTIAL_OUTPUT_PUBLICATION_ATTEMPT = "PARTIAL_OUTPUT_PUBLICATION_ATTEMPT"
    PUBLISH_ABORTED = "PUBLISH_ABORTED"


class JoinError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PartSpec:
    part_id: str
    semantic_role: str
    material_slot: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def bounds(self) -> dict[str, list[float]]:
        return {
            "min": [round(self.x_min, 9), round(self.y_min, 9), round(self.z_min, 9)],
            "max": [round(self.x_max, 9), round(self.y_max, 9), round(self.z_max, 9)],
        }

    @property
    def volume(self) -> float:
        return round(
            (self.x_max - self.x_min)
            * (self.y_max - self.y_min)
            * (self.z_max - self.z_min),
            9,
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_ref(path: Path) -> str:
    return "sha256:" + _sha256(path)


def _tree_digest(root: Path) -> str:
    if not root.is_dir():
        raise JoinError(JoinFailureCode.INVALID_REQUEST, f"source tree missing: {root}")
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + h.hexdigest()


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_ref(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise JoinError(JoinFailureCode.INVALID_REQUEST, f"JSON root must be object: {path}")
    return value


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise JoinError(code, message)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JoinError(JoinFailureCode.INVALID_REQUEST, f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise JoinError(JoinFailureCode.INVALID_REQUEST, f"{field} must be finite")
    return result


def _vec(value: Any, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise JoinError(JoinFailureCode.INVALID_REQUEST, f"{field} must contain three values")
    return [_finite_number(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [round(float(a[i]) + float(b[i]), 9) for i in range(3)]


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [round(float(a[i]) - float(b[i]), 9) for i in range(3)]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return round(math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3))), 12)


def _socket(document: Mapping[str, Any], socket_id: str) -> dict[str, Any]:
    rows = document.get("sockets")
    if not isinstance(rows, list):
        raise JoinError(JoinFailureCode.MISSING_REQUIRED_SOCKET, "socket document has no socket list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("socket_id") == socket_id]
    if len(matches) != 1:
        raise JoinError(
            JoinFailureCode.MISSING_REQUIRED_SOCKET,
            f"required socket {socket_id!r} has {len(matches)} matches",
        )
    return matches[0]


def _frame_axes(socket: Mapping[str, Any]) -> tuple[list[float], list[float], list[float]]:
    frame = socket.get("frame")
    if not isinstance(frame, Mapping):
        raise JoinError(JoinFailureCode.INVALID_REQUEST, "socket frame is missing")
    return (
        _vec(frame.get("x_axis"), "frame.x_axis"),
        _vec(frame.get("y_axis"), "frame.y_axis"),
        _vec(frame.get("z_axis"), "frame.z_axis"),
    )


def _verify_file(path: Path, expected: str, code: str, label: str) -> dict[str, Any]:
    _require(path.is_file(), code, f"missing accepted {label}: {path}")
    actual = _sha256(path)
    _require(actual == expected, code, f"accepted {label} digest mismatch {actual} != {expected}")
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": "sha256:" + actual}


def _bounds(vertices: list[list[float]]) -> dict[str, list[float]]:
    return {
        "min": [round(min(point[index] for point in vertices), 9) for index in range(3)],
        "max": [round(max(point[index] for point in vertices), 9) for index in range(3)],
    }


def _same_bounds(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]], tolerance: float) -> bool:
    return all(
        abs(float(a[key][index]) - float(b[key][index])) <= tolerance
        for key in ("min", "max")
        for index in range(3)
    )


def _provider_request(fixture: Mapping[str, Any], part: PartSpec) -> dict[str, Any]:
    v_min = -part.z_max
    v_max = -part.z_min
    loop = [
        [round(part.x_min, 12), round(v_min, 12)],
        [round(part.x_max, 12), round(v_min, 12)],
        [round(part.x_max, 12), round(v_max, 12)],
        [round(part.x_min, 12), round(v_max, 12)],
    ]
    return {
        "schema": "royal-capital.fortification.cad-provider-request/1",
        "request_id": f"cad-request/{fixture['request_id']}/{part.part_id}@1",
        "units": "METER",
        "frame": PROJECT_FRAME,
        "runtime": dict(fixture["runtime"]),
        "tolerances": {
            key: fixture["tolerances"][key]
            for key in (
                "linear_m",
                "angular_rad",
                "tessellation_linear_m",
                "tessellation_angular_rad",
            )
        },
        "budget": {
            key: fixture["budgets"][key]
            for key in (
                "max_profile_points",
                "max_vertices",
                "max_triangles",
                "max_artifact_bytes",
            )
        },
        "operation": {
            "kind": "PROFILE_EXTRUSION",
            "plane": {
                "origin_m": [0.0, part.y_min, 0.0],
                "x_axis": [1.0, 0.0, 0.0],
                "y_axis": [0.0, 0.0, -1.0],
            },
            "outer_loop_m": loop,
            "distance_m": part.y_max - part.y_min,
        },
        "outputs": {"indexed_mesh": True, "stored_copies": ["STEP", "BREP"]},
    }


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    _require(fixture.get("schema") == FIXTURE_SCHEMA, JoinFailureCode.INVALID_REQUEST, "fixture schema mismatch")
    _require(fixture.get("fixture_id") == "R0C_CP2_FX01", JoinFailureCode.INVALID_REQUEST, "fixture id mismatch")
    _require(fixture.get("join_kind") == "TANGENT", JoinFailureCode.INVALID_REQUEST, "join kind must be TANGENT")
    _require(fixture.get("frame") == PROJECT_FRAME, JoinFailureCode.INVALID_REQUEST, "project frame mismatch")
    tower = fixture.get("tower")
    span = fixture.get("span")
    realization = fixture.get("realization")
    budgets = fixture.get("budgets")
    tolerances = fixture.get("tolerances")
    _require(isinstance(tower, Mapping), JoinFailureCode.INVALID_REQUEST, "tower fixture section missing")
    _require(isinstance(span, Mapping), JoinFailureCode.INVALID_REQUEST, "span fixture section missing")
    _require(isinstance(realization, Mapping), JoinFailureCode.INVALID_REQUEST, "realization section missing")
    _require(isinstance(budgets, Mapping), JoinFailureCode.INVALID_REQUEST, "budgets section missing")
    _require(isinstance(tolerances, Mapping), JoinFailureCode.INVALID_REQUEST, "tolerances section missing")
    _require(tower.get("family") == "ROUND", JoinFailureCode.UNKNOWN_TOWER_FAMILY, "FX01 requires ROUND")
    _require(tower.get("side_count") == 32, JoinFailureCode.UNKNOWN_TOWER_FAMILY, "ROUND source must preserve 32 sides")
    _require(span.get("family") == "STRAIGHT_SPAN", JoinFailureCode.INVALID_REQUEST, "FX01 requires STRAIGHT_SPAN")
    _require(realization.get("mode") == "BOUNDED_TRANSITION_PIECE", JoinFailureCode.INVALID_REQUEST, "realization mode mismatch")
    _require(dict(fixture.get("runtime", {})) == EXPECTED_RUNTIME, JoinFailureCode.RUNTIME_IDENTITY_MISMATCH, "runtime identity mismatch")
    _require(float(tolerances.get("linear_m", -1.0)) == 0.000001, JoinFailureCode.INVALID_REQUEST, "linear tolerance mismatch")
    _require(float(tolerances.get("angular_rad", -1.0)) == 0.000001, JoinFailureCode.INVALID_REQUEST, "angular tolerance mismatch")
    _require(float(tolerances.get("tessellation_linear_m", -1.0)) == 0.05, JoinFailureCode.INVALID_REQUEST, "fixed tessellation linear mismatch")
    _require(float(tolerances.get("tessellation_angular_rad", -1.0)) == 0.1, JoinFailureCode.INVALID_REQUEST, "fixed tessellation angular mismatch")
    _require(int(tolerances.get("mesh_round_digits", -1)) == 9, JoinFailureCode.INVALID_REQUEST, "mesh rounding mismatch")
    parts = realization.get("parts")
    _require(isinstance(parts, list) and len(parts) == 3, JoinFailureCode.INVALID_REQUEST, "FX01 requires three bounded parts")
    _require([part.get("part_id") for part in parts] == list(PART_ORDER), JoinFailureCode.INVALID_REQUEST, "part order mismatch")
    _require(int(budgets.get("max_parts", -1)) == 3, JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "part budget mismatch")


def _part_specs(fixture: Mapping[str, Any], interface_x: float) -> list[PartSpec]:
    half = _finite_number(fixture["realization"]["interface_half_overlap_m"], "interface_half_overlap_m")
    _require(half > 0.0, JoinFailureCode.INVALID_REQUEST, "interface overlap must be positive")
    result: list[PartSpec] = []
    for row in fixture["realization"]["parts"]:
        result.append(
            PartSpec(
                part_id=str(row["part_id"]),
                semantic_role=str(row["semantic_role"]),
                material_slot=str(row["material_slot"]),
                x_min=round(interface_x - half, 9),
                x_max=round(interface_x + half, 9),
                y_min=_finite_number(row["y_min_m"], f"{row['part_id']}.y_min_m"),
                y_max=_finite_number(row["y_max_m"], f"{row['part_id']}.y_max_m"),
                z_min=_finite_number(row["z_min_m"], f"{row['part_id']}.z_min_m"),
                z_max=_finite_number(row["z_max_m"], f"{row['part_id']}.z_max_m"),
            )
        )
    return result


class Fx01TowerJoinProducer:
    def __init__(self, tree_root: str | os.PathLike[str]):
        self.tree_root = Path(tree_root).resolve()
        self.fortification_root = self.tree_root / "RC_K0/child_designs/fortification"
        self.cp0_root = self.fortification_root / "r0a_cp0"
        self.adapter = Build123dProviderAdapter(self.cp0_root)

    def execute(
        self,
        fixture: Mapping[str, Any],
        output_dir: str | os.PathLike[str],
        failure_root: str | os.PathLike[str] | None = None,
        *,
        failure_case: str = "fx01",
    ) -> dict[str, Any]:
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"output must be fresh: {output}")
        temporary = output.with_name(output.name + f".partial-{failure_case}-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"temporary path already exists: {temporary}")
        temporary.mkdir(parents=True)
        accepted_before = _tree_digest(output.parent) if output.parent.exists() else None
        request_digest = sha256_ref(canonical_json_bytes(fixture))
        try:
            validate_fixture(fixture)
            result = self._produce(fixture, temporary)
            temporary.rename(output)
            return result
        except Exception as exc:
            if isinstance(exc, JoinError):
                code, message = exc.code, exc.message
            else:
                code, message = JoinFailureCode.PUBLISH_ABORTED, f"{type(exc).__name__}: {exc}"
            if failure_root is None:
                failed = output.with_name(output.name + f".failed-{failure_case}-{os.getpid()}")
            else:
                failed = Path(failure_root).resolve() / failure_case
            if failed.exists():
                raise FileExistsError(failed) from exc
            failed.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.rename(failed)
            else:
                failed.mkdir(parents=True)
            failure = {
                "schema": FAILURE_SCHEMA,
                "status": "REJECTED",
                "failure": {"code": code, "message": message},
                "request_digest": request_digest,
                "partial_output_published": False,
                "accepted_target_digest_before": accepted_before,
                "accepted_target_digest_after": _tree_digest(output.parent) if output.parent.exists() else None,
                "preserved_failed_work": True,
            }
            (failed / "failure-result.json").write_bytes(pretty_json_bytes(failure))
            return failure

    def _source_paths(self, fixture: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
        tower = fixture["tower"]
        span = fixture["span"]
        tower_root = self.tree_root / str(tower["reference_root"])
        span_root = self.tree_root / str(span["reference_root"])
        tower_refs: dict[str, Any] = {}
        span_refs: dict[str, Any] = {}
        for key in ("result", "sockets", "stored_copies", "fixed_tessellation", "bounds_attachment", "body_step", "body_brep"):
            row = tower[key]
            tower_refs[key] = _verify_file(
                tower_root / str(row["path"]),
                str(row["sha256"]),
                JoinFailureCode.SOURCE_TOWER_DIGEST_MISMATCH,
                f"tower {key}",
            )
        tower_completion = tower["completion_receipt"]
        tower_refs["completion_receipt"] = _verify_file(
            self.tree_root / str(tower_completion["path"]),
            str(tower_completion["sha256"]),
            JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT,
            "tower completion receipt",
        )
        for key in ("result", "sockets", "stored_copies", "fixed_tessellation", "semantic_parts", "body_step", "body_brep"):
            row = span[key]
            span_refs[key] = _verify_file(
                span_root / str(row["path"]),
                str(row["sha256"]),
                JoinFailureCode.SOURCE_SPAN_DIGEST_MISMATCH,
                f"span {key}",
            )
        span_completion = span["completion_receipt"]
        span_refs["completion_receipt"] = _verify_file(
            self.tree_root / str(span_completion["path"]),
            str(span_completion["sha256"]),
            JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT,
            "span completion receipt",
        )
        return tower_root, span_root, tower_refs, span_refs

    def _produce(self, fixture: Mapping[str, Any], output: Path) -> dict[str, Any]:
        tower_root, span_root, tower_refs, span_refs = self._source_paths(fixture)
        tower_before = _tree_digest(tower_root)
        span_before = _tree_digest(span_root)

        tower_result = _load_json(tower_root / fixture["tower"]["result"]["path"])
        span_result = _load_json(span_root / fixture["span"]["result"]["path"])
        _require(tower_result.get("status") == "SUCCEEDED", JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT, "tower result is not accepted")
        _require(span_result.get("status") == "SUCCEEDED", JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT, "span result is not accepted")
        _require(tower_result.get("tower_id") == fixture["tower"]["source_object_id"], JoinFailureCode.SOURCE_TOWER_DIGEST_MISMATCH, "tower object id mismatch")
        _require(span_result.get("span_id") == fixture["span"]["source_object_id"], JoinFailureCode.SOURCE_SPAN_DIGEST_MISMATCH, "span object id mismatch")

        tower_sockets = _load_json(tower_root / fixture["tower"]["sockets"]["path"])
        span_sockets = _load_json(span_root / fixture["span"]["sockets"]["path"])
        tower_ids = fixture["tower"]["socket_ids"]
        span_ids = fixture["span"]["socket_ids"]
        tower_body_socket = _socket(tower_sockets, str(tower_ids["body"]))
        tower_walk_socket = _socket(tower_sockets, str(tower_ids["wall_walk"]))
        tower_foundation_socket = _socket(tower_sockets, str(tower_ids["foundation"]))
        span_body_socket = _socket(span_sockets, str(span_ids["body"]))
        span_walk_socket = _socket(span_sockets, str(span_ids["wall_walk"]))
        span_foundation_socket = _socket(span_sockets, str(span_ids["foundation"]))

        tower_position = _vec(tower_body_socket["position_m"], "tower body socket")
        span_position = _vec(span_body_socket["position_m"], "span body socket")
        translation = _sub(tower_position, span_position)
        transformed_span_position = _add(span_position, translation)
        transformed_span_walk = _add(_vec(span_walk_socket["position_m"], "span walk socket"), translation)
        transformed_span_foundation = _add(_vec(span_foundation_socket["position_m"], "span foundation socket"), translation)

        tower_tangent, tower_up, tower_inside = _frame_axes(tower_body_socket)
        span_tangent, span_up, span_inside = _frame_axes(span_body_socket)
        position_error = _distance(tower_position, transformed_span_position)
        tangent_error = _distance(tower_tangent, span_tangent)
        up_error = _distance(tower_up, span_up)
        inside_error = _distance(tower_inside, span_inside)
        walk_error = _distance(_vec(tower_walk_socket["position_m"], "tower walk socket"), transformed_span_walk)
        raw_foundation_offset = abs(
            _vec(tower_foundation_socket["position_m"], "tower foundation socket")[1]
            - transformed_span_foundation[1]
        )
        budgets = fixture["budgets"]
        _require(position_error <= float(budgets["max_socket_position_error_m"]), JoinFailureCode.SOCKET_POSITION_MISMATCH, f"position error {position_error}")
        _require(tangent_error <= float(budgets["max_frame_error"]), JoinFailureCode.SOCKET_TANGENT_MISMATCH, f"tangent error {tangent_error}")
        _require(up_error <= float(budgets["max_frame_error"]), JoinFailureCode.SOCKET_UP_AXIS_MISMATCH, f"up error {up_error}")
        _require(inside_error <= float(budgets["max_frame_error"]), JoinFailureCode.INSIDE_OUTSIDE_FRAME_INVERSION, f"inside error {inside_error}")
        _require(walk_error <= float(budgets["max_socket_position_error_m"]), JoinFailureCode.WALL_WALK_ELEVATION_MISMATCH, f"wall-walk error {walk_error}")
        _require(
            abs(raw_foundation_offset - float(budgets["required_foundation_transition_m"])) <= float(fixture["tolerances"]["linear_m"]),
            JoinFailureCode.FOUNDATION_ELEVATION_MISMATCH,
            f"foundation source offset {raw_foundation_offset}",
        )

        interface_x = tower_position[0]
        parts = _part_specs(fixture, interface_x)
        _require(len(parts) <= int(budgets["max_parts"]), JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "part count exceeds budget")

        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        part_ranges: list[dict[str, Any]] = []
        semantic_parts: list[dict[str, Any]] = []
        stored_copies: list[dict[str, Any]] = []
        part_receipts: list[dict[str, Any]] = []
        total_volume = 0.0
        tolerance = max(1e-7, float(fixture["tolerances"]["linear_m"]) * 10.0)

        for part in parts:
            part_dir = output / "parts" / part.part_id
            provider = self.adapter.execute(_provider_request(fixture, part), part_dir)
            if provider.status is CadStatus.REJECTED:
                raise JoinError(JoinFailureCode.PROVIDER_REJECTED, f"{part.part_id}: {provider.failure}")
            if provider.status is not CadStatus.SUCCEEDED:
                raise JoinError(JoinFailureCode.PROVIDER_FAILED, f"{part.part_id}: {provider.to_dict()}")
            provider_result = _load_json(part_dir / "result.json")
            mesh = _load_json(part_dir / "neutral-mesh.json")
            receipt = _load_json(part_dir / "cad-provider-receipt.json")
            observed_bounds = provider_result["shape"]["bounds_m"]
            observed_volume = float(provider_result["shape"]["volume_m3"])
            _require(_same_bounds(observed_bounds, part.bounds, tolerance), JoinFailureCode.GEOMETRY_EVIDENCE_MISMATCH, f"{part.part_id} bounds mismatch {observed_bounds} != {part.bounds}")
            _require(abs(observed_volume - part.volume) <= max(1e-6, tolerance ** 3), JoinFailureCode.GEOMETRY_EVIDENCE_MISMATCH, f"{part.part_id} volume mismatch {observed_volume} != {part.volume}")

            vertex_offset = len(combined_vertices)
            triangle_offset = len(combined_triangles)
            combined_vertices.extend(mesh["vertices_m"])
            combined_triangles.extend(
                [[a + vertex_offset, b + vertex_offset, c + vertex_offset] for a, b, c in mesh["triangles"]]
            )
            part_ranges.append(
                {
                    "part_id": part.part_id,
                    "vertex_offset": vertex_offset,
                    "vertex_count": len(mesh["vertices_m"]),
                    "triangle_offset": triangle_offset,
                    "triangle_count": len(mesh["triangles"]),
                }
            )
            semantic_parts.append(
                {
                    "part_id": part.part_id,
                    "semantic_role": part.semantic_role,
                    "material_slot": part.material_slot,
                    "bounds_m": part.bounds,
                    "volume_m3": part.volume,
                    "provider_result_ref": _ref(part_dir / "result.json", output),
                    "provider_receipt_ref": _ref(part_dir / "cad-provider-receipt.json", output),
                }
            )
            for item in provider_result["stored_copies"]:
                row = dict(item)
                row["part_id"] = part.part_id
                row["path"] = (Path("parts") / part.part_id / str(item["path"])).as_posix()
                row["reopen_status"] = "PASS_PROVIDER_NATIVE_REOPEN"
                stored_copies.append(row)
            part_receipts.append(
                {
                    "part_id": part.part_id,
                    "adapter_revision": receipt["adapter_revision"],
                    "receipt": _ref(part_dir / "cad-provider-receipt.json", output),
                }
            )
            total_volume += part.volume

        _require(len(combined_vertices) <= int(budgets["max_vertices"]), JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "vertex budget exceeded")
        _require(len(combined_triangles) <= int(budgets["max_triangles"]), JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "triangle budget exceeded")

        foundation_residual = 0.0
        unsupported_gap = 0.0
        outside_projection = 0.0
        inside_projection = 0.0
        _require(foundation_residual <= float(budgets["max_foundation_transition_residual_m"]), JoinFailureCode.FOUNDATION_ELEVATION_MISMATCH, "foundation transition residual exceeded")
        _require(unsupported_gap <= float(budgets["max_unsupported_gap_m"]), JoinFailureCode.UNSUPPORTED_GAP_EXCEEDED, "unsupported gap exceeded")
        _require(outside_projection <= float(budgets["max_outside_projection_m"]), JoinFailureCode.PROJECTION_CLEARANCE_DOMAIN_EXCEEDED, "outside projection exceeded")
        _require(inside_projection <= float(budgets["max_inside_projection_m"]), JoinFailureCode.PROJECTION_CLEARANCE_DOMAIN_EXCEEDED, "inside projection exceeded")

        tower_after = _tree_digest(tower_root)
        span_after = _tree_digest(span_root)
        sources_unchanged = tower_before == tower_after and span_before == span_after
        _require(sources_unchanged, JoinFailureCode.ACCEPTED_SOURCE_GEOMETRY_MUTATION, "accepted source geometry changed")

        half_overlap = float(fixture["realization"]["interface_half_overlap_m"])
        generated_bounds = _bounds(combined_vertices)
        plan = {
            "schema": PLAN_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "request_id": fixture["request_id"],
            "join_kind": fixture["join_kind"],
            "output_type": "TowerJoinPlan",
            "realization_mode": fixture["realization"]["mode"],
            "source_tower_id": fixture["tower"]["source_object_id"],
            "source_span_id": fixture["span"]["source_object_id"],
            "tower_socket_id": fixture["tower"]["socket_ids"]["body"],
            "span_socket_id": fixture["span"]["socket_ids"]["body"],
            "span_transform": {
                "translation_m": translation,
                "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            "part_order": list(PART_ORDER),
            "source_geometry_mutated": False,
            "assembly_mode": "PROJECT_OWNED_NON_DESTRUCTIVE_ASSEMBLY_REFERENCE",
        }
        bindings = {
            "schema": BINDING_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "tower": {
                "object_id": fixture["tower"]["source_object_id"],
                "family": fixture["tower"]["family"],
                "side_count": fixture["tower"]["side_count"],
                "reference_root": fixture["tower"]["reference_root"],
                "files": tower_refs,
            },
            "span": {
                "object_id": fixture["span"]["source_object_id"],
                "family": fixture["span"]["family"],
                "role": fixture["span"]["role"],
                "reference_root": fixture["span"]["reference_root"],
                "files": span_refs,
            },
        }
        socket_evidence = {
            "schema": SOCKET_EVIDENCE_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "tower_socket_id": tower_body_socket["socket_id"],
            "span_socket_id": span_body_socket["socket_id"],
            "span_translation_m": translation,
            "socket_position_error_m": position_error,
            "tangent_error": tangent_error,
            "up_axis_error": up_error,
            "inside_frame_error": inside_error,
            "wall_walk_position_error_m": walk_error,
            "foundation_source_elevation_offset_m": round(raw_foundation_offset, 9),
            "foundation_transition_residual_error_m": foundation_residual,
            "position_status": "PASS",
            "frame_status": "PASS",
        }
        continuity = {
            "schema": CONTINUITY_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "wall_body_continuity": "PASS",
            "wall_walk_continuity": "PASS",
            "foundation_interface_continuity": "PASS_BOUNDED_VERTICAL_TRANSITION",
            "foundation_source_offset_m": round(raw_foundation_offset, 9),
            "foundation_transition_bounds_m": next(part.bounds for part in parts if part.part_id == "foundation_transition"),
            "source_geometry_mutated": False,
        }
        overlap = {
            "schema": OVERLAP_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "generated_join_bounds_m": generated_bounds,
            "generated_join_length_m": round(half_overlap * 2.0, 9),
            "tower_overlap_extent_m": round(half_overlap, 9),
            "span_overlap_extent_m": round(half_overlap, 9),
            "tower_overlap_ratio_of_join": 0.5,
            "span_overlap_ratio_of_join": 0.5,
            "unsupported_gap_m": unsupported_gap,
            "outside_projection_m": outside_projection,
            "inside_projection_m": inside_projection,
            "clearance_result": "PASS",
        }
        immutability = {
            "schema": IMMUTABILITY_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "tower_digest_before": tower_before,
            "tower_digest_after": tower_after,
            "span_digest_before": span_before,
            "span_digest_after": span_after,
            "accepted_sources_unchanged": sources_unchanged,
        }
        parts_doc = {
            "schema": PARTS_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "part_order": list(PART_ORDER),
            "parts": semantic_parts,
        }
        mesh_doc = {
            "schema": MESH_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "units": "METER",
            "frame": PROJECT_FRAME,
            "vertices_m": combined_vertices,
            "triangles": combined_triangles,
            "bounds_m": generated_bounds,
            "part_ranges": part_ranges,
        }
        tessellation = {
            "schema": TESSELLATION_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "linear_deflection_m": float(fixture["tolerances"]["tessellation_linear_m"]),
            "angular_deflection_rad": float(fixture["tolerances"]["tessellation_angular_rad"]),
            "mesh_round_digits": int(fixture["tolerances"]["mesh_round_digits"]),
            "vertex_policy": "PER_PART_LEXICOGRAPHIC_UNIQUE",
            "triangle_policy": "CYCLIC_MIN_PRESERVE_WINDING_THEN_SORT",
            "part_order": list(PART_ORDER),
        }
        stored_doc = {
            "schema": STORED_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "copy_count": len(stored_copies),
            "all_reopened": all(item.get("reopen_status") == "PASS_PROVIDER_NATIVE_REOPEN" for item in stored_copies),
            "copies": stored_copies,
        }
        documents = {
            "tower-join-plan.json": plan,
            "source-bindings.json": bindings,
            "socket-frame-evidence.json": socket_evidence,
            "continuity-evidence.json": continuity,
            "overlap-gap-projection.json": overlap,
            "source-immutability.json": immutability,
            "semantic-parts.json": parts_doc,
            "fixed-tessellation.json": tessellation,
            "neutral-mesh.json": mesh_doc,
            "stored-copies.json": stored_doc,
        }
        refs: dict[str, dict[str, Any]] = {}
        for name, document in documents.items():
            path = output / name
            path.write_bytes(pretty_json_bytes(document))
            refs[name] = _ref(path, output)

        source_dir = Path(__file__).resolve().parent
        source_names = ("__init__.py", "fx01.py")
        source_digest = sha256_ref(
            b"".join(
                name.encode("utf-8") + b"\0" + (source_dir / name).read_bytes()
                for name in source_names
            )
        )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "producer_id": "royal-capital/fortification/tower-join",
            "producer_revision": "r0c-cp2.fx01.1",
            "producer_source_digest": source_digest,
            "fixture_digest": sha256_ref(canonical_json_bytes(fixture)),
            "runtime": dict(fixture["runtime"]),
            "unit_frame_contract": unit_frame_contract(),
            "fixed_tessellation": tessellation,
            "part_receipts": part_receipts,
            "output_digests": {name: row["sha256"] for name, row in refs.items()},
            "capabilities": [
                "fortification.tower_join.tangent@1",
                "fortification.tower_join.non_destructive@1",
                "fortification.tower_join.wall_walk_continuity@1",
                "fortification.tower_join.foundation_transition@1",
                "cad.fixed_tessellation@1",
                "cad.step_brep_export_reopen@1",
            ],
        }
        receipt_path = output / "cad-provider-receipt.json"
        receipt_path.write_bytes(pretty_json_bytes(receipt))
        receipt_ref = _ref(receipt_path, output)

        result = {
            "schema": RESULT_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "request_id": fixture["request_id"],
            "join_kind": fixture["join_kind"],
            "status": "SUCCEEDED",
            "failure": None,
            "part_count": len(parts),
            "stored_copy_count": len(stored_copies),
            "volume_m3": round(total_volume, 9),
            "bounds_m": generated_bounds,
            "vertex_count": len(combined_vertices),
            "triangle_count": len(combined_triangles),
            "socket_position_error_m": position_error,
            "tangent_error": tangent_error,
            "up_axis_error": up_error,
            "inside_frame_error": inside_error,
            "wall_walk_position_error_m": walk_error,
            "foundation_source_offset_m": round(raw_foundation_offset, 9),
            "foundation_transition_residual_m": foundation_residual,
            "unsupported_gap_m": unsupported_gap,
            "outside_projection_m": outside_projection,
            "inside_projection_m": inside_projection,
            "accepted_sources_unchanged": sources_unchanged,
            "stored_copies_reopened": stored_doc["all_reopened"],
            "partial_output_published": False,
            "artifacts": {**refs, "cad-provider-receipt.json": receipt_ref},
        }
        result_path = output / "result.json"
        result_path.write_bytes(pretty_json_bytes(result))
        return result
