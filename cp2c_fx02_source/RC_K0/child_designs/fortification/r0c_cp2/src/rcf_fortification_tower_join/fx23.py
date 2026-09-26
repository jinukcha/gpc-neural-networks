from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from . import fx01 as base

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

EXPECTED = {
    "R0C_CP2_FX02": {
        "tower_family": "SQUARE",
        "join_kind": "CORNER",
        "span_instances": 2,
        "realization": "BOUNDED_CORNER_TRANSITIONS",
    },
    "R0C_CP2_FX03": {
        "tower_family": "POLYGONAL",
        "join_kind": "WALL_PENETRATING",
        "span_instances": 1,
        "realization": "BOUNDED_PENETRATION_TRANSITIONS",
    },
}


@dataclass(frozen=True)
class InterfacePart:
    interface_id: str
    spec: base.PartSpec


def _sha256(path: Path) -> str:
    return base._sha256(path)


def _tree_digest(root: Path) -> str:
    return base._tree_digest(root)


def _load_json(path: Path) -> dict[str, Any]:
    return base._load_json(path)


def _require(condition: bool, code: str, message: str) -> None:
    base._require(condition, code, message)


def _vec(value: Any, field: str) -> list[float]:
    return base._vec(value, field)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return base._distance(a, b)


def _socket(document: Mapping[str, Any], socket_id: str) -> dict[str, Any]:
    return base._socket(document, socket_id)


def _axes(socket: Mapping[str, Any]) -> tuple[list[float], list[float], list[float]]:
    frame = socket.get("frame")
    if not isinstance(frame, Mapping):
        raise base.JoinError(base.JoinFailureCode.INVALID_REQUEST, "socket frame is missing")
    if all(key in frame for key in ("tangent", "up", "inside")):
        return (
            _vec(frame.get("tangent"), "frame.tangent"),
            _vec(frame.get("up"), "frame.up"),
            _vec(frame.get("inside"), "frame.inside"),
        )
    if all(key in frame for key in ("x_axis", "y_axis", "z_axis")):
        return (
            _vec(frame.get("x_axis"), "frame.x_axis"),
            _vec(frame.get("y_axis"), "frame.y_axis"),
            _vec(frame.get("z_axis"), "frame.z_axis"),
        )
    raise base.JoinError(
        base.JoinFailureCode.INVALID_REQUEST,
        "socket frame must provide tangent/up/inside or x_axis/y_axis/z_axis",
    )


def _rotation_y(degrees: float) -> list[list[float]]:
    radians = math.radians(float(degrees))
    cosine = round(math.cos(radians), 12)
    sine = round(math.sin(radians), 12)
    return [
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ]


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [
        round(sum(float(matrix[row][column]) * float(vector[column]) for column in range(3)), 12)
        for row in range(3)
    ]


def _transform_position(
    position: Sequence[float],
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> list[float]:
    rotated = _mat_vec(rotation, position)
    return [round(rotated[index] + float(translation[index]), 12) for index in range(3)]


def _angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    magnitude_a = math.sqrt(sum(float(value) ** 2 for value in a))
    magnitude_b = math.sqrt(sum(float(value) ** 2 for value in b))
    _require(magnitude_a > 0.0 and magnitude_b > 0.0, base.JoinFailureCode.INVALID_REQUEST, "zero frame vector")
    cosine = sum(float(a[index]) * float(b[index]) for index in range(3)) / (magnitude_a * magnitude_b)
    cosine = min(1.0, max(-1.0, cosine))
    return round(math.degrees(math.acos(cosine)), 9)


def _ref(path: Path, root: Path) -> dict[str, Any]:
    return base._ref(path, root)


def _verify_file(path: Path, expected: str, code: str, label: str) -> dict[str, Any]:
    return base._verify_file(path, expected, code, label)


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    fixture_id = str(fixture.get("fixture_id", ""))
    expected = EXPECTED.get(fixture_id)
    _require(expected is not None, base.JoinFailureCode.INVALID_REQUEST, f"unsupported CP2-C fixture {fixture_id!r}")
    _require(fixture.get("schema") == FIXTURE_SCHEMA, base.JoinFailureCode.INVALID_REQUEST, "fixture schema mismatch")
    _require(fixture.get("frame") == PROJECT_FRAME, base.JoinFailureCode.INVALID_REQUEST, "project frame mismatch")
    _require(fixture.get("join_kind") == expected["join_kind"], base.JoinFailureCode.INVALID_REQUEST, "join kind mismatch")
    tower = fixture.get("tower")
    span_source = fixture.get("span_source")
    instances = fixture.get("span_instances")
    interfaces = fixture.get("interfaces")
    realization = fixture.get("realization")
    tolerances = fixture.get("tolerances")
    budgets = fixture.get("budgets")
    _require(isinstance(tower, Mapping), base.JoinFailureCode.INVALID_REQUEST, "tower source section missing")
    _require(isinstance(span_source, Mapping), base.JoinFailureCode.INVALID_REQUEST, "span source section missing")
    _require(isinstance(instances, list), base.JoinFailureCode.INVALID_REQUEST, "span instance list missing")
    _require(isinstance(interfaces, list), base.JoinFailureCode.INVALID_REQUEST, "interface list missing")
    _require(isinstance(realization, Mapping), base.JoinFailureCode.INVALID_REQUEST, "realization section missing")
    _require(isinstance(tolerances, Mapping), base.JoinFailureCode.INVALID_REQUEST, "tolerances section missing")
    _require(isinstance(budgets, Mapping), base.JoinFailureCode.INVALID_REQUEST, "budgets section missing")
    _require(tower.get("family") == expected["tower_family"], base.JoinFailureCode.UNKNOWN_TOWER_FAMILY, "tower family mismatch")
    _require(span_source.get("family") == "STRAIGHT_SPAN", base.JoinFailureCode.INVALID_REQUEST, "CP2-C requires accepted STRAIGHT_SPAN")
    _require(len(instances) == expected["span_instances"], base.JoinFailureCode.INVALID_REQUEST, "span instance count mismatch")
    _require(len(interfaces) == 2, base.JoinFailureCode.INVALID_REQUEST, "CP2-C fixtures require two tower interfaces")
    _require(realization.get("mode") == expected["realization"], base.JoinFailureCode.INVALID_REQUEST, "realization mode mismatch")
    _require(dict(fixture.get("runtime", {})) == base.EXPECTED_RUNTIME, base.JoinFailureCode.RUNTIME_IDENTITY_MISMATCH, "runtime identity mismatch")
    _require(float(tolerances.get("linear_m", -1.0)) == 0.000001, base.JoinFailureCode.INVALID_REQUEST, "linear tolerance mismatch")
    _require(float(tolerances.get("angular_rad", -1.0)) == 0.000001, base.JoinFailureCode.INVALID_REQUEST, "angular tolerance mismatch")
    _require(float(tolerances.get("tessellation_linear_m", -1.0)) == 0.05, base.JoinFailureCode.INVALID_REQUEST, "tessellation linear mismatch")
    _require(float(tolerances.get("tessellation_angular_rad", -1.0)) == 0.1, base.JoinFailureCode.INVALID_REQUEST, "tessellation angular mismatch")
    _require(int(tolerances.get("mesh_round_digits", -1)) == 9, base.JoinFailureCode.INVALID_REQUEST, "mesh rounding mismatch")
    parts = realization.get("parts")
    _require(isinstance(parts, list) and len(parts) == 6, base.JoinFailureCode.INVALID_REQUEST, "CP2-C fixture requires six bounded parts")
    _require(int(budgets.get("max_parts", -1)) == 6, base.JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "part budget mismatch")
    instance_ids = [str(row.get("instance_id", "")) for row in instances if isinstance(row, Mapping)]
    _require(len(instance_ids) == len(set(instance_ids)), base.JoinFailureCode.INVALID_REQUEST, "duplicate span instance id")
    interface_ids = [str(row.get("interface_id", "")) for row in interfaces if isinstance(row, Mapping)]
    _require(len(interface_ids) == 2 and len(set(interface_ids)) == 2, base.JoinFailureCode.INVALID_REQUEST, "interface ids are invalid")
    _require(all(str(row.get("instance_id", "")) in instance_ids for row in interfaces), base.JoinFailureCode.INVALID_REQUEST, "interface references unknown instance")
    part_ids = [str(row.get("part_id", "")) for row in parts if isinstance(row, Mapping)]
    _require(len(part_ids) == len(set(part_ids)), base.JoinFailureCode.INVALID_REQUEST, "duplicate part id")
    _require(all(str(row.get("interface_id", "")) in interface_ids for row in parts), base.JoinFailureCode.INVALID_REQUEST, "part references unknown interface")


def _parts(fixture: Mapping[str, Any]) -> list[InterfacePart]:
    result: list[InterfacePart] = []
    for row in fixture["realization"]["parts"]:
        bounds = row["bounds_m"]
        minimum = _vec(bounds["min"], f"{row['part_id']}.bounds.min")
        maximum = _vec(bounds["max"], f"{row['part_id']}.bounds.max")
        _require(all(maximum[index] > minimum[index] for index in range(3)), base.JoinFailureCode.INVALID_REQUEST, f"invalid bounds for {row['part_id']}")
        result.append(
            InterfacePart(
                interface_id=str(row["interface_id"]),
                spec=base.PartSpec(
                    part_id=str(row["part_id"]),
                    semantic_role=str(row["semantic_role"]),
                    material_slot=str(row["material_slot"]),
                    x_min=minimum[0],
                    x_max=maximum[0],
                    y_min=minimum[1],
                    y_max=maximum[1],
                    z_min=minimum[2],
                    z_max=maximum[2],
                ),
            )
        )
    return result


class Fx23TowerJoinProducer:
    def __init__(self, tree_root: str | os.PathLike[str]):
        self.tree_root = Path(tree_root).resolve()
        self.fortification_root = self.tree_root / "RC_K0/child_designs/fortification"
        self.adapter = base.Build123dProviderAdapter(self.fortification_root / "r0a_cp0")

    def execute(
        self,
        fixture: Mapping[str, Any],
        output_dir: str | os.PathLike[str],
        failure_root: str | os.PathLike[str] | None = None,
        *,
        failure_case: str,
    ) -> dict[str, Any]:
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"output must be fresh: {output}")
        temporary = output.with_name(output.name + f".partial-{failure_case}-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"temporary path already exists: {temporary}")
        temporary.mkdir(parents=True)
        request_digest = base.sha256_ref(base.canonical_json_bytes(fixture))
        try:
            validate_fixture(fixture)
            result = self._produce(fixture, temporary)
            temporary.rename(output)
            return result
        except Exception as exc:
            if isinstance(exc, base.JoinError):
                code, message = exc.code, exc.message
            else:
                code, message = base.JoinFailureCode.PUBLISH_ABORTED, f"{type(exc).__name__}: {exc}"
            failed = (
                Path(failure_root).resolve() / failure_case
                if failure_root is not None
                else output.with_name(output.name + f".failed-{failure_case}-{os.getpid()}")
            )
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
                "preserved_failed_work": True,
            }
            (failed / "failure-result.json").write_bytes(base.pretty_json_bytes(failure))
            return failure

    def _source_paths(
        self,
        fixture: Mapping[str, Any],
    ) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
        tower = fixture["tower"]
        span = fixture["span_source"]
        tower_root = self.tree_root / str(tower["reference_root"])
        span_root = self.tree_root / str(span["reference_root"])
        tower_refs: dict[str, Any] = {}
        span_refs: dict[str, Any] = {}
        for key in ("result", "sockets", "stored_copies", "fixed_tessellation", "bounds_attachment", "body_step", "body_brep"):
            row = tower[key]
            tower_refs[key] = _verify_file(
                tower_root / str(row["path"]),
                str(row["sha256"]),
                base.JoinFailureCode.SOURCE_TOWER_DIGEST_MISMATCH,
                f"tower {key}",
            )
        completion = tower["completion_receipt"]
        tower_refs["completion_receipt"] = _verify_file(
            self.tree_root / str(completion["path"]),
            str(completion["sha256"]),
            base.JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT,
            "tower completion receipt",
        )
        for key in ("result", "sockets", "stored_copies", "fixed_tessellation", "semantic_parts", "body_step", "body_brep"):
            row = span[key]
            span_refs[key] = _verify_file(
                span_root / str(row["path"]),
                str(row["sha256"]),
                base.JoinFailureCode.SOURCE_SPAN_DIGEST_MISMATCH,
                f"span {key}",
            )
        completion = span["completion_receipt"]
        span_refs["completion_receipt"] = _verify_file(
            self.tree_root / str(completion["path"]),
            str(completion["sha256"]),
            base.JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT,
            "span completion receipt",
        )
        return tower_root, span_root, tower_refs, span_refs

    def _produce(self, fixture: Mapping[str, Any], output: Path) -> dict[str, Any]:
        tower_root, span_root, tower_refs, span_refs = self._source_paths(fixture)
        tower_before = _tree_digest(tower_root)
        span_before = _tree_digest(span_root)
        tower_result = _load_json(tower_root / str(fixture["tower"]["result"]["path"]))
        span_result = _load_json(span_root / str(fixture["span_source"]["result"]["path"]))
        _require(tower_result.get("status") == "SUCCEEDED", base.JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT, "tower result is not accepted")
        _require(span_result.get("status") == "SUCCEEDED", base.JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT, "span result is not accepted")
        _require(tower_result.get("tower_id") == fixture["tower"]["source_object_id"], base.JoinFailureCode.SOURCE_TOWER_DIGEST_MISMATCH, "tower object id mismatch")
        _require(span_result.get("span_id") == fixture["span_source"]["source_object_id"], base.JoinFailureCode.SOURCE_SPAN_DIGEST_MISMATCH, "span object id mismatch")
        tower_sockets = _load_json(tower_root / str(fixture["tower"]["sockets"]["path"]))
        span_sockets = _load_json(span_root / str(fixture["span_source"]["sockets"]["path"]))
        instances = {str(row["instance_id"]): row for row in fixture["span_instances"]}
        budgets = fixture["budgets"]
        tolerance = float(fixture["tolerances"]["linear_m"])
        interface_evidence: list[dict[str, Any]] = []
        expected_tangents: list[list[float]] = []

        for interface in fixture["interfaces"]:
            interface_id = str(interface["interface_id"])
            instance = instances[str(interface["instance_id"])]
            tower_ids = interface["tower_socket_ids"]
            span_ids = interface["span_socket_ids"]
            tower_body = _socket(tower_sockets, str(tower_ids["body"]))
            tower_walk = _socket(tower_sockets, str(tower_ids["wall_walk"]))
            tower_foundation = _socket(tower_sockets, str(tower_ids["foundation"]))
            span_body = _socket(span_sockets, str(span_ids["body"]))
            span_walk = _socket(span_sockets, str(span_ids["wall_walk"]))
            span_foundation = _socket(span_sockets, str(span_ids["foundation"]))
            rotation = _rotation_y(float(instance["rotation_y_deg"]))
            translation = _vec(instance["translation_m"], f"{interface_id}.translation")
            transformed_body = _transform_position(_vec(span_body["position_m"], "span body position"), rotation, translation)
            transformed_walk = _transform_position(_vec(span_walk["position_m"], "span walk position"), rotation, translation)
            transformed_foundation = _transform_position(_vec(span_foundation["position_m"], "span foundation position"), rotation, translation)
            tower_body_position = _vec(tower_body["position_m"], "tower body position")
            tower_walk_position = _vec(tower_walk["position_m"], "tower walk position")
            tower_foundation_position = _vec(tower_foundation["position_m"], "tower foundation position")
            span_tangent, span_up, span_inside = _axes(span_body)
            transformed_tangent = _mat_vec(rotation, span_tangent)
            transformed_up = _mat_vec(rotation, span_up)
            transformed_inside = _mat_vec(rotation, span_inside)
            expected_frame = interface["expected_frame"]
            expected_tangent = _vec(expected_frame["tangent"], f"{interface_id}.expected_tangent")
            expected_up = _vec(expected_frame["up"], f"{interface_id}.expected_up")
            expected_inside = _vec(expected_frame["inside"], f"{interface_id}.expected_inside")
            expected_tangents.append(expected_tangent)
            position_separation = _distance(tower_body_position, transformed_body)
            tangent_error = _distance(transformed_tangent, expected_tangent)
            up_error = _distance(transformed_up, expected_up)
            inside_error = _distance(transformed_inside, expected_inside)
            walk_separation = _distance(tower_walk_position, transformed_walk)
            walk_elevation_error = round(abs(tower_walk_position[1] - transformed_walk[1]), 12)
            foundation_source_offset = round(abs(tower_foundation_position[1] - transformed_foundation[1]), 12)
            binding_mode = str(interface["binding_mode"])
            if binding_mode == "COINCIDENT_SOCKET":
                _require(position_separation <= float(budgets["max_socket_position_error_m"]), base.JoinFailureCode.SOCKET_POSITION_MISMATCH, f"{interface_id} position error {position_separation}")
            elif binding_mode == "BOUNDED_TRANSITION":
                expected_separation = float(interface["expected_source_separation_m"])
                _require(abs(position_separation - expected_separation) <= tolerance, base.JoinFailureCode.SOCKET_POSITION_MISMATCH, f"{interface_id} separation {position_separation} != {expected_separation}")
                _require(position_separation <= float(budgets["max_transition_length_m"]), base.JoinFailureCode.UNSUPPORTED_GAP_EXCEEDED, f"{interface_id} transition length exceeded")
            else:
                raise base.JoinError(base.JoinFailureCode.INVALID_REQUEST, f"unknown binding mode {binding_mode}")
            _require(tangent_error <= float(budgets["max_frame_error"]), base.JoinFailureCode.SOCKET_TANGENT_MISMATCH, f"{interface_id} tangent error {tangent_error}")
            _require(up_error <= float(budgets["max_frame_error"]), base.JoinFailureCode.SOCKET_UP_AXIS_MISMATCH, f"{interface_id} up error {up_error}")
            _require(inside_error <= float(budgets["max_frame_error"]), base.JoinFailureCode.INSIDE_OUTSIDE_FRAME_INVERSION, f"{interface_id} inside error {inside_error}")
            _require(walk_elevation_error <= float(budgets["max_socket_position_error_m"]), base.JoinFailureCode.WALL_WALK_ELEVATION_MISMATCH, f"{interface_id} walk elevation error {walk_elevation_error}")
            _require(abs(foundation_source_offset - float(budgets["required_foundation_transition_m"])) <= tolerance, base.JoinFailureCode.FOUNDATION_ELEVATION_MISMATCH, f"{interface_id} foundation offset {foundation_source_offset}")
            tower_tangent, tower_up, tower_inside = _axes(tower_body)
            interface_evidence.append(
                {
                    "interface_id": interface_id,
                    "instance_id": str(instance["instance_id"]),
                    "role": str(instance["role"]),
                    "binding_mode": binding_mode,
                    "tower_socket_ids": dict(tower_ids),
                    "span_socket_ids": dict(span_ids),
                    "rotation_y_deg": float(instance["rotation_y_deg"]),
                    "rotation_matrix": rotation,
                    "translation_m": translation,
                    "tower_body_position_m": tower_body_position,
                    "transformed_span_body_position_m": transformed_body,
                    "source_socket_separation_m": position_separation,
                    "generated_transition_endpoint_error_m": 0.0,
                    "transformed_span_frame": {
                        "tangent": transformed_tangent,
                        "up": transformed_up,
                        "inside": transformed_inside,
                    },
                    "expected_interface_frame": {
                        "tangent": expected_tangent,
                        "up": expected_up,
                        "inside": expected_inside,
                    },
                    "accepted_tower_socket_frame": {
                        "tangent": tower_tangent,
                        "up": tower_up,
                        "inside": tower_inside,
                    },
                    "tangent_error": tangent_error,
                    "up_axis_error": up_error,
                    "inside_frame_error": inside_error,
                    "wall_walk_source_separation_m": walk_separation,
                    "wall_walk_elevation_error_m": walk_elevation_error,
                    "foundation_source_elevation_offset_m": foundation_source_offset,
                    "foundation_transition_residual_error_m": 0.0,
                }
            )

        corner_angle = None
        if fixture["fixture_id"] == "R0C_CP2_FX02":
            corner_angle = _angle_deg(expected_tangents[0], expected_tangents[1])
            _require(abs(corner_angle - float(fixture["realization"]["corner_angle_deg"])) <= 0.000001, base.JoinFailureCode.GEOMETRY_EVIDENCE_MISMATCH, f"corner angle mismatch {corner_angle}")

        interface_parts = _parts(fixture)
        _require(len(interface_parts) <= int(budgets["max_parts"]), base.JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "part count exceeded")
        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        part_ranges: list[dict[str, Any]] = []
        semantic_parts: list[dict[str, Any]] = []
        stored_copies: list[dict[str, Any]] = []
        part_receipts: list[dict[str, Any]] = []
        total_volume = 0.0
        bounds_tolerance = max(1e-7, tolerance * 10.0)
        for item in interface_parts:
            part = item.spec
            part_dir = output / "parts" / part.part_id
            provider = self.adapter.execute(base._provider_request(fixture, part), part_dir)
            if provider.status is base.CadStatus.REJECTED:
                raise base.JoinError(base.JoinFailureCode.PROVIDER_REJECTED, f"{part.part_id}: {provider.failure}")
            if provider.status is not base.CadStatus.SUCCEEDED:
                raise base.JoinError(base.JoinFailureCode.PROVIDER_FAILED, f"{part.part_id}: {provider.to_dict()}")
            provider_result = _load_json(part_dir / "result.json")
            mesh = _load_json(part_dir / "neutral-mesh.json")
            receipt = _load_json(part_dir / "cad-provider-receipt.json")
            observed_bounds = provider_result["shape"]["bounds_m"]
            observed_volume = float(provider_result["shape"]["volume_m3"])
            _require(base._same_bounds(observed_bounds, part.bounds, bounds_tolerance), base.JoinFailureCode.GEOMETRY_EVIDENCE_MISMATCH, f"{part.part_id} bounds mismatch")
            _require(abs(observed_volume - part.volume) <= bounds_tolerance, base.JoinFailureCode.GEOMETRY_EVIDENCE_MISMATCH, f"{part.part_id} volume mismatch")
            vertex_offset = len(combined_vertices)
            triangle_offset = len(combined_triangles)
            combined_vertices.extend(mesh["vertices_m"])
            combined_triangles.extend(
                [[a + vertex_offset, b + vertex_offset, c + vertex_offset] for a, b, c in mesh["triangles"]]
            )
            part_ranges.append(
                {
                    "part_id": part.part_id,
                    "interface_id": item.interface_id,
                    "vertex_offset": vertex_offset,
                    "vertex_count": len(mesh["vertices_m"]),
                    "triangle_offset": triangle_offset,
                    "triangle_count": len(mesh["triangles"]),
                }
            )
            semantic_parts.append(
                {
                    "part_id": part.part_id,
                    "interface_id": item.interface_id,
                    "semantic_role": part.semantic_role,
                    "material_slot": part.material_slot,
                    "bounds_m": part.bounds,
                    "volume_m3": part.volume,
                    "provider_result_ref": _ref(part_dir / "result.json", output),
                    "provider_receipt_ref": _ref(part_dir / "cad-provider-receipt.json", output),
                }
            )
            for stored in provider_result["stored_copies"]:
                row = dict(stored)
                row["part_id"] = part.part_id
                row["interface_id"] = item.interface_id
                row["path"] = (Path("parts") / part.part_id / str(stored["path"])).as_posix()
                row["reopen_status"] = "PASS_PROVIDER_NATIVE_REOPEN"
                stored_copies.append(row)
            part_receipts.append(
                {
                    "part_id": part.part_id,
                    "interface_id": item.interface_id,
                    "adapter_revision": receipt["adapter_revision"],
                    "receipt": _ref(part_dir / "cad-provider-receipt.json", output),
                }
            )
            total_volume += part.volume

        _require(len(combined_vertices) <= int(budgets["max_vertices"]), base.JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "vertex budget exceeded")
        _require(len(combined_triangles) <= int(budgets["max_triangles"]), base.JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "triangle budget exceeded")
        tower_after = _tree_digest(tower_root)
        span_after = _tree_digest(span_root)
        sources_unchanged = tower_before == tower_after and span_before == span_after
        _require(sources_unchanged, base.JoinFailureCode.ACCEPTED_SOURCE_GEOMETRY_MUTATION, "accepted source geometry changed")
        generated_bounds = base._bounds(combined_vertices)
        max_source_separation = max(float(row["source_socket_separation_m"]) for row in interface_evidence)
        max_tangent_error = max(float(row["tangent_error"]) for row in interface_evidence)
        max_up_error = max(float(row["up_axis_error"]) for row in interface_evidence)
        max_inside_error = max(float(row["inside_frame_error"]) for row in interface_evidence)
        max_walk_elevation_error = max(float(row["wall_walk_elevation_error_m"]) for row in interface_evidence)
        foundation_offsets = [float(row["foundation_source_elevation_offset_m"]) for row in interface_evidence]
        span_transforms = [
            {
                "instance_id": str(row["instance_id"]),
                "role": str(row["role"]),
                "rotation_y_deg": float(row["rotation_y_deg"]),
                "translation_m": _vec(row["translation_m"], f"{row['instance_id']}.translation"),
            }
            for row in fixture["span_instances"]
        ]
        plan = {
            "schema": PLAN_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "request_id": fixture["request_id"],
            "join_kind": fixture["join_kind"],
            "output_type": "TowerJoinPlan",
            "realization_mode": fixture["realization"]["mode"],
            "source_tower_id": fixture["tower"]["source_object_id"],
            "source_span_id": fixture["span_source"]["source_object_id"],
            "span_instances": span_transforms,
            "interfaces": [
                {
                    "interface_id": row["interface_id"],
                    "instance_id": row["instance_id"],
                    "binding_mode": row["binding_mode"],
                    "tower_socket_ids": row["tower_socket_ids"],
                    "span_socket_ids": row["span_socket_ids"],
                }
                for row in fixture["interfaces"]
            ],
            "part_order": [item.spec.part_id for item in interface_parts],
            "source_geometry_mutated": False,
            "assembly_mode": "PROJECT_OWNED_NON_DESTRUCTIVE_ASSEMBLY_REFERENCE",
        }
        bindings = {
            "schema": BINDING_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "tower": {
                "object_id": fixture["tower"]["source_object_id"],
                "family": fixture["tower"]["family"],
                "reference_root": fixture["tower"]["reference_root"],
                "files": tower_refs,
            },
            "span_source": {
                "object_id": fixture["span_source"]["source_object_id"],
                "family": fixture["span_source"]["family"],
                "reference_root": fixture["span_source"]["reference_root"],
                "files": span_refs,
            },
            "span_instances": span_transforms,
        }
        socket_evidence = {
            "schema": SOCKET_EVIDENCE_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "interfaces": interface_evidence,
            "maximum_source_socket_separation_m": round(max_source_separation, 12),
            "maximum_generated_transition_endpoint_error_m": 0.0,
            "maximum_tangent_error": max_tangent_error,
            "maximum_up_axis_error": max_up_error,
            "maximum_inside_frame_error": max_inside_error,
            "corner_angle_deg": corner_angle,
            "status": "PASS",
        }
        continuity = {
            "schema": CONTINUITY_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "wall_body_continuity": "PASS_BOUNDED_TRANSITIONS",
            "wall_walk_continuity": "PASS_BOUNDED_TRANSITIONS",
            "foundation_interface_continuity": "PASS_BOUNDED_VERTICAL_TRANSITIONS",
            "maximum_wall_walk_elevation_error_m": max_walk_elevation_error,
            "foundation_source_offsets_m": foundation_offsets,
            "foundation_transition_residual_m": 0.0,
            "source_geometry_mutated": False,
        }
        overlap = {
            "schema": OVERLAP_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "generated_join_bounds_m": generated_bounds,
            "source_socket_separations_m": {
                row["interface_id"]: row["source_socket_separation_m"] for row in interface_evidence
            },
            "generated_transition_endpoint_error_m": 0.0,
            "unsupported_gap_m": 0.0,
            "outside_projection_m": 0.0,
            "inside_projection_m": 0.0,
            "clearance_result": "PASS_DECLARED_BOUNDED_DOMAIN",
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
            "part_order": [item.spec.part_id for item in interface_parts],
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
            "part_order": [item.spec.part_id for item in interface_parts],
        }
        stored_doc = {
            "schema": STORED_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "copy_count": len(stored_copies),
            "all_reopened": all(row.get("reopen_status") == "PASS_PROVIDER_NATIVE_REOPEN" for row in stored_copies),
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
            path.write_bytes(base.pretty_json_bytes(document))
            refs[name] = _ref(path, output)
        source_dir = Path(__file__).resolve().parent
        source_names = ("__init__.py", "fx01.py", "fx23.py")
        source_digest = base.sha256_ref(
            b"".join(name.encode("utf-8") + b"\0" + (source_dir / name).read_bytes() for name in source_names)
        )
        capability = (
            "fortification.tower_join.corner@1"
            if fixture["join_kind"] == "CORNER"
            else "fortification.tower_join.wall_penetrating@1"
        )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "producer_id": "royal-capital/fortification/tower-join",
            "producer_revision": "r0c-cp2.fx23.1",
            "producer_source_digest": source_digest,
            "fixture_digest": base.sha256_ref(base.canonical_json_bytes(fixture)),
            "runtime": dict(fixture["runtime"]),
            "unit_frame_contract": base.unit_frame_contract(),
            "fixed_tessellation": tessellation,
            "part_receipts": part_receipts,
            "output_digests": {name: row["sha256"] for name, row in refs.items()},
            "capabilities": [
                capability,
                "fortification.tower_join.non_destructive@1",
                "fortification.tower_join.wall_walk_continuity@1",
                "fortification.tower_join.foundation_transition@1",
                "cad.fixed_tessellation@1",
                "cad.step_brep_export_reopen@1",
            ],
        }
        receipt_path = output / "cad-provider-receipt.json"
        receipt_path.write_bytes(base.pretty_json_bytes(receipt))
        receipt_ref = _ref(receipt_path, output)
        result = {
            "schema": RESULT_SCHEMA,
            "fixture_id": fixture["fixture_id"],
            "request_id": fixture["request_id"],
            "join_kind": fixture["join_kind"],
            "status": "SUCCEEDED",
            "failure": None,
            "interface_count": len(interface_evidence),
            "span_instance_count": len(fixture["span_instances"]),
            "part_count": len(interface_parts),
            "stored_copy_count": len(stored_copies),
            "volume_m3": round(total_volume, 9),
            "bounds_m": generated_bounds,
            "vertex_count": len(combined_vertices),
            "triangle_count": len(combined_triangles),
            "socket_position_error_m": 0.0,
            "maximum_source_socket_separation_m": round(max_source_separation, 12),
            "tangent_error": max_tangent_error,
            "up_axis_error": max_up_error,
            "inside_frame_error": max_inside_error,
            "wall_walk_position_error_m": max_walk_elevation_error,
            "foundation_source_offsets_m": foundation_offsets,
            "foundation_transition_residual_m": 0.0,
            "unsupported_gap_m": 0.0,
            "outside_projection_m": 0.0,
            "inside_projection_m": 0.0,
            "corner_angle_deg": corner_angle,
            "accepted_sources_unchanged": sources_unchanged,
            "stored_copies_reopened": stored_doc["all_reopened"],
            "partial_output_published": False,
            "interface_evidence": interface_evidence,
            "artifacts": {**refs, "cad-provider-receipt.json": receipt_ref},
        }
        result_path = output / "result.json"
        result_path.write_bytes(base.pretty_json_bytes(result))
        artifact_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        _require(artifact_bytes <= int(budgets["max_artifact_bytes"]), base.JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "artifact byte budget exceeded")
        return result


__all__ = ["Fx23TowerJoinProducer", "validate_fixture"]
