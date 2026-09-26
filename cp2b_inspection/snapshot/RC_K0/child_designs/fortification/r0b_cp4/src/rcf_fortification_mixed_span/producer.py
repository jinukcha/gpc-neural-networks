from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .coverage import build_assembly_coverage, build_module_coverage
from .model import (
    ASSEMBLY_PLAN_SCHEMA,
    EXPECTED_RUNTIME,
    MESH_SCHEMA,
    MODULE_INDEX_SCHEMA,
    PROJECT_FRAME,
    RECEIPT_SCHEMA,
    RESULT_SCHEMA,
    SOCKET_ALIGNMENT_SCHEMA,
    TRANSFORM_SCHEMA,
    MixedSpanContractError,
    angle_between,
    canonical_json_bytes,
    canonicalize_segmentation,
    heading_deg,
    pretty_json_bytes,
    rotate_heading,
    sha256_ref,
    sub,
    transform_point,
    transform_vector,
    validate_fixture,
    _round,
    _rv,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _bounds(vertices: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    if not vertices:
        raise MixedSpanContractError("cannot compute bounds of an empty mesh")
    return {
        "min": [_round(min(float(vertex[axis]) for vertex in vertices)) for axis in range(3)],
        "max": [_round(max(float(vertex[axis]) for vertex in vertices)) for axis in range(3)],
    }


def _source_ref(path: Path, fortification_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(fortification_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MixedSpanContractError(f"required source artifact missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MixedSpanContractError(f"source artifact must be an object: {path}")
    return value


def _socket_file(source_dir: Path, module_kind: str) -> Path:
    name = {
        "PATH_SPAN": "sockets.json",
        "JOIN": "join-sockets.json",
        "TERRAIN_SPAN": "interface-sockets.json",
    }[module_kind]
    return source_dir / name


def _socket_lookup(source_dir: Path, module_kind: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = _load_json(_socket_file(source_dir, module_kind))
    rows = document.get("sockets")
    if not isinstance(rows, list):
        raise MixedSpanContractError(f"socket document is malformed: {source_dir}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        socket_id = row.get("socket_id")
        if not isinstance(socket_id, str) or socket_id in by_id:
            raise MixedSpanContractError(f"invalid or duplicate source socket in {source_dir}")
        by_id[socket_id] = row
    return document, by_id


def _transform_socket(socket: Mapping[str, Any], heading: float, translation: Sequence[float]) -> dict[str, Any]:
    frame = socket["frame"]
    transformed_frame = {
        "tangent": _rv(transform_vector(frame["tangent"], heading)),
        "up": _rv(transform_vector(frame["up"], heading)),
        "inside": _rv(transform_vector(frame["inside"], heading)),
        "outside": _rv(transform_vector(frame["outside"], heading)),
        "orientation_determinant": 1.0,
    }
    return {
        "source_socket_id": socket["socket_id"],
        "role": socket["role"],
        "position_m": _rv(transform_point(socket["position_m"], heading, translation)),
        "frame": transformed_frame,
        "required": bool(socket.get("required", True)),
    }


def _alignment_errors(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    position_error = sum((float(a["position_m"][axis]) - float(b["position_m"][axis])) ** 2 for axis in range(3)) ** 0.5
    return {
        "position_m": _round(position_error, 12),
        "tangent_rad": _round(angle_between(a["frame"]["tangent"], b["frame"]["tangent"]), 12),
        "up_rad": _round(angle_between(a["frame"]["up"], b["frame"]["up"]), 12),
        "inside_rad": _round(angle_between(a["frame"]["inside"], b["frame"]["inside"]), 12),
        "outside_rad": _round(angle_between(a["frame"]["outside"], b["frame"]["outside"]), 12),
    }


def _runtime_identity() -> dict[str, str]:
    return {
        "build123d_version": importlib.metadata.version("build123d"),
        "ocp_version": importlib.metadata.version("cadquery-ocp-novtk"),
        "build123d_source_commit": EXPECTED_RUNTIME["build123d_source_commit"],
        "ocp_wheel_sha256": EXPECTED_RUNTIME["ocp_wheel_sha256"],
    }


def _range_metadata(source_dir: Path, module_kind: str, mesh: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if module_kind in {"PATH_SPAN", "JOIN"}:
        parts_document = _load_json(source_dir / "semantic-parts.json")
        by_part = {row["part_id"]: row for row in parts_document["parts"]}
        for row in mesh["part_ranges"]:
            part_id = row["part_id"]
            part = by_part[part_id]
            rows.append({
                "range_id": f"part/{part_id}",
                "part_id": part_id,
                "unit_id": None,
                "segment_id": None,
                "semantic_role": part["semantic_role"],
                "triangle_offset": int(row["triangle_offset"]),
                "triangle_count": int(row["triangle_count"]),
            })
    else:
        unit_document = _load_json(source_dir / "construction-units.json")
        by_unit = {row["unit_id"]: row for row in unit_document["units"]}
        for row in mesh["unit_ranges"]:
            unit_id = row["unit_id"]
            unit = by_unit[unit_id]
            rows.append({
                "range_id": f"unit/{unit_id}",
                "part_id": row["part_id"],
                "unit_id": unit_id,
                "segment_id": row["segment_id"],
                "semantic_role": unit["semantic_role"],
                "triangle_offset": int(row["triangle_offset"]),
                "triangle_count": int(row["triangle_count"]),
            })
    return sorted(rows, key=lambda row: (row["triangle_offset"], row["range_id"]))


def _classification_frames(source_dir: Path, module_kind: str) -> list[dict[str, Any]]:
    if module_kind == "PATH_SPAN":
        return list(_load_json(source_dir / "local-frames.json")["frames"])
    if module_kind == "JOIN":
        return list(_load_json(source_dir / "section-plan.json")["sections"])
    axis = _load_json(source_dir / "terrain-span-plan.json")["axis_contract"]
    return [{
        "origin_m": axis["start_m"],
        "tangent": axis["tangent"],
        "up": axis["up"],
        "inside": axis["inside"],
        "outside": axis["outside"],
        "orientation_determinant": axis["orientation_determinant"],
    }]


def _source_family(result: Mapping[str, Any]) -> str:
    family = result.get("family", result.get("span_family"))
    if not isinstance(family, str):
        raise MixedSpanContractError("source result does not declare a family")
    return family


class MixedSpanAssemblyProducer:
    def __init__(self, fortification_root: str | os.PathLike[str]):
        self.fortification_root = Path(fortification_root).resolve()

    def execute(self, fixture: Mapping[str, Any], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
        spec = validate_fixture(fixture)
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"output must be fresh: {output}")
        temporary = output.with_name(output.name + f".partial-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"temporary output exists: {temporary}")
        temporary.mkdir(parents=True)
        try:
            result = self._produce(spec, temporary)
            temporary.rename(output)
            return result
        except Exception:
            failed = output.with_name(output.name + f".failed-{os.getpid()}")
            if temporary.exists():
                temporary.rename(failed)
            raise

    def _load_module(self, module: Mapping[str, Any]) -> dict[str, Any]:
        source_dir = (self.fortification_root / module["source_dir"]).resolve()
        try:
            source_dir.relative_to(self.fortification_root)
        except ValueError as exc:
            raise MixedSpanContractError(f"source directory escapes fortification root: {source_dir}") from exc
        result_path = source_dir / "result.json"
        mesh_path = source_dir / "neutral-mesh.json"
        receipt_path = source_dir / "cad-provider-receipt.json"
        stored_path = source_dir / "stored-copies.json"
        result = _load_json(result_path)
        mesh = _load_json(mesh_path)
        if _digest(result_path) != module["source_result_sha256"]:
            raise MixedSpanContractError(f"source result digest mismatch for {module['module_id']}")
        if _digest(mesh_path) != module["source_mesh_sha256"]:
            raise MixedSpanContractError(f"source mesh digest mismatch for {module['module_id']}")
        if result.get("status") != "SUCCEEDED":
            raise MixedSpanContractError(f"source module is not accepted: {module['module_id']}")
        if _source_family(result) != module["family"]:
            raise MixedSpanContractError(f"source family mismatch for {module['module_id']}")
        socket_doc, sockets = _socket_lookup(source_dir, module["module_kind"])
        if module["input_socket"] not in sockets or module["output_socket"] not in sockets:
            raise MixedSpanContractError(f"declared source socket is absent for {module['module_id']}")
        source_refs = {
            "source_dir": module["source_dir"],
            "result": _source_ref(result_path, self.fortification_root),
            "neutral_mesh": _source_ref(mesh_path, self.fortification_root),
            "provider_receipt": _source_ref(receipt_path, self.fortification_root),
            "stored_copies": _source_ref(stored_path, self.fortification_root),
            "sockets": _source_ref(_socket_file(source_dir, module["module_kind"]), self.fortification_root),
        }
        return {
            "module": dict(module),
            "source_dir": source_dir,
            "result": result,
            "mesh": mesh,
            "socket_document": socket_doc,
            "sockets": sockets,
            "ranges": _range_metadata(source_dir, module["module_kind"], mesh),
            "frames": _classification_frames(source_dir, module["module_kind"]),
            "source_refs": source_refs,
        }

    def _produce(self, spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
        actual_runtime = _runtime_identity()
        if actual_runtime != EXPECTED_RUNTIME:
            raise MixedSpanContractError(f"runtime identity mismatch: {actual_runtime}")
        segmentation = canonicalize_segmentation(spec)
        module_data = {module["module_id"]: self._load_module(module) for module in spec["modules"]}
        chain = segmentation["chain_order"]
        transforms: dict[str, dict[str, Any]] = {}
        root_transform = spec["root_transform"]
        transforms[chain[0]] = {
            "module_id": chain[0],
            "heading_deg": _round(root_transform["heading_deg"]),
            "translation_m": _rv(root_transform["translation_m"]),
            "source": "ROOT_TRANSFORM",
        }
        connections_by_source = {row["from_module"]: row for row in spec["connections"]}
        alignment_rows: list[dict[str, Any]] = []
        linear_tolerance = float(spec["tolerances"]["linear_m"])
        angular_tolerance = float(spec["tolerances"]["angular_rad"])
        for source_module_id in chain[:-1]:
            connection = connections_by_source[source_module_id]
            target_module_id = connection["to_module"]
            source_data = module_data[source_module_id]
            target_data = module_data[target_module_id]
            source_transform = transforms[source_module_id]
            source_socket = source_data["sockets"][connection["from_socket"]]
            target_socket_local = target_data["sockets"][connection["to_socket"]]
            target_world_socket = _transform_socket(source_socket, source_transform["heading_deg"], source_transform["translation_m"])
            heading_delta = _round(heading_deg(target_world_socket["frame"]["tangent"]) - heading_deg(target_socket_local["frame"]["tangent"]), 9)
            rotated_input_position = rotate_heading(target_socket_local["position_m"], heading_delta)
            translation = sub(target_world_socket["position_m"], rotated_input_position)
            transforms[target_module_id] = {
                "module_id": target_module_id,
                "heading_deg": heading_delta,
                "translation_m": _rv(translation),
                "source": "SOCKET_ALIGNMENT",
                "aligned_from_module": source_module_id,
                "aligned_from_socket": connection["from_socket"],
                "aligned_to_socket": connection["to_socket"],
            }
            transformed_target_socket = _transform_socket(target_socket_local, heading_delta, translation)
            errors = _alignment_errors(target_world_socket, transformed_target_socket)
            status = "PASS" if errors["position_m"] <= linear_tolerance and max(errors[key] for key in ("tangent_rad", "up_rad", "inside_rad", "outside_rad")) <= angular_tolerance else "FAIL"
            if status != "PASS":
                raise MixedSpanContractError(f"socket alignment failed for {source_module_id}->{target_module_id}: {errors}")
            if target_world_socket["role"] != transformed_target_socket["role"]:
                raise MixedSpanContractError(f"socket role mismatch for {source_module_id}->{target_module_id}")
            alignment_rows.append({
                "connection_index": len(alignment_rows),
                **connection,
                "station_m": next(module["station"].get("end_m", module["station"].get("anchor_m")) for module in spec["modules"] if module["module_id"] == source_module_id),
                "source_world_socket": target_world_socket,
                "target_world_socket": transformed_target_socket,
                "errors": errors,
                "status": status,
            })

        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        module_ranges: list[dict[str, Any]] = []
        module_index_rows: list[dict[str, Any]] = []
        transformed_socket_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        coverage_summaries: list[dict[str, Any]] = []
        total_volume = 0.0
        for module_id in chain:
            data = module_data[module_id]
            module = data["module"]
            transform = transforms[module_id]
            mesh = data["mesh"]
            vertex_offset = len(combined_vertices)
            triangle_offset = len(combined_triangles)
            transformed_vertices = [
                _rv(transform_point(vertex, transform["heading_deg"], transform["translation_m"]), int(spec["tolerances"]["mesh_round_digits"]))
                for vertex in mesh["vertices_m"]
            ]
            combined_vertices.extend(transformed_vertices)
            combined_triangles.extend([[int(a) + vertex_offset, int(b) + vertex_offset, int(c) + vertex_offset] for a, b, c in mesh["triangles"]])
            source_triangle_count = len(mesh["triangles"])
            module_range = {
                "module_id": module_id,
                "module_kind": module["module_kind"],
                "family": module["family"],
                "vertex_offset": vertex_offset,
                "vertex_count": len(transformed_vertices),
                "triangle_offset": triangle_offset,
                "triangle_count": source_triangle_count,
            }
            module_ranges.append(module_range)
            transformed_sockets = []
            for source_socket_id in sorted(data["sockets"]):
                transformed_socket = _transform_socket(data["sockets"][source_socket_id], transform["heading_deg"], transform["translation_m"])
                transformed_socket["socket_id"] = f"{module_id}::{source_socket_id}"
                transformed_socket["module_id"] = module_id
                transformed_sockets.append(transformed_socket)
                transformed_socket_rows.append(transformed_socket)
            source_refs = data["source_refs"]
            rows, coverage_summary = build_module_coverage(
                module_id=module_id,
                module_kind=module["module_kind"],
                family=module["family"],
                source_mesh=mesh,
                source_ranges=data["ranges"],
                frames=data["frames"],
                global_triangle_offset=triangle_offset,
                source_refs=source_refs,
            )
            coverage_rows.extend(rows)
            coverage_summaries.append(coverage_summary)
            module_index_rows.append({
                "module_id": module_id,
                "module_kind": module["module_kind"],
                "family": module["family"],
                "station": module["station"],
                "input_socket": module["input_socket"],
                "output_socket": module["output_socket"],
                "transform": transform,
                "source_refs": source_refs,
                "source_bounds_m": mesh["bounds_m"],
                "world_bounds_m": _bounds(transformed_vertices),
                "source_vertex_count": len(mesh["vertices_m"]),
                "source_triangle_count": source_triangle_count,
                "source_solid_count": int(data["result"].get("solid_count", 0)),
                "source_volume_m3": float(data["result"].get("volume_m3", 0.0)),
                "module_range": module_range,
                "transformed_socket_count": len(transformed_sockets),
                "coverage_surface_count": coverage_summary["surface_count"],
            })
            total_volume += float(data["result"].get("volume_m3", 0.0))

        if len(combined_vertices) > int(spec["budget"]["max_vertices"]) or len(combined_triangles) > int(spec["budget"]["max_triangles"]):
            raise MixedSpanContractError("combined mesh exceeds geometry budget")
        coverage = build_assembly_coverage(spec["assembly_id"], coverage_summaries, coverage_rows, len(combined_triangles))
        if len(coverage_rows) > int(spec["budget"]["max_coverage_rows"]):
            raise MixedSpanContractError("source coverage row budget exceeded")
        assembly_mesh = {
            "schema": MESH_SCHEMA,
            "assembly_id": spec["assembly_id"],
            "units": "METER",
            "frame": PROJECT_FRAME,
            "module_order": chain,
            "vertices_m": combined_vertices,
            "triangles": combined_triangles,
            "bounds_m": _bounds(combined_vertices),
            "module_ranges": module_ranges,
        }
        transform_document = {
            "schema": TRANSFORM_SCHEMA,
            "assembly_id": spec["assembly_id"],
            "frame": PROJECT_FRAME,
            "transforms": [transforms[module_id] for module_id in chain],
        }
        alignment_document = {
            "schema": SOCKET_ALIGNMENT_SCHEMA,
            "assembly_id": spec["assembly_id"],
            "status": "PASS",
            "linear_tolerance_m": linear_tolerance,
            "angular_tolerance_rad": angular_tolerance,
            "alignment_count": len(alignment_rows),
            "alignments": alignment_rows,
        }
        module_index = {
            "schema": MODULE_INDEX_SCHEMA,
            "assembly_id": spec["assembly_id"],
            "module_count": len(module_index_rows),
            "module_order": chain,
            "modules": module_index_rows,
        }
        socket_document = {
            "schema": "royal-capital.fortification.mixed-span-sockets/1",
            "assembly_id": spec["assembly_id"],
            "socket_count": len(transformed_socket_rows),
            "sockets": transformed_socket_rows,
        }
        plan = {
            "schema": ASSEMBLY_PLAN_SCHEMA,
            "assembly_id": spec["assembly_id"],
            "frame": PROJECT_FRAME,
            "segmentation_digest": segmentation["segmentation_digest"],
            "module_order": chain,
            "connection_count": len(spec["connections"]),
            "module_count": len(chain),
            "span_segment_count": segmentation["span_segment_count"],
            "join_count": segmentation["join_count"],
            "assembly_mode": "EXACT_MODULE_REFERENCES_WITH_RIGID_TRANSFORMS_NO_GLOBAL_BOOLEAN",
            "identity_policy": segmentation["identity_policy"],
            "source_coverage_policy": coverage["coverage_policy"],
            "terrain_mutation": False,
            "product_cook": "DEFERRED_TO_R0E",
        }
        documents = {
            "segmentation-plan.json": segmentation,
            "mixed-span-plan.json": plan,
            "instance-transforms.json": transform_document,
            "socket-alignment.json": alignment_document,
            "module-index.json": module_index,
            "module-sockets.json": socket_document,
            "neutral-mesh.json": assembly_mesh,
            "source-coverage-map.json": coverage,
        }
        references: dict[str, dict[str, Any]] = {}
        for name, document in documents.items():
            path = output / name
            path.write_bytes(pretty_json_bytes(document))
            references[name] = _ref(path, output)
        source_dir = Path(__file__).resolve().parent
        source_names = ("__init__.py", "model.py", "coverage.py", "producer.py")
        source_digest_input = b"".join(name.encode("utf-8") + b"\0" + (source_dir / name).read_bytes() for name in source_names)
        fixture_digest = sha256_ref(canonical_json_bytes(spec))
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "producer_id": "royal-capital/fortification/mixed-span-assembly",
            "producer_revision": "r0b-cp4.1",
            "producer_source_digest": sha256_ref(source_digest_input),
            "fixture_digest": fixture_digest,
            "segmentation_digest": segmentation["segmentation_digest"],
            "runtime": actual_runtime,
            "module_count": len(chain),
            "connection_count": len(alignment_rows),
            "module_source_digests": {
                row["module_id"]: {
                    "result": row["source_refs"]["result"]["sha256"],
                    "neutral_mesh": row["source_refs"]["neutral_mesh"]["sha256"],
                    "provider_receipt": row["source_refs"]["provider_receipt"]["sha256"],
                }
                for row in module_index_rows
            },
            "source_coverage_digest": references["source-coverage-map.json"]["sha256"],
            "socket_alignment_digest": references["socket-alignment.json"]["sha256"],
            "output_digests": {name: reference["sha256"] for name, reference in references.items()},
            "capabilities": [
                "fortification.mixed_span_assembly@1",
                "fortification.segmentation_determinism@1",
                "fortification.source_coverage@1",
                "fortification.exact_module_transform@1",
            ],
            "global_boolean_used": False,
            "provider_face_identity_used": False,
        }
        receipt_path = output / "cad-provider-receipt.json"
        receipt_path.write_bytes(pretty_json_bytes(receipt))
        receipt_ref = _ref(receipt_path, output)
        result = {
            "schema": RESULT_SCHEMA,
            "assembly_id": spec["assembly_id"],
            "status": "SUCCEEDED",
            "failure": None,
            "module_count": len(chain),
            "span_segment_count": segmentation["span_segment_count"],
            "join_count": segmentation["join_count"],
            "connection_count": len(alignment_rows),
            "solid_count": sum(int(row["source_solid_count"]) for row in module_index_rows),
            "volume_m3": _round(total_volume),
            "bounds_m": assembly_mesh["bounds_m"],
            "vertex_count": len(combined_vertices),
            "triangle_count": len(combined_triangles),
            "source_surface_count": coverage["summary"]["surface_count"],
            "covered_triangle_count": coverage["summary"]["covered_triangle_count"],
            "coverage_gap_count": coverage["summary"]["gap_count"],
            "coverage_overlap_count": coverage["summary"]["overlap_count"],
            "socket_alignment_status": alignment_document["status"],
            "segmentation_digest": segmentation["segmentation_digest"],
            "artifacts": {**references, "cad-provider-receipt.json": receipt_ref},
        }
        result_path = output / "result.json"
        result_path.write_bytes(pretty_json_bytes(result))
        total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if total_bytes > int(spec["budget"]["max_artifact_bytes"]):
            raise MixedSpanContractError(f"assembly artifact byte budget exceeded: {total_bytes}")
        return result
