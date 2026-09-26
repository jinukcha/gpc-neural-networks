from __future__ import annotations

from io import BytesIO
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from rcf_fortification_cad.canonical import (
    canonicalize_mesh,
    normalized_step_sha256,
    pretty_json_bytes,
    project_to_provider,
    sha256_ref,
    unit_frame_contract,
)

from .model import (
    EXPECTED_RUNTIME,
    JOIN_MESH_SCHEMA,
    JOIN_PARTS_SCHEMA,
    JOIN_PLAN_SCHEMA,
    JOIN_RECEIPT_SCHEMA,
    JOIN_RESULT_SCHEMA,
    PART_ORDER,
    STORED_SCHEMA,
    TESSELLATION_SCHEMA,
    bounded_overlap,
    canonical_json_bytes,
    interface_sockets,
    profile_part,
    section_plan,
    sha256_ref as model_sha256_ref,
    socket_alignment,
    validate_fixture,
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
    return {
        "min": [min(float(vertex[index]) for vertex in vertices) for index in range(3)],
        "max": [max(float(vertex[index]) for vertex in vertices) for index in range(3)],
    }


def _point(section: Mapping[str, Any], y: float, z: float) -> tuple[float, float, float]:
    origin = section["origin_m"]
    up = section["up"]
    inside = section["inside"]
    return tuple(float(origin[index]) + y * float(up[index]) + z * float(inside[index]) for index in range(3))


def _section_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return sum((float(a["origin_m"][index]) - float(b["origin_m"][index])) ** 2 for index in range(3)) ** 0.5


def _area(part: Mapping[str, Any]) -> float:
    return (float(part["y_max"]) - float(part["y_min"])) * (float(part["z_max"]) - float(part["z_min"]))


class WallJoinProducer:
    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.cp0_root = Path(cp0_root).resolve()

    def execute(self, fixture: Mapping[str, Any], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
        spec = validate_fixture(fixture)
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"output must be fresh: {output}")
        temporary = output.with_name(output.name + f".partial-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"temporary exists: {temporary}")
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

    def _produce(self, spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
        import build123d as b3d

        actual = {
            "build123d_version": importlib.metadata.version("build123d"),
            "ocp_version": importlib.metadata.version("cadquery-ocp-novtk"),
            "build123d_source_commit": EXPECTED_RUNTIME["build123d_source_commit"],
            "ocp_wheel_sha256": EXPECTED_RUNTIME["ocp_wheel_sha256"],
        }
        if actual != EXPECTED_RUNTIME:
            raise RuntimeError(f"runtime identity mismatch {actual}")

        sections = section_plan(spec)
        sockets_doc = interface_sockets(spec)
        alignment_doc = socket_alignment(spec, sockets_doc)
        overlap_doc = bounded_overlap(spec, sections)
        if alignment_doc["status"] != "PASS":
            raise RuntimeError("socket alignment failed")
        if overlap_doc["status"] != "PASS":
            raise RuntimeError("bounded overlap failed")

        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        part_ranges: list[dict[str, Any]] = []
        semantic_parts: list[dict[str, Any]] = []
        stored: list[dict[str, Any]] = []
        part_receipts: list[dict[str, Any]] = []
        total_volume = 0.0

        for part_id in PART_ORDER:
            part_dir = output / "parts" / part_id
            part_dir.mkdir(parents=True)
            wires = []
            section_rows = []
            area_integral = 0.0
            for section_index, section in enumerate(sections):
                part = profile_part(section["profile"], part_id)
                project_points = [
                    _point(section, part["y_min"], part["z_min"]),
                    _point(section, part["y_max"], part["z_min"]),
                    _point(section, part["y_max"], part["z_max"]),
                    _point(section, part["y_min"], part["z_max"]),
                ]
                provider_points = [project_to_provider(point) for point in project_points]
                wires.append(b3d.Wire.make_polygon(provider_points, close=True))
                section_rows.append({
                    "section_id": section["section_id"],
                    "origin_m": section["origin_m"],
                    "tangent": section["tangent"],
                    "inside": section["inside"],
                    "profile_id": section["profile"]["profile_id"],
                    "part_profile": part,
                    "area_m2": round(_area(part), 9),
                })
                if section_index:
                    previous = section_rows[section_index - 1]
                    distance = _section_distance(sections[section_index - 1], section)
                    area_integral += 0.5 * (float(previous["area_m2"]) + float(section_rows[section_index]["area_m2"])) * distance

            shape = b3d.Solid.make_loft(wires, ruled=True)
            if not shape.is_valid or shape.volume <= 0:
                raise RuntimeError(f"invalid join loft for {part_id}")
            provider_vertices, provider_triangles = shape.tessellate(
                float(spec["tolerances"]["tessellation_linear_m"]) * 1000.0,
                float(spec["tolerances"]["tessellation_angular_rad"]),
            )
            mesh = canonicalize_mesh(provider_vertices, provider_triangles)
            if len(mesh["vertices_m"]) > int(spec["budget"]["max_vertices"]) or len(mesh["triangles"]) > int(spec["budget"]["max_triangles"]):
                raise RuntimeError("geometry budget exceeded")
            (part_dir / "neutral-mesh.json").write_bytes(pretty_json_bytes(mesh))

            copies = []
            step_buffer = BytesIO()
            if not b3d.export_step(shape, step_buffer, unit=b3d.Unit.MM, timestamp="1970-01-01T00:00:00"):
                raise RuntimeError("STEP export failed")
            step = step_buffer.getvalue()
            (part_dir / "shape.step").write_bytes(step)
            reopened_step = b3d.import_step(part_dir / "shape.step")
            copies.append({
                "format": "STEP",
                "path": "shape.step",
                "bytes": len(step),
                "raw_sha256": "sha256:" + hashlib.sha256(step).hexdigest(),
                "canonical_sha256": "sha256:" + normalized_step_sha256(step),
                "reopen_volume_m3": round(float(reopened_step.volume) / 1_000_000_000.0, 9),
                "stored_numeric_unit": "MILLIMETER",
                "project_unit": "METER",
            })
            brep_buffer = BytesIO()
            if not b3d.export_brep(shape, brep_buffer):
                raise RuntimeError("BREP export failed")
            brep = brep_buffer.getvalue()
            (part_dir / "shape.brep").write_bytes(brep)
            reopened_brep = b3d.import_brep(part_dir / "shape.brep")
            brep_digest = "sha256:" + hashlib.sha256(brep).hexdigest()
            copies.append({
                "format": "BREP",
                "path": "shape.brep",
                "bytes": len(brep),
                "raw_sha256": brep_digest,
                "canonical_sha256": brep_digest,
                "reopen_volume_m3": round(float(reopened_brep.volume) / 1_000_000_000.0, 9),
                "stored_numeric_unit": "MILLIMETER",
                "project_unit": "METER",
            })
            volume = round(float(shape.volume) / 1_000_000_000.0, 9)
            approximation_error = abs(volume - area_integral) / max(volume, area_integral, 1e-9)
            if approximation_error > 0.35:
                raise RuntimeError(f"join volume integration error too large for {part_id}: {approximation_error}")
            part_result = {
                "schema": "royal-capital.fortification.wall-join-part-result/1",
                "join_id": spec["join_id"],
                "family": spec["family"],
                "part_id": part_id,
                "status": "SUCCEEDED",
                "geometry_route": {
                    "MITER": "BOUNDED_MITER_SECTION_LOFT",
                    "BEVEL": "BOUNDED_BEVEL_SECTION_LOFT",
                    "PROFILE_TRANSITION": "BOUNDED_PROFILE_TRANSITION_LOFT",
                }[spec["family"]],
                "section_count": len(sections),
                "sections": section_rows,
                "volume_m3": volume,
                "section_area_integral_m3": round(area_integral, 9),
                "relative_integration_error": round(approximation_error, 12),
                "bounds_m": mesh["bounds_m"],
                "vertex_count": len(mesh["vertices_m"]),
                "triangle_count": len(mesh["triangles"]),
                "stored_copies": copies,
            }
            (part_dir / "result.json").write_bytes(pretty_json_bytes(part_result))
            part_receipt = {
                "schema": "royal-capital.fortification.wall-join-part-receipt/1",
                "status": "PASS",
                "join_id": spec["join_id"],
                "family": spec["family"],
                "part_id": part_id,
                "fixture_digest": model_sha256_ref(canonical_json_bytes(spec)),
                "runtime": actual,
                "geometry_route": part_result["geometry_route"],
                "section_ids": [section["section_id"] for section in sections],
                "neutral_mesh_sha256": _digest(part_dir / "neutral-mesh.json"),
                "stored_copy_canonical_digests": {copy["format"]: copy["canonical_sha256"] for copy in copies},
            }
            (part_dir / "cad-provider-receipt.json").write_bytes(pretty_json_bytes(part_receipt))

            vertex_offset = len(combined_vertices)
            triangle_offset = len(combined_triangles)
            combined_vertices.extend(mesh["vertices_m"])
            combined_triangles.extend([[a + vertex_offset, b + vertex_offset, c + vertex_offset] for a, b, c in mesh["triangles"]])
            part_ranges.append({
                "part_id": part_id,
                "vertex_offset": vertex_offset,
                "vertex_count": len(mesh["vertices_m"]),
                "triangle_offset": triangle_offset,
                "triangle_count": len(mesh["triangles"]),
            })
            first_part = profile_part(spec["incoming_profile"], part_id)
            last_part = profile_part(spec["outgoing_profile"], part_id)
            semantic_parts.append({
                "part_id": part_id,
                "semantic_role": first_part["semantic_role"],
                "incoming_material_slot": first_part["material_slot"],
                "outgoing_material_slot": last_part["material_slot"],
                "bounds_m": mesh["bounds_m"],
                "volume_m3": volume,
                "provider_result_ref": _ref(part_dir / "result.json", output),
                "provider_receipt_ref": _ref(part_dir / "cad-provider-receipt.json", output),
                "surface_coverage": "DEFERRED_TO_R0B_CP4",
            })
            for copy in copies:
                item = dict(copy)
                item["part_id"] = part_id
                item["path"] = (Path("parts") / part_id / copy["path"]).as_posix()
                stored.append(item)
            part_receipts.append({"part_id": part_id, "receipt": _ref(part_dir / "cad-provider-receipt.json", output)})
            total_volume += volume

        mesh_doc = {
            "schema": JOIN_MESH_SCHEMA,
            "units": "METER",
            "frame": spec["frame"],
            "part_order": list(PART_ORDER),
            "vertices_m": combined_vertices,
            "triangles": combined_triangles,
            "bounds_m": _bounds(combined_vertices),
            "part_ranges": part_ranges,
        }
        parts_doc = {"schema": JOIN_PARTS_SCHEMA, "join_id": spec["join_id"], "parts": semantic_parts}
        sections_doc = {
            "schema": "royal-capital.fortification.wall-join-sections/1",
            "join_id": spec["join_id"],
            "family": spec["family"],
            "sections": [
                {
                    "section_id": section["section_id"],
                    "origin_m": section["origin_m"],
                    "tangent": section["tangent"],
                    "up": section["up"],
                    "inside": section["inside"],
                    "outside": section["outside"],
                    "orientation_determinant": section["orientation_determinant"],
                    "profile_id": section["profile"]["profile_id"],
                }
                for section in sections
            ],
        }
        tessellation_doc = {
            "schema": TESSELLATION_SCHEMA,
            "linear_deflection_m": float(spec["tolerances"]["tessellation_linear_m"]),
            "angular_deflection_rad": float(spec["tolerances"]["tessellation_angular_rad"]),
            "mesh_round_digits": int(spec["tolerances"]["mesh_round_digits"]),
            "part_order": list(PART_ORDER),
            "geometry_route": {
                "MITER": "BOUNDED_MITER_SECTION_LOFT",
                "BEVEL": "BOUNDED_BEVEL_SECTION_LOFT",
                "PROFILE_TRANSITION": "BOUNDED_PROFILE_TRANSITION_LOFT",
            }[spec["family"]],
        }
        join_plan = {
            "schema": JOIN_PLAN_SCHEMA,
            "join_id": spec["join_id"],
            "family": spec["family"],
            "anchor_m": spec["incoming"]["anchor_m"],
            "turn_angle_deg": spec["turn_angle_deg"],
            "incoming_span_id": spec["incoming"]["span_id"],
            "outgoing_span_id": spec["outgoing"]["span_id"],
            "incoming_profile_id": spec["incoming_profile"]["profile_id"],
            "outgoing_profile_id": spec["outgoing_profile"]["profile_id"],
            "section_count": len(sections),
            "section_ids": [section["section_id"] for section in sections],
            "part_order": list(PART_ORDER),
            "assembly_mode": "BOUNDED_NON_UNIONED_PARTS",
            "join_ownership": "JOIN_MODULE_ONLY_SPAN_GEOMETRY_UNCHANGED",
            "source_coverage_stage": "R0B_CP4",
        }
        stored_doc = {"schema": STORED_SCHEMA, "join_id": spec["join_id"], "copies": stored}
        documents = {
            "join-plan.json": join_plan,
            "section-plan.json": sections_doc,
            "semantic-parts.json": parts_doc,
            "join-sockets.json": sockets_doc,
            "socket-alignment.json": alignment_doc,
            "bounded-overlap.json": overlap_doc,
            "fixed-tessellation.json": tessellation_doc,
            "neutral-mesh.json": mesh_doc,
            "stored-copies.json": stored_doc,
        }
        references = {}
        for name, value in documents.items():
            path = output / name
            path.write_bytes(pretty_json_bytes(value))
            references[name] = _ref(path, output)

        source_dir = Path(__file__).resolve().parent
        source_names = ("__init__.py", "model.py", "producer.py")
        source_digest = b"".join(name.encode("utf-8") + b"\0" + (source_dir / name).read_bytes() for name in source_names)
        receipt = {
            "schema": JOIN_RECEIPT_SCHEMA,
            "status": "PASS",
            "producer_id": "royal-capital/fortification/wall-join",
            "producer_revision": "r0b-cp2.1",
            "producer_source_digest": sha256_ref(source_digest),
            "fixture_digest": model_sha256_ref(canonical_json_bytes(spec)),
            "runtime": actual,
            "unit_frame_contract": unit_frame_contract(),
            "family": spec["family"],
            "turn_angle_deg": spec["turn_angle_deg"],
            "geometry_route": tessellation_doc["geometry_route"],
            "part_receipts": part_receipts,
            "socket_alignment_digest": _digest(output / "socket-alignment.json"),
            "bounded_overlap_digest": _digest(output / "bounded-overlap.json"),
            "output_digests": {name: reference["sha256"] for name, reference in references.items()},
            "capabilities": [
                "fortification.miter_join@1",
                "fortification.bevel_join@1",
                "fortification.profile_transition_join@1",
                "fortification.socket_alignment@1",
                "fortification.bounded_overlap@1",
            ],
            "surface_source_coverage": "DEFERRED_R0B_CP4",
        }
        (output / "cad-provider-receipt.json").write_bytes(pretty_json_bytes(receipt))
        receipt_ref = _ref(output / "cad-provider-receipt.json", output)
        result = {
            "schema": JOIN_RESULT_SCHEMA,
            "join_id": spec["join_id"],
            "status": "SUCCEEDED",
            "failure": None,
            "family": spec["family"],
            "turn_angle_deg": spec["turn_angle_deg"],
            "section_count": len(sections),
            "solid_count": len(PART_ORDER),
            "volume_m3": round(total_volume, 9),
            "bounds_m": mesh_doc["bounds_m"],
            "vertex_count": len(combined_vertices),
            "triangle_count": len(combined_triangles),
            "part_count": len(semantic_parts),
            "socket_count": len(sockets_doc["sockets"]),
            "socket_alignment_count": alignment_doc["alignment_count"],
            "socket_alignment_status": alignment_doc["status"],
            "bounded_overlap_status": overlap_doc["status"],
            "artifacts": {**references, "cad-provider-receipt.json": receipt_ref},
        }
        (output / "result.json").write_bytes(pretty_json_bytes(result))
        total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if total_bytes > int(spec["budget"]["max_artifact_bytes"]):
            raise RuntimeError("artifact byte budget exceeded")
        return result
