from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from rcf_fortification_cad import Build123dProviderAdapter, CadStatus
from rcf_fortification_cad.canonical import pretty_json_bytes, sha256_ref, unit_frame_contract
from rcf_fortification_path_span.model import EXPECTED_RUNTIME, PART_ORDER

from .model import (
    CONTACT_EVIDENCE_SCHEMA,
    FOUNDATION_INTERFACE_SCHEMA,
    GRADE_EVIDENCE_SCHEMA,
    MESH_SCHEMA,
    PARTS_SCHEMA,
    PLAN_SCHEMA,
    RECEIPT_SCHEMA,
    RESULT_SCHEMA,
    SOCKETS_SCHEMA,
    STORED_SCHEMA,
    TESSELLATION_SCHEMA,
    axis_contract,
    canonical_json_bytes,
    canonical_terrain_profile,
    construction_units,
    contact_evidence,
    foundation_interface,
    grade_evidence,
    provider_request,
    sha256_ref as model_sha256_ref,
    socket_plan,
    terrain_profile_digest,
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


def _union_bounds(rows: Sequence[Mapping[str, Sequence[float]]]) -> dict[str, list[float]]:
    return {
        "min": [min(float(row["min"][index]) for row in rows) for index in range(3)],
        "max": [max(float(row["max"][index]) for row in rows) for index in range(3)],
    }


class TerrainWallSpanProducer:
    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.cp0_root = Path(cp0_root).resolve()
        self.adapter = Build123dProviderAdapter(self.cp0_root)

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
        actual_runtime = {
            "build123d_version": importlib.metadata.version("build123d"),
            "ocp_version": importlib.metadata.version("cadquery-ocp-novtk"),
            "build123d_source_commit": EXPECTED_RUNTIME["build123d_source_commit"],
            "ocp_wheel_sha256": EXPECTED_RUNTIME["ocp_wheel_sha256"],
        }
        if actual_runtime != EXPECTED_RUNTIME:
            raise RuntimeError(f"runtime identity mismatch: {actual_runtime}")

        axis = axis_contract(spec)
        profile_doc = canonical_terrain_profile(spec)
        interface_doc = foundation_interface(spec)
        contact_doc = contact_evidence(spec, interface_doc)
        grade_doc = grade_evidence(spec)
        sockets_doc = {"schema": SOCKETS_SCHEMA, "span_id": spec["span_id"], "sockets": socket_plan(spec)}
        if interface_doc["summary"]["status"] != "PASS":
            raise RuntimeError("foundation interface failed")
        if contact_doc["summary"]["status"] != "PASS":
            raise RuntimeError("contact/gap evidence failed")
        if grade_doc["status"] != "PASS":
            raise RuntimeError("grade evidence failed")

        units = construction_units(spec)
        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        unit_ranges: list[dict[str, Any]] = []
        unit_rows: list[dict[str, Any]] = []
        stored_rows: list[dict[str, Any]] = []
        part_aggregates: dict[str, dict[str, Any]] = {
            part_id: {"unit_refs": [], "bounds": [], "volume_m3": 0.0, "semantic_role": None, "material_slot": None}
            for part_id in PART_ORDER
        }
        total_volume = 0.0

        for unit in units:
            segment_directory = output / "units" / unit["segment_id"] / unit["part_id"]
            request = provider_request(spec, unit)
            provider_result = self.adapter.execute(request, segment_directory)
            if provider_result.status is not CadStatus.SUCCEEDED:
                raise RuntimeError(f"provider failure for {unit['unit_id']}: {provider_result.to_dict()}")
            mesh = json.loads((segment_directory / "neutral-mesh.json").read_text(encoding="utf-8"))
            result = json.loads((segment_directory / "result.json").read_text(encoding="utf-8"))
            receipt = json.loads((segment_directory / "cad-provider-receipt.json").read_text(encoding="utf-8"))
            expected_volume = float(unit["profile_area_m2"]) * float(unit["distance_m"])
            observed_volume = float(result["shape"]["volume_m3"])
            relative_error = abs(observed_volume - expected_volume) / max(abs(expected_volume), 1e-12)
            if relative_error > 1e-8:
                raise RuntimeError(
                    f"unit volume mismatch {unit['unit_id']}: observed={observed_volume} expected={expected_volume} relative={relative_error}"
                )

            vertex_offset = len(combined_vertices)
            triangle_offset = len(combined_triangles)
            combined_vertices.extend(mesh["vertices_m"])
            combined_triangles.extend([
                [int(a) + vertex_offset, int(b) + vertex_offset, int(c) + vertex_offset]
                for a, b, c in mesh["triangles"]
            ])
            unit_ranges.append({
                "unit_id": unit["unit_id"],
                "segment_id": unit["segment_id"],
                "part_id": unit["part_id"],
                "vertex_offset": vertex_offset,
                "vertex_count": len(mesh["vertices_m"]),
                "triangle_offset": triangle_offset,
                "triangle_count": len(mesh["triangles"]),
            })
            unit_reference = {
                "unit_id": unit["unit_id"],
                "segment_id": unit["segment_id"],
                "segment_index": unit["segment_index"],
                "part_id": unit["part_id"],
                "semantic_role": unit["semantic_role"],
                "material_slot": unit["material_slot"],
                "station_interval_m": [unit["start_station_m"], unit["end_station_m"]],
                "terrain_elevation_m": unit["terrain_elevation_m"],
                "profile_area_m2": unit["profile_area_m2"],
                "distance_m": unit["distance_m"],
                "volume_m3": round(observed_volume, 9),
                "expected_volume_m3": round(expected_volume, 9),
                "relative_volume_error": round(relative_error, 12),
                "bounds_m": result["shape"]["bounds_m"],
                "vertex_count": len(mesh["vertices_m"]),
                "triangle_count": len(mesh["triangles"]),
                "provider_result_ref": _ref(segment_directory / "result.json", output),
                "provider_receipt_ref": _ref(segment_directory / "cad-provider-receipt.json", output),
            }
            unit_rows.append(unit_reference)
            aggregate = part_aggregates[unit["part_id"]]
            aggregate["semantic_role"] = unit["semantic_role"]
            aggregate["material_slot"] = unit["material_slot"]
            aggregate["unit_refs"].append({
                "unit_id": unit["unit_id"],
                "provider_result_ref": unit_reference["provider_result_ref"],
                "provider_receipt_ref": unit_reference["provider_receipt_ref"],
            })
            aggregate["bounds"].append(result["shape"]["bounds_m"])
            aggregate["volume_m3"] += observed_volume
            total_volume += observed_volume

            for copy in result["stored_copies"]:
                item = dict(copy)
                item["unit_id"] = unit["unit_id"]
                item["segment_id"] = unit["segment_id"]
                item["part_id"] = unit["part_id"]
                item["path"] = (Path("units") / unit["segment_id"] / unit["part_id"] / copy["path"]).as_posix()
                stored_rows.append(item)
            if receipt["runtime"]["build123d_version"] != EXPECTED_RUNTIME["build123d_version"]:
                raise RuntimeError("provider receipt runtime mismatch")

        if len(combined_vertices) > int(spec["budget"]["max_vertices"]) or len(combined_triangles) > int(spec["budget"]["max_triangles"]):
            raise RuntimeError("combined geometry budget exceeded")

        geometry_route = {
            "TERRAIN_STEPPED": "BOUNDED_TERRACE_SEGMENT_PROFILE_EXTRUSIONS",
            "RETAINING": "ASYMMETRIC_RETAINING_PROFILE_EXTRUSION",
        }[spec["family"]]
        mesh_doc = {
            "schema": MESH_SCHEMA,
            "units": "METER",
            "frame": spec["frame"],
            "part_order": list(PART_ORDER),
            "unit_order": [unit["unit_id"] for unit in units],
            "vertices_m": combined_vertices,
            "triangles": combined_triangles,
            "bounds_m": _bounds(combined_vertices),
            "unit_ranges": unit_ranges,
        }
        semantic_parts = []
        for part_id in PART_ORDER:
            aggregate = part_aggregates[part_id]
            semantic_parts.append({
                "part_id": part_id,
                "semantic_role": aggregate["semantic_role"],
                "material_slot": aggregate["material_slot"],
                "unit_count": len(aggregate["unit_refs"]),
                "unit_refs": aggregate["unit_refs"],
                "bounds_m": _union_bounds(aggregate["bounds"]),
                "volume_m3": round(float(aggregate["volume_m3"]), 9),
                "surface_coverage": "DEFERRED_TO_R0B_CP4",
            })
        parts_doc = {"schema": PARTS_SCHEMA, "span_id": spec["span_id"], "parts": semantic_parts}
        unit_doc = {
            "schema": "royal-capital.fortification.terrain-span-construction-units/1",
            "span_id": spec["span_id"],
            "family": spec["family"],
            "unit_count": len(unit_rows),
            "units": unit_rows,
        }
        tessellation_doc = {
            "schema": TESSELLATION_SCHEMA,
            "linear_deflection_m": float(spec["tolerances"]["tessellation_linear_m"]),
            "angular_deflection_rad": float(spec["tolerances"]["tessellation_angular_rad"]),
            "mesh_round_digits": int(spec["tolerances"]["mesh_round_digits"]),
            "geometry_route": geometry_route,
            "part_order": list(PART_ORDER),
            "unit_order": [unit["unit_id"] for unit in units],
        }
        plan_doc = {
            "schema": PLAN_SCHEMA,
            "span_id": spec["span_id"],
            "family": spec["family"],
            "profile_id": spec["profile_id"],
            "axis_contract": axis,
            "terrain_profile_digest": terrain_profile_digest(spec),
            "terrain_source": dict(spec["terrain_source"]),
            "geometry_route": geometry_route,
            "unit_count": len(units),
            "segment_count": len(spec["terrain"].get("terraces", [])) if spec["family"] == "TERRAIN_STEPPED" else 1,
            "part_order": list(PART_ORDER),
            "assembly_mode": "BOUNDED_NON_UNIONED_TERRAIN_UNITS",
            "terrain_mutation": False,
            "surface_coverage_stage": "R0B_CP4",
        }
        stored_doc = {"schema": STORED_SCHEMA, "span_id": spec["span_id"], "copies": stored_rows}
        documents = {
            "canonical-terrain-profile.json": profile_doc,
            "terrain-span-plan.json": plan_doc,
            "construction-units.json": unit_doc,
            "foundation-interface.json": interface_doc,
            "contact-evidence.json": contact_doc,
            "grade-evidence.json": grade_doc,
            "interface-sockets.json": sockets_doc,
            "semantic-parts.json": parts_doc,
            "fixed-tessellation.json": tessellation_doc,
            "neutral-mesh.json": mesh_doc,
            "stored-copies.json": stored_doc,
        }
        references: dict[str, dict[str, Any]] = {}
        for name, document in documents.items():
            path = output / name
            path.write_bytes(pretty_json_bytes(document))
            references[name] = _ref(path, output)

        source_directory = Path(__file__).resolve().parent
        source_names = ("__init__.py", "model.py", "producer.py")
        source_digest_input = b"".join(
            name.encode("utf-8") + b"\0" + (source_directory / name).read_bytes()
            for name in source_names
        )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "producer_id": "royal-capital/fortification/terrain-wall-span",
            "producer_revision": "r0b-cp3.1",
            "producer_source_digest": sha256_ref(source_digest_input),
            "fixture_digest": model_sha256_ref(canonical_json_bytes(spec)),
            "terrain_profile_digest": terrain_profile_digest(spec),
            "runtime": actual_runtime,
            "unit_frame_contract": unit_frame_contract(),
            "geometry_route": geometry_route,
            "unit_count": len(units),
            "foundation_interface_digest": _digest(output / "foundation-interface.json"),
            "contact_evidence_digest": _digest(output / "contact-evidence.json"),
            "grade_evidence_digest": _digest(output / "grade-evidence.json"),
            "output_digests": {name: reference["sha256"] for name, reference in references.items()},
            "capabilities": [
                "fortification.terrain_stepped_span@1",
                "fortification.retaining_span@1",
                "fortification.foundation_interface@1",
                "fortification.contact_gap_evidence@1",
                "fortification.grade_evidence@1",
            ],
            "surface_source_coverage": "DEFERRED_R0B_CP4",
            "terrain_mutation": False,
        }
        (output / "cad-provider-receipt.json").write_bytes(pretty_json_bytes(receipt))
        receipt_ref = _ref(output / "cad-provider-receipt.json", output)
        result_doc = {
            "schema": RESULT_SCHEMA,
            "span_id": spec["span_id"],
            "status": "SUCCEEDED",
            "failure": None,
            "family": spec["family"],
            "geometry_route": geometry_route,
            "segment_count": plan_doc["segment_count"],
            "unit_count": len(units),
            "solid_count": len(units),
            "part_count": len(PART_ORDER),
            "socket_count": len(sockets_doc["sockets"]),
            "volume_m3": round(total_volume, 9),
            "bounds_m": mesh_doc["bounds_m"],
            "vertex_count": len(combined_vertices),
            "triangle_count": len(combined_triangles),
            "foundation_interface_status": interface_doc["summary"]["status"],
            "contact_evidence_status": contact_doc["summary"]["status"],
            "grade_evidence_status": grade_doc["status"],
            "maximum_gap_m": contact_doc["summary"]["maximum_gap_m"],
            "maximum_embedment_depth_m": interface_doc["summary"]["maximum_embedment_depth_m"],
            "overall_centerline_grade": grade_doc["overall_centerline_grade"],
            "maximum_step_height_m": grade_doc["maximum_step_height_m"],
            "retained_height_m": grade_doc["retained_height_m"],
            "terrain_mutation": False,
            "artifacts": {**references, "cad-provider-receipt.json": receipt_ref},
        }
        (output / "result.json").write_bytes(pretty_json_bytes(result_doc))
        total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if total_bytes > int(spec["budget"]["max_artifact_bytes"]):
            raise RuntimeError(f"artifact byte budget exceeded: {total_bytes}")
        return result_doc
