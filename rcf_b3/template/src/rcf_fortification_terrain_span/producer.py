from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from rcf_fortification_cad import Build123dProviderAdapter, CadStatus
from rcf_fortification_cad.canonical import pretty_json_bytes, unit_frame_contract

from .model import (
    CONTACT_SCHEMA, EXPECTED_RUNTIME, FOUNDATION_SCHEMA, GAP_SCHEMA, GRADE_SCHEMA,
    MESH_SCHEMA, PART_ORDER, PARTS_SCHEMA, PLAN_SCHEMA, RECEIPT_SCHEMA,
    RESULT_SCHEMA, SOCKETS_SCHEMA, STORED_SCHEMA, TerrainFailure, TerrainSpanError,
    bands_for_contract, canonical_bytes, derive_contract, pretty_bytes,
    provider_request, sha256_ref, socket_plan, validate_fixture,
)


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path, base: Path) -> dict[str, Any]:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def bounds(vertices: list[list[float]]) -> dict[str, list[float]]:
    return {"min": [min(row[i] for row in vertices) for i in range(3)], "max": [max(row[i] for row in vertices) for i in range(3)]}


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            h.update(path.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + h.hexdigest()


class TerrainSpanProducer:
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
        actual = {
            "build123d_version": importlib.metadata.version("build123d"),
            "ocp_version": importlib.metadata.version("cadquery-ocp-novtk"),
            "build123d_source_commit": EXPECTED_RUNTIME["build123d_source_commit"],
            "ocp_wheel_sha256": EXPECTED_RUNTIME["ocp_wheel_sha256"],
        }
        if actual != EXPECTED_RUNTIME:
            raise TerrainSpanError(TerrainFailure.RUNTIME_MISMATCH, f"runtime identity mismatch {actual}")
        contract = derive_contract(spec)
        bands = bands_for_contract(contract)
        segments = contract["foundation_interface"]["segments"]
        frame_map = {row["segment_id"]: row for row in contract["segment_frames"]["frames"]}
        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        component_ranges: list[dict[str, Any]] = []
        semantic_aggregate = {band.part_id: {"part_id": band.part_id, "semantic_role": band.semantic_role, "material_slot": band.material_slot, "components": [], "volume_m3": 0.0} for band in bands}
        stored_copies: list[dict[str, Any]] = []
        provider_receipts: list[dict[str, Any]] = []
        total_volume = 0.0

        for segment in segments:
            frame = frame_map[segment["segment_id"]]
            for band in bands:
                component_id = f"{segment['segment_id']}--{band.part_id}"
                component_dir = output / "components" / segment["segment_id"] / band.part_id
                request = provider_request(contract, segment, frame, band)
                result = self.adapter.execute(request, component_dir)
                if result.status is not CadStatus.SUCCEEDED:
                    raise TerrainSpanError(TerrainFailure.PROVIDER_EXECUTION_FAILED, f"component {component_id} provider failure {result.to_dict()}")
                mesh = json.loads((component_dir / "neutral-mesh.json").read_text(encoding="utf-8"))
                provider_result = json.loads((component_dir / "result.json").read_text(encoding="utf-8"))
                receipt = json.loads((component_dir / "cad-provider-receipt.json").read_text(encoding="utf-8"))
                volume = float(provider_result["shape"]["volume_m3"])
                expected_volume = float(segment["length_m"]) * (band.y_max - band.y_min) * (band.z_max - band.z_min)
                if abs(volume - expected_volume) > max(1e-8, expected_volume * 1e-9):
                    raise TerrainSpanError(TerrainFailure.PROVIDER_EXECUTION_FAILED, f"component {component_id} volume mismatch {volume} != {expected_volume}")
                vertex_offset, triangle_offset = len(combined_vertices), len(combined_triangles)
                combined_vertices.extend(mesh["vertices_m"])
                combined_triangles.extend([[a + vertex_offset, b + vertex_offset, c + vertex_offset] for a, b, c in mesh["triangles"]])
                component_range = {
                    "component_id": component_id,
                    "segment_id": segment["segment_id"],
                    "part_id": band.part_id,
                    "vertex_offset": vertex_offset,
                    "vertex_count": len(mesh["vertices_m"]),
                    "triangle_offset": triangle_offset,
                    "triangle_count": len(mesh["triangles"]),
                    "bounds_m": mesh["bounds_m"],
                    "volume_m3": round(volume, 9),
                    "provider_result_ref": ref(component_dir / "result.json", output),
                    "provider_receipt_ref": ref(component_dir / "cad-provider-receipt.json", output),
                }
                component_ranges.append(component_range)
                semantic_aggregate[band.part_id]["components"].append(component_range)
                semantic_aggregate[band.part_id]["volume_m3"] += volume
                for item in provider_result["stored_copies"]:
                    copy = dict(item)
                    copy["component_id"] = component_id
                    copy["segment_id"] = segment["segment_id"]
                    copy["part_id"] = band.part_id
                    copy["path"] = (Path("components") / segment["segment_id"] / band.part_id / item["path"]).as_posix()
                    stored_copies.append(copy)
                provider_receipts.append({"component_id": component_id, "segment_id": segment["segment_id"], "part_id": band.part_id, "receipt": ref(component_dir / "cad-provider-receipt.json", output), "adapter_revision": receipt["adapter_revision"]})
                total_volume += volume

        if len(combined_vertices) > spec["budget"]["max_vertices"] or len(combined_triangles) > spec["budget"]["max_triangles"]:
            raise TerrainSpanError(TerrainFailure.GEOMETRY_BUDGET_EXCEEDED, "combined mesh exceeds requested budget")

        mesh_doc = {
            "schema": MESH_SCHEMA,
            "units": "METER",
            "frame": spec["frame"],
            "part_order": list(PART_ORDER),
            "segment_order": [segment["segment_id"] for segment in segments],
            "vertices_m": combined_vertices,
            "triangles": combined_triangles,
            "bounds_m": bounds(combined_vertices),
            "component_ranges": component_ranges,
        }
        semantic_parts = []
        for part_id in PART_ORDER:
            row = semantic_aggregate[part_id]
            row["volume_m3"] = round(row["volume_m3"], 9)
            row["component_count"] = len(row["components"])
            row["surface_coverage"] = "DEFERRED_TO_R0B_CP4"
            semantic_parts.append(row)
        parts_doc = {"schema": PARTS_SCHEMA, "span_id": spec["span_id"], "parts": semantic_parts}
        sockets_doc = {"schema": SOCKETS_SCHEMA, "span_id": spec["span_id"], "sockets": socket_plan(contract)}
        stored_doc = {"schema": STORED_SCHEMA, "span_id": spec["span_id"], "copies": stored_copies}
        plan_doc = {
            "schema": PLAN_SCHEMA,
            "span_id": spec["span_id"],
            "span_family": spec["span_family"],
            "canonical_centerline_digest": contract["canonical_centerline"]["canonical_digest"],
            "length_m": round(contract["total_length_m"], 9),
            "segment_count": len(segments),
            "component_count": len(component_ranges),
            "part_order": list(PART_ORDER),
            "assembly_mode": "BOUNDED_NON_UNIONED_SEGMENT_PARTS",
            "terrain_mutation_requested": False,
            "join_source_surface_coverage": "DEFERRED_TO_R0B_CP4",
        }
        docs = {
            "canonical-centerline.json": contract["canonical_centerline"],
            "segment-local-frames.json": contract["segment_frames"],
            "terrain-span-plan.json": plan_doc,
            "foundation-interface.json": contract["foundation_interface"],
            "contact-evidence.json": contract["contact_evidence"],
            "gap-evidence.json": contract["gap_evidence"],
            "grade-evidence.json": contract["grade_evidence"],
            "semantic-parts.json": parts_doc,
            "sockets.json": sockets_doc,
            "neutral-mesh.json": mesh_doc,
            "stored-copies.json": stored_doc,
        }
        refs: dict[str, Any] = {}
        for name, value in docs.items():
            path = output / name
            path.write_bytes(pretty_bytes(value))
            refs[name] = ref(path, output)
        source_dir = Path(__file__).resolve().parent
        source_names = ("__init__.py", "model.py", "producer.py")
        source_digest = b"".join(name.encode() + b"\0" + (source_dir / name).read_bytes() for name in source_names)
        receipt_doc = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "producer_id": "royal-capital/fortification/terrain-span",
            "producer_revision": "r0b-cp3.1",
            "producer_source_digest": sha256_ref(source_digest),
            "fixture_digest": sha256_ref(canonical_bytes(spec)),
            "canonical_centerline_digest": contract["canonical_centerline"]["canonical_digest"],
            "runtime": actual,
            "unit_frame_contract": unit_frame_contract(),
            "geometry_route": "SEGMENTED_PROFILE_EXTRUSION_WITH_EXPLICIT_FOUNDATION_INTERFACE",
            "provider_receipts": provider_receipts,
            "output_digests": {name: item["sha256"] for name, item in refs.items()},
            "capabilities": ["fortification.terrain_stepped_span@1", "fortification.retaining_span@1", "fortification.foundation_interface@1", "fortification.contact_gap_grade_evidence@1"],
            "terrain_mutated": False,
            "surface_source_coverage": "DEFERRED_R0B_CP4",
        }
        receipt_path = output / "cad-provider-receipt.json"
        receipt_path.write_bytes(pretty_bytes(receipt_doc))
        receipt_ref = ref(receipt_path, output)
        result_doc = {
            "schema": RESULT_SCHEMA,
            "span_id": spec["span_id"],
            "status": "SUCCEEDED",
            "failure": None,
            "span_family": spec["span_family"],
            "centerline_length_m": round(contract["total_length_m"], 9),
            "foundation_segment_count": len(segments),
            "component_count": len(component_ranges),
            "semantic_part_count": len(PART_ORDER),
            "solid_count": len(component_ranges),
            "volume_m3": round(total_volume, 9),
            "bounds_m": mesh_doc["bounds_m"],
            "vertex_count": len(combined_vertices),
            "triangle_count": len(combined_triangles),
            "socket_count": len(sockets_doc["sockets"]),
            "contact_sample_count": contract["contact_evidence"]["sample_count"],
            "maximum_grade": contract["grade_evidence"]["maximum_observed_grade"],
            "maximum_gap_m": contract["gap_evidence"]["maximum_observed_gap_m"],
            "maximum_penetration_m": contract["gap_evidence"]["maximum_observed_penetration_m"],
            "terrain_mutated": False,
            "artifacts": {**refs, "cad-provider-receipt.json": receipt_ref},
        }
        (output / "result.json").write_bytes(pretty_bytes(result_doc))
        total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if total_bytes > spec["budget"]["max_artifact_bytes"]:
            raise TerrainSpanError(TerrainFailure.GEOMETRY_BUDGET_EXCEEDED, f"artifact bytes {total_bytes} exceed budget")
        return result_doc


class TerrainSpanPublisher:
    """Preserves failed staging work and never replaces an accepted result."""

    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.producer = TerrainSpanProducer(cp0_root)

    def execute(self, fixture: Mapping[str, Any], accepted_dir: str | os.PathLike[str], failure_root: str | os.PathLike[str], failure_case: str) -> dict[str, Any]:
        accepted = Path(accepted_dir).resolve()
        failures = Path(failure_root).resolve()
        failures.mkdir(parents=True, exist_ok=True)
        before = tree_digest(accepted) if accepted.exists() else None
        try:
            result = self.producer.execute(fixture, accepted)
            return result
        except Exception as exc:
            code = exc.code if isinstance(exc, TerrainSpanError) else TerrainFailure.PROVIDER_EXECUTION_FAILED.value
            failed_target = failures / failure_case
            if failed_target.exists():
                raise FileExistsError(failed_target) from exc
            partials = sorted(accepted.parent.glob(accepted.name + ".failed-*"))
            if partials:
                if len(partials) != 1:
                    raise RuntimeError(f"ambiguous failed staging paths: {partials}") from exc
                partials[0].rename(failed_target)
            else:
                failed_target.mkdir(parents=True)
            after = tree_digest(accepted) if accepted.exists() else None
            failure = {
                "schema": "royal-capital.fortification.terrain-span-failure/1",
                "status": "REJECTED",
                "failure": {"code": str(code), "message": str(exc)},
                "accepted_target_existed_before": before is not None,
                "accepted_target_unchanged": before == after,
                "accepted_target_digest_before": before,
                "accepted_target_digest_after": after,
                "partial_output_published": False,
                "failed_staging_preserved": True,
            }
            (failed_target / "failure-result.json").write_bytes(pretty_json_bytes(failure))
            return failure
