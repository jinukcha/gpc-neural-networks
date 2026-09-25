from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from rcf_fortification_cad import Build123dProviderAdapter, CadStatus
from rcf_fortification_cad.canonical import canonical_json_bytes, pretty_json_bytes, sha256_ref, unit_frame_contract

from .model import (
    BOUNDS_SCHEMA,
    FAILURE_SCHEMA,
    FOUNDATION_SCHEMA,
    MESH_SCHEMA,
    PART_ORDER,
    PARTS_SCHEMA,
    PLAN_SCHEMA,
    RECEIPT_SCHEMA,
    RESULT_SCHEMA,
    SOCKETS_SCHEMA,
    TESSELLATION_SCHEMA,
    TowerError,
    TowerFailureCode,
    TowerSpec,
    part_specs,
    provider_request,
    socket_plan,
    validate_fixture,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": _digest(path)}


def _bounds(vertices: list[list[float]]) -> dict[str, list[float]]:
    return {
        "min": [round(min(v[i] for v in vertices), 9) for i in range(3)],
        "max": [round(max(v[i] for v in vertices), 9) for i in range(3)],
    }


def _almost_bounds(a: Mapping[str, list[float]], b: Mapping[str, list[float]], tolerance: float) -> bool:
    return all(abs(float(a[key][i]) - float(b[key][i])) <= tolerance for key in ("min", "max") for i in range(3))


def _tree_digest(root: Path) -> str | None:
    if not root.exists():
        return None
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + h.hexdigest()


class TowerFamilyProducer:
    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.cp0_root = Path(cp0_root).resolve()
        self.adapter = Build123dProviderAdapter(self.cp0_root)

    def execute(
        self,
        fixture: Mapping[str, Any],
        output_dir: str | os.PathLike[str],
        failure_root: str | os.PathLike[str] | None = None,
        *,
        failure_case: str = "failure",
    ) -> dict[str, Any]:
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"output must be fresh: {output}")
        request_digest = sha256_ref(canonical_json_bytes(fixture))
        temporary = output.with_name(output.name + f".partial-{failure_case}-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"temporary path already exists: {temporary}")
        temporary.mkdir(parents=True)
        accepted_before = _tree_digest(output)
        try:
            spec = validate_fixture(fixture)
            result = self._produce(spec, temporary)
            temporary.rename(output)
            return result
        except Exception as exc:
            if isinstance(exc, TowerError):
                code, message = exc.code, exc.message
            else:
                code, message = str(TowerFailureCode.PUBLISH_ABORTED), f"{type(exc).__name__}: {exc}"
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
                "accepted_target_digest_after": _tree_digest(output),
                "accepted_target_unchanged": accepted_before == _tree_digest(output),
                "preserved_failed_work": True,
            }
            (failed / "failure-result.json").write_bytes(pretty_json_bytes(failure))
            return failure

    def _produce(self, spec: TowerSpec, output: Path) -> dict[str, Any]:
        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        part_ranges: list[dict[str, Any]] = []
        semantic_parts: list[dict[str, Any]] = []
        stored: list[dict[str, Any]] = []
        part_receipts: list[dict[str, Any]] = []
        total_volume = 0.0
        tolerance = max(1e-7, float(spec.raw["tolerances"]["linear_m"]) * 10.0)

        for part in part_specs(spec):
            part_dir = output / "parts" / part.part_id
            provider = self.adapter.execute(provider_request(spec, part), part_dir)
            if provider.status is CadStatus.REJECTED:
                failure = provider.failure or {}
                code = TowerFailureCode.GEOMETRY_BUDGET_EXCEEDED if "BUDGET" in str(failure.get("code", "")) else TowerFailureCode.PROVIDER_REJECTED
                raise TowerError(code, f"{part.part_id}: {failure}")
            if provider.status is not CadStatus.SUCCEEDED:
                raise TowerError(TowerFailureCode.PROVIDER_FAILED, f"{part.part_id}: {provider.to_dict()}")

            mesh = json.loads((part_dir / "neutral-mesh.json").read_text(encoding="utf-8"))
            provider_result = json.loads((part_dir / "result.json").read_text(encoding="utf-8"))
            provider_receipt = json.loads((part_dir / "cad-provider-receipt.json").read_text(encoding="utf-8"))
            expected_volume = round(spec.areas[part.part_id] * (part.y_max - part.y_min), 9)
            expected_bounds = spec.bounds[part.part_id]
            observed_volume = float(provider_result["shape"]["volume_m3"])
            observed_bounds = provider_result["shape"]["bounds_m"]
            if abs(observed_volume - expected_volume) > max(1e-6, tolerance ** 3):
                raise TowerError(TowerFailureCode.GEOMETRY_EVIDENCE_MISMATCH, f"{part.part_id} volume {observed_volume} != {expected_volume}")
            if not _almost_bounds(observed_bounds, expected_bounds, tolerance):
                raise TowerError(TowerFailureCode.GEOMETRY_EVIDENCE_MISMATCH, f"{part.part_id} bounds {observed_bounds} != {expected_bounds}")

            vertex_offset = len(combined_vertices)
            triangle_offset = len(combined_triangles)
            combined_vertices.extend(mesh["vertices_m"])
            combined_triangles.extend([[a + vertex_offset, b + vertex_offset, c + vertex_offset] for a, b, c in mesh["triangles"]])
            part_ranges.append({
                "part_id": part.part_id,
                "vertex_offset": vertex_offset,
                "vertex_count": len(mesh["vertices_m"]),
                "triangle_offset": triangle_offset,
                "triangle_count": len(mesh["triangles"]),
            })
            semantic_parts.append({
                "part_id": part.part_id,
                "semantic_role": part.semantic_role,
                "material_slot": part.material_slot,
                "bounds_m": expected_bounds,
                "volume_m3": expected_volume,
                "profile_point_count": len(spec.loops[part.part_id]),
                "provider_result_ref": _ref(part_dir / "result.json", output),
                "provider_receipt_ref": _ref(part_dir / "cad-provider-receipt.json", output),
                "surface_coverage": "DEFERRED_TO_R0C_CP4",
            })
            for item in provider_result["stored_copies"]:
                row = dict(item)
                row["part_id"] = part.part_id
                row["path"] = (Path("parts") / part.part_id / item["path"]).as_posix()
                stored.append(row)
            part_receipts.append({
                "part_id": part.part_id,
                "adapter_revision": provider_receipt["adapter_revision"],
                "receipt": _ref(part_dir / "cad-provider-receipt.json", output),
            })
            total_volume += expected_volume

        if len(combined_vertices) > spec.raw["budget"]["max_vertices"] or len(combined_triangles) > spec.raw["budget"]["max_triangles"]:
            raise TowerError(TowerFailureCode.GEOMETRY_BUDGET_EXCEEDED, "combined tower mesh exceeds fixture budget")

        sockets = socket_plan(spec)
        mesh_doc = {
            "schema": MESH_SCHEMA,
            "tower_id": spec.tower_id,
            "family": spec.family,
            "units": "METER",
            "frame": spec.raw.get("frame", "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"),
            "part_order": list(PART_ORDER),
            "vertices_m": combined_vertices,
            "triangles": combined_triangles,
            "bounds_m": _bounds(combined_vertices),
            "part_ranges": part_ranges,
        }
        plan = {
            "schema": PLAN_SCHEMA,
            "tower_id": spec.tower_id,
            "family": spec.family,
            "construction_route": spec.route,
            "center_m": [spec.center_x, 0.0, spec.center_z],
            "outer_width_m": spec.outer_width_m,
            "height_m": spec.height_m,
            "foundation_depth_m": spec.foundation_depth_m,
            "side_count": spec.side_count,
            "part_order": list(PART_ORDER),
            "assembly_mode": "BOUNDED_NON_UNIONED_PARTS",
            "roof_realization": "DEFERRED_TO_ARCHITECTURE_PRODUCER",
            "battlement_realization": "DEFERRED_TO_R0C_CP3",
            "tower_join_realization": "DEFERRED_TO_R0C_CP2",
            "surface_source_coverage": "DEFERRED_TO_R0C_CP4",
        }
        parts_doc = {"schema": PARTS_SCHEMA, "tower_id": spec.tower_id, "family": spec.family, "parts": semantic_parts}
        sockets_doc = {"schema": SOCKETS_SCHEMA, "tower_id": spec.tower_id, "family": spec.family, "socket_order": list(SOCKET_ORDER), "sockets": sockets}
        bounds_doc = {
            "schema": BOUNDS_SCHEMA,
            "tower_id": spec.tower_id,
            "family": spec.family,
            "body_bounds_m": spec.bounds["tower_body"],
            "foundation_bounds_m": spec.bounds["foundation"],
            "crown_bounds_m": spec.bounds["tower_crown"],
            "overall_bounds_m": mesh_doc["bounds_m"],
            "wall_centerline_z_m": spec.wall_centerline_z_m,
            "wall_outside_face_z_m": spec.wall_outside_face_z_m,
            "attachment_x_m": list(spec.attachment_x),
            "attachment_width_m": round(spec.attachment_x[1] - spec.attachment_x[0], 9),
            "body_outside_projection_m": spec.body_projection_m,
            "foundation_outside_projection_m": spec.foundation_projection_m,
            "maximum_body_projection_m": spec.max_body_projection_m,
            "attachment_intersection_count": 2,
            "bounds_status": "PASS",
        }
        foundation_loop = spec.loops["foundation"]
        foundation_doc = {
            "schema": FOUNDATION_SCHEMA,
            "tower_id": spec.tower_id,
            "family": spec.family,
            "support_class": "FLAT_REFERENCE_FOUNDATION",
            "bearing_elevation_m": -spec.foundation_depth_m,
            "bearing_polygon_xz_m": [list(point) for point in foundation_loop],
            "bearing_area_m2": round(spec.areas["foundation"], 9),
            "contact_area_m2": round(spec.areas["foundation"], 9),
            "contact_ratio": 1.0,
            "maximum_gap_m": 0.0,
            "embedment_depth_m": spec.foundation_depth_m,
            "terrain_mutation": False,
            "foundation_socket_ids": ["foundation_in", "foundation_out", "foundation_center", "foundation_outside", "foundation_inside"],
            "status": "PASS",
        }
        tess_doc = {
            "schema": TESSELLATION_SCHEMA,
            "linear_deflection_m": float(spec.raw["tolerances"]["tessellation_linear_m"]),
            "angular_deflection_rad": float(spec.raw["tolerances"]["tessellation_angular_rad"]),
            "mesh_round_digits": int(spec.raw["tolerances"]["mesh_round_digits"]),
            "vertex_policy": "PER_PART_LEXICOGRAPHIC_UNIQUE",
            "triangle_policy": "CYCLIC_MIN_PRESERVE_WINDING_THEN_SORT",
            "round_route": spec.route if spec.family == "ROUND" else None,
            "round_radial_chord_error_m": spec.approximation_error_m if spec.family == "ROUND" else 0.0,
            "part_order": list(PART_ORDER),
        }
        stored_doc = {
            "schema": "royal-capital.fortification.tower-stored-copies/1",
            "tower_id": spec.tower_id,
            "family": spec.family,
            "copies": stored,
        }
        documents = {
            "tower-plan.json": plan,
            "semantic-parts.json": parts_doc,
            "sockets.json": sockets_doc,
            "bounds-attachment.json": bounds_doc,
            "foundation-interface.json": foundation_doc,
            "fixed-tessellation.json": tess_doc,
            "neutral-mesh.json": mesh_doc,
            "stored-copies.json": stored_doc,
        }
        refs: dict[str, dict[str, Any]] = {}
        for name, document in documents.items():
            path = output / name
            path.write_bytes(pretty_json_bytes(document))
            refs[name] = _ref(path, output)

        source_dir = Path(__file__).resolve().parent
        source_files = ("__init__.py", "model.py", "producer.py")
        source_digest = sha256_ref(b"".join(name.encode("utf-8") + b"\0" + (source_dir / name).read_bytes() for name in source_files))
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "producer_id": "royal-capital/fortification/tower-family",
            "producer_revision": "r0c-cp1.1",
            "producer_source_digest": source_digest,
            "fixture_digest": sha256_ref(canonical_json_bytes(spec.raw)),
            "family": spec.family,
            "construction_route": spec.route,
            "runtime": dict(spec.raw["runtime"]),
            "unit_frame_contract": unit_frame_contract(),
            "fixed_tessellation": tess_doc,
            "part_receipts": part_receipts,
            "output_digests": {name: row["sha256"] for name, row in refs.items()},
            "surface_source_coverage": "NOT_AUTHORED_R0C_CP4",
            "capabilities": [
                "fortification.tower_family@1",
                "fortification.tower_bounds_attachment@1",
                "fortification.tower_foundation_sockets@1",
                "cad.fixed_tessellation@1",
            ],
        }
        receipt_path = output / "cad-provider-receipt.json"
        receipt_path.write_bytes(pretty_json_bytes(receipt))
        receipt_ref = _ref(receipt_path, output)
        result = {
            "schema": RESULT_SCHEMA,
            "tower_id": spec.tower_id,
            "family": spec.family,
            "status": "SUCCEEDED",
            "failure": None,
            "solid_count": len(PART_ORDER),
            "part_count": len(PART_ORDER),
            "socket_count": len(sockets),
            "stored_copy_count": len(stored),
            "volume_m3": round(total_volume, 9),
            "bounds_m": mesh_doc["bounds_m"],
            "vertex_count": len(combined_vertices),
            "triangle_count": len(combined_triangles),
            "body_outside_projection_m": spec.body_projection_m,
            "foundation_contact_ratio": 1.0,
            "foundation_maximum_gap_m": 0.0,
            "artifacts": {**refs, "cad-provider-receipt.json": receipt_ref},
        }
        result_path = output / "result.json"
        result_path.write_bytes(pretty_json_bytes(result))
        return result
