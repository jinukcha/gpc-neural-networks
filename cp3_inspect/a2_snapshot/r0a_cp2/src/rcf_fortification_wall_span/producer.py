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
    COMBINED_MESH_SCHEMA, PART_ORDER, PART_SPECS, PARTS_SCHEMA, PLAN_SCHEMA,
    RECEIPT_SCHEMA, RESULT_SCHEMA, SOCKETS_SCHEMA, TESSELLATION_SCHEMA,
    expected_bounds, provider_request, socket_plan, validate_fixture,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": _digest(path)}


def _bounds(vertices: list[list[float]]) -> dict[str, list[float]]:
    return {"min": [min(v[i] for v in vertices) for i in range(3)], "max": [max(v[i] for v in vertices) for i in range(3)]}


class StraightWallSpanProducer:
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
            # Preserve failed unique work; never overwrite it or publish a partial success.
            failed = output.with_name(output.name + f".failed-{os.getpid()}")
            if temporary.exists():
                temporary.rename(failed)
            raise

    def _produce(self, spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
        length = float(spec["length_m"])
        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        part_ranges = []
        semantic_parts = []
        stored = []
        part_receipts = []
        total_volume = 0.0
        for part in PART_SPECS:
            part_dir = output / "parts" / part.part_id
            request = provider_request(spec, part)
            result = self.adapter.execute(request, part_dir)
            if result.status is not CadStatus.SUCCEEDED:
                raise RuntimeError(f"part {part.part_id} provider failure: {result.to_dict()}")
            mesh = json.loads((part_dir / "neutral-mesh.json").read_text(encoding="utf-8"))
            provider_result = json.loads((part_dir / "result.json").read_text(encoding="utf-8"))
            receipt = json.loads((part_dir / "cad-provider-receipt.json").read_text(encoding="utf-8"))
            expected_volume = round(length * part.volume_factor, 9)
            expected_box = expected_bounds(length, part)
            if provider_result["shape"]["volume_m3"] != expected_volume:
                raise RuntimeError(f"{part.part_id} volume mismatch")
            if provider_result["shape"]["bounds_m"] != expected_box:
                raise RuntimeError(f"{part.part_id} bounds mismatch {provider_result['shape']['bounds_m']} != {expected_box}")
            v0, t0 = len(combined_vertices), len(combined_triangles)
            combined_vertices.extend(mesh["vertices_m"])
            combined_triangles.extend([[a + v0, b + v0, c + v0] for a, b, c in mesh["triangles"]])
            part_ranges.append({
                "part_id": part.part_id,
                "vertex_offset": v0, "vertex_count": len(mesh["vertices_m"]),
                "triangle_offset": t0, "triangle_count": len(mesh["triangles"]),
            })
            semantic_parts.append({
                "part_id": part.part_id, "semantic_role": part.semantic_role,
                "material_slot": part.material_slot, "bounds_m": expected_box,
                "volume_m3": expected_volume,
                "provider_result_ref": _ref(part_dir / "result.json", output),
                "provider_receipt_ref": _ref(part_dir / "cad-provider-receipt.json", output),
                "surface_coverage": "DEFERRED_TO_R0A_CP3",
            })
            for item in provider_result["stored_copies"]:
                copy = dict(item)
                copy["part_id"] = part.part_id
                copy["path"] = (Path("parts") / part.part_id / item["path"]).as_posix()
                stored.append(copy)
            part_receipts.append({"part_id": part.part_id, "receipt": _ref(part_dir / "cad-provider-receipt.json", output), "adapter_revision": receipt["adapter_revision"]})
            total_volume += expected_volume

        mesh = {
            "schema": COMBINED_MESH_SCHEMA, "units": "METER",
            "frame": spec["frame"], "part_order": list(PART_ORDER),
            "vertices_m": combined_vertices, "triangles": combined_triangles,
            "bounds_m": _bounds(combined_vertices), "part_ranges": part_ranges,
        }
        plan = {
            "schema": PLAN_SCHEMA, "span_id": spec["span_id"], "span_family": "STRAIGHT_SPAN",
            "length_m": length, "axis": {"start_m": [0.0, 0.0, 0.0], "end_m": [length, 0.0, 0.0]},
            "inside_side": spec["inside_side"], "outside_side": spec["outside_side"],
            "part_order": list(PART_ORDER), "assembly_mode": "BOUNDED_NON_UNIONED_PARTS",
            "source_coverage_stage": "R0A_CP3",
        }
        parts_doc = {"schema": PARTS_SCHEMA, "span_id": spec["span_id"], "parts": semantic_parts}
        sockets_doc = {"schema": SOCKETS_SCHEMA, "span_id": spec["span_id"], "sockets": socket_plan(length)}
        tess_doc = {
            "schema": TESSELLATION_SCHEMA,
            "linear_deflection_m": float(spec["tolerances"]["tessellation_linear_m"]),
            "angular_deflection_rad": float(spec["tolerances"]["tessellation_angular_rad"]),
            "mesh_round_digits": int(spec["tolerances"]["mesh_round_digits"]),
            "part_order": list(PART_ORDER), "vertex_policy": "PER_PART_LEXICOGRAPHIC_UNIQUE",
            "triangle_policy": "CYCLIC_MIN_PRESERVE_WINDING_THEN_SORT",
        }
        stored_doc = {"schema": "royal-capital.fortification.wall-span-stored-copies/1", "span_id": spec["span_id"], "copies": stored}
        documents = {
            "wall-span-plan.json": plan, "semantic-parts.json": parts_doc,
            "sockets.json": sockets_doc, "fixed-tessellation.json": tess_doc,
            "neutral-mesh.json": mesh, "stored-copies.json": stored_doc,
        }
        refs = {}
        for name, value in documents.items():
            path = output / name
            path.write_bytes(pretty_json_bytes(value))
            refs[name] = _ref(path, output)

        source_dir = Path(__file__).resolve().parent
        source_names = ("__init__.py", "model.py", "producer.py")
        source_digest_input = b"".join(name.encode() + b"\0" + (source_dir / name).read_bytes() for name in source_names)
        receipt = {
            "schema": RECEIPT_SCHEMA, "status": "PASS",
            "producer_id": "royal-capital/fortification/straight-wall-span",
            "producer_revision": "r0a-cp2.1",
            "producer_source_digest": sha256_ref(source_digest_input),
            "fixture_digest": sha256_ref(canonical_json_bytes(spec)),
            "provider_adapter": {"id": "royal-capital/fortification/build123d-adapter", "revision": "r0a-cp1.1"},
            "runtime": dict(spec["runtime"]), "unit_frame_contract": unit_frame_contract(),
            "fixed_tessellation": tess_doc,
            "part_receipts": part_receipts,
            "output_digests": {name: ref["sha256"] for name, ref in refs.items()},
            "capabilities": ["fortification.straight_span@1", "fortification.semantic_parts@1", "fortification.sockets@1", "cad.fixed_tessellation@1"],
            "surface_source_coverage": "NOT_AUTHORED_CP3",
        }
        receipt_path = output / "cad-provider-receipt.json"
        receipt_path.write_bytes(pretty_json_bytes(receipt))
        receipt_ref = _ref(receipt_path, output)
        result = {
            "schema": RESULT_SCHEMA, "span_id": spec["span_id"], "status": "SUCCEEDED",
            "failure": None, "solid_count": len(PART_ORDER),
            "volume_m3": round(total_volume, 9), "bounds_m": mesh["bounds_m"],
            "vertex_count": len(combined_vertices), "triangle_count": len(combined_triangles),
            "part_count": len(semantic_parts), "socket_count": len(sockets_doc["sockets"]),
            "artifacts": {**refs, "cad-provider-receipt.json": receipt_ref},
        }
        (output / "result.json").write_bytes(pretty_json_bytes(result))
        return result
