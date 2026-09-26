from __future__ import annotations

from io import BytesIO
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping

from rcf_fortification_cad.canonical import canonicalize_mesh, normalized_step_sha256, pretty_json_bytes, project_to_provider, sha256_ref, unit_frame_contract

from .model import (
    COMBINED_MESH_SCHEMA, EXPECTED_RUNTIME, PART_ORDER, PART_SPECS, PARTS_SCHEMA, PLAN_SCHEMA,
    RECEIPT_SCHEMA, RESULT_SCHEMA, SOCKETS_SCHEMA, STORED_SCHEMA, TESSELLATION_SCHEMA,
    canonical_json_bytes, canonicalize_centerline, evaluate_frame, sha256_ref as model_sha256_ref,
    socket_plan, theoretical_volume_m3, validate_fixture,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": _digest(path)}


def _bounds(vertices: list[list[float]]) -> dict[str, list[float]]:
    return {"min": [min(v[i] for v in vertices) for i in range(3)], "max": [max(v[i] for v in vertices) for i in range(3)]}


def _point_at(frame: Mapping[str, Any], y: float, z: float) -> tuple[float,float,float]:
    o,up,inside=frame["origin_m"],frame["up"],frame["inside"]
    return tuple(float(o[i])+y*float(up[i])+z*float(inside[i]) for i in range(3))


class PathWallSpanProducer:
    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.cp0_root = Path(cp0_root).resolve()

    def execute(self, fixture: Mapping[str, Any], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
        spec = validate_fixture(fixture)
        output = Path(output_dir).resolve()
        if output.exists(): raise FileExistsError(f"output must be fresh: {output}")
        temporary = output.with_name(output.name+f".partial-{os.getpid()}")
        if temporary.exists(): raise FileExistsError(f"temporary exists: {temporary}")
        temporary.mkdir(parents=True)
        try:
            result=self._produce(spec,temporary)
            temporary.rename(output)
            return result
        except Exception:
            failed=output.with_name(output.name+f".failed-{os.getpid()}")
            if temporary.exists(): temporary.rename(failed)
            raise

    def _produce(self, spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
        import build123d as b3d
        actual={"build123d_version":importlib.metadata.version("build123d"),"ocp_version":importlib.metadata.version("cadquery-ocp-novtk"),"build123d_source_commit":EXPECTED_RUNTIME["build123d_source_commit"],"ocp_wheel_sha256":EXPECTED_RUNTIME["ocp_wheel_sha256"]}
        if actual != EXPECTED_RUNTIME: raise RuntimeError(f"runtime identity mismatch {actual}")
        centerline,frames=canonicalize_centerline(spec)
        combined_vertices=[]; combined_triangles=[]; part_ranges=[]; semantic_parts=[]; stored=[]; part_receipts=[]; total_volume=0.0
        for part in PART_SPECS:
            part_dir=output/"parts"/part.part_id; part_dir.mkdir(parents=True)
            wires=[]
            for frame in frames["frames"]:
                project_points=[_point_at(frame,part.y_min,part.z_min),_point_at(frame,part.y_max,part.z_min),_point_at(frame,part.y_max,part.z_max),_point_at(frame,part.y_min,part.z_max)]
                provider_points=[project_to_provider(p) for p in project_points]
                wires.append(b3d.Wire.make_polygon(provider_points,close=True))
            shape=b3d.Solid.make_loft(wires,ruled=True)
            if not shape.is_valid or shape.volume <= 0: raise RuntimeError(f"invalid loft for {part.part_id}")
            pv,pt=shape.tessellate(float(spec["tolerances"]["tessellation_linear_m"])*1000.0,float(spec["tolerances"]["tessellation_angular_rad"]))
            mesh=canonicalize_mesh(pv,pt)
            if len(mesh["vertices_m"])>spec["budget"]["max_vertices"] or len(mesh["triangles"])>spec["budget"]["max_triangles"]: raise RuntimeError("geometry budget exceeded")
            (part_dir/"neutral-mesh.json").write_bytes(pretty_json_bytes(mesh))
            copies=[]
            buf=BytesIO();
            if not b3d.export_step(shape,buf,unit=b3d.Unit.MM,timestamp="1970-01-01T00:00:00"): raise RuntimeError("STEP export failed")
            step=buf.getvalue(); (part_dir/"shape.step").write_bytes(step); reopen_step=b3d.import_step(part_dir/"shape.step")
            copies.append({"format":"STEP","path":"shape.step","bytes":len(step),"raw_sha256":"sha256:"+hashlib.sha256(step).hexdigest(),"canonical_sha256":"sha256:"+normalized_step_sha256(step),"reopen_volume_m3":round(float(reopen_step.volume)/1_000_000_000.0,9),"stored_numeric_unit":"MILLIMETER","project_unit":"METER"})
            buf=BytesIO();
            if not b3d.export_brep(shape,buf): raise RuntimeError("BREP export failed")
            brep=buf.getvalue(); (part_dir/"shape.brep").write_bytes(brep); reopen_brep=b3d.import_brep(part_dir/"shape.brep"); bd="sha256:"+hashlib.sha256(brep).hexdigest()
            copies.append({"format":"BREP","path":"shape.brep","bytes":len(brep),"raw_sha256":bd,"canonical_sha256":bd,"reopen_volume_m3":round(float(reopen_brep.volume)/1_000_000_000.0,9),"stored_numeric_unit":"MILLIMETER","project_unit":"METER"})
            volume=round(float(shape.volume)/1_000_000_000.0,9); expected=theoretical_volume_m3(centerline,part); rel=abs(volume-expected)/expected
            if rel>0.01: raise RuntimeError(f"volume approximation error {part.part_id}: {volume} expected {expected}")
            part_result={"schema":"royal-capital.fortification.path-span-part-result/1","part_id":part.part_id,"status":"SUCCEEDED","volume_m3":volume,"theoretical_volume_m3":round(expected,9),"relative_volume_error":round(rel,12),"bounds_m":mesh["bounds_m"],"vertex_count":len(mesh["vertices_m"]),"triangle_count":len(mesh["triangles"]),"stored_copies":copies}
            (part_dir/"result.json").write_bytes(pretty_json_bytes(part_result))
            receipt={"schema":"royal-capital.fortification.path-span-part-receipt/1","status":"PASS","part_id":part.part_id,"centerline_digest":centerline["canonical_digest"],"geometry_route":"BOUNDED_SAMPLED_SECTION_LOFT_RULED","runtime":actual,"neutral_mesh_sha256":_digest(part_dir/"neutral-mesh.json"),"stored_copy_canonical_digests":{c["format"]:c["canonical_sha256"] for c in copies}}
            (part_dir/"cad-provider-receipt.json").write_bytes(pretty_json_bytes(receipt))
            v0,t0=len(combined_vertices),len(combined_triangles); combined_vertices.extend(mesh["vertices_m"]); combined_triangles.extend([[a+v0,b+v0,c+v0] for a,b,c in mesh["triangles"]])
            part_ranges.append({"part_id":part.part_id,"vertex_offset":v0,"vertex_count":len(mesh["vertices_m"]),"triangle_offset":t0,"triangle_count":len(mesh["triangles"])})
            semantic_parts.append({"part_id":part.part_id,"semantic_role":part.semantic_role,"material_slot":part.material_slot,"bounds_m":mesh["bounds_m"],"volume_m3":volume,"theoretical_volume_m3":round(expected,9),"provider_result_ref":_ref(part_dir/"result.json",output),"provider_receipt_ref":_ref(part_dir/"cad-provider-receipt.json",output),"surface_coverage":"DEFERRED_TO_R0B_CP4"})
            for c in copies:
                item=dict(c); item["part_id"]=part.part_id; item["path"]=(Path("parts")/part.part_id/c["path"]).as_posix(); stored.append(item)
            part_receipts.append({"part_id":part.part_id,"receipt":_ref(part_dir/"cad-provider-receipt.json",output)})
            total_volume+=volume
        mesh={"schema":COMBINED_MESH_SCHEMA,"units":"METER","frame":spec["frame"],"part_order":list(PART_ORDER),"vertices_m":combined_vertices,"triangles":combined_triangles,"bounds_m":_bounds(combined_vertices),"part_ranges":part_ranges}
        sockets={"schema":SOCKETS_SCHEMA,"span_id":spec["span_id"],"sockets":socket_plan(centerline)}
        parts={"schema":PARTS_SCHEMA,"span_id":spec["span_id"],"parts":semantic_parts}
        tess={"schema":TESSELLATION_SCHEMA,"linear_deflection_m":float(spec["tolerances"]["tessellation_linear_m"]),"angular_deflection_rad":float(spec["tolerances"]["tessellation_angular_rad"]),"mesh_round_digits":int(spec["tolerances"]["mesh_round_digits"]),"geometry_route":"BOUNDED_SAMPLED_SECTION_LOFT_RULED","part_order":list(PART_ORDER)}
        plan={"schema":PLAN_SCHEMA,"span_id":spec["span_id"],"span_family":"STRAIGHT_SPAN" if centerline["kind"]=="LINE" else "CURVED_SPAN","canonical_centerline_digest":centerline["canonical_digest"],"length_m":centerline["length_m"],"segment_count":centerline["segment_count"],"inside_side":spec["inside_side"],"outside_side":spec["outside_side"],"part_order":list(PART_ORDER),"assembly_mode":"BOUNDED_NON_UNIONED_PARTS","surface_coverage_stage":"R0B_CP4"}
        stored_doc={"schema":STORED_SCHEMA,"span_id":spec["span_id"],"copies":stored}
        docs={"canonical-centerline.json":centerline,"local-frames.json":frames,"path-span-plan.json":plan,"semantic-parts.json":parts,"sockets.json":sockets,"fixed-tessellation.json":tess,"neutral-mesh.json":mesh,"stored-copies.json":stored_doc}
        refs={}
        for name,value in docs.items():
            p=output/name; p.write_bytes(pretty_json_bytes(value)); refs[name]=_ref(p,output)
        source_dir=Path(__file__).resolve().parent; source_names=("__init__.py","model.py","producer.py"); source_digest=b"".join(n.encode()+b"\0"+(source_dir/n).read_bytes() for n in source_names)
        receipt={"schema":RECEIPT_SCHEMA,"status":"PASS","producer_id":"royal-capital/fortification/path-wall-span","producer_revision":"r0b-cp1.1","producer_source_digest":sha256_ref(source_digest),"fixture_digest":model_sha256_ref(canonical_json_bytes(spec)),"canonical_centerline_digest":centerline["canonical_digest"],"runtime":actual,"unit_frame_contract":unit_frame_contract(),"geometry_route":"BOUNDED_SAMPLED_SECTION_LOFT_RULED","part_receipts":part_receipts,"output_digests":{n:r["sha256"] for n,r in refs.items()},"capabilities":["fortification.canonical_centerline@1","fortification.local_frame@1","fortification.straight_span@2","fortification.curved_span@1"],"surface_source_coverage":"DEFERRED_R0B_CP4"}
        (output/"cad-provider-receipt.json").write_bytes(pretty_json_bytes(receipt)); receipt_ref=_ref(output/"cad-provider-receipt.json",output)
        result={"schema":RESULT_SCHEMA,"span_id":spec["span_id"],"status":"SUCCEEDED","failure":None,"span_family":plan["span_family"],"centerline_length_m":centerline["length_m"],"centerline_segment_count":centerline["segment_count"],"solid_count":len(PART_ORDER),"volume_m3":round(total_volume,9),"bounds_m":mesh["bounds_m"],"vertex_count":len(combined_vertices),"triangle_count":len(combined_triangles),"part_count":len(semantic_parts),"socket_count":len(sockets["sockets"]),"artifacts":{**refs,"cad-provider-receipt.json":receipt_ref}}
        (output/"result.json").write_bytes(pretty_json_bytes(result))
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file())>spec["budget"]["max_artifact_bytes"]: raise RuntimeError("artifact byte budget exceeded")
        return result
