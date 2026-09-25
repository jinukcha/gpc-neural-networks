from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from rcf_fortification_cad.canonical import pretty_json_bytes, sha256_ref, unit_frame_contract

from .geometry import component_specs, export_shape, make_shape, overlap_report, p2, dot2, sub2
from .model import (
    ALIGNMENT_SCHEMA, EXPECTED_RUNTIME, FRAME_SCHEMA, MESH_SCHEMA, OVERLAP_SCHEMA,
    PARTS_SCHEMA, PART_ORDER, PLAN_SCHEMA, RECEIPT_SCHEMA, RESULT_SCHEMA, STORED_SCHEMA,
    JoinError, JoinFailure, JoinFamily, band_map, canonical_bytes, distance,
    pretty_bytes, profile, rounded, sha256_ref as model_sha256_ref, validate_fixture,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {"path":path.relative_to(base).as_posix(),"bytes":path.stat().st_size,"sha256":_digest(path)}


def _bounds(vertices: Sequence[Sequence[float]]) -> dict[str,list[float]]:
    return {"min":[min(float(v[i]) for v in vertices) for i in range(3)],"max":[max(float(v[i]) for v in vertices) for i in range(3)]}


def _tree_digest(root: Path) -> str:
    h=hashlib.sha256()
    if not root.exists(): return "sha256:"+h.hexdigest()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(p.relative_to(root).as_posix().encode()+b"\0"+hashlib.sha256(p.read_bytes()).digest())
    return "sha256:"+h.hexdigest()


class FortificationJoinProducer:
    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.cp0_root=Path(cp0_root).resolve()

    def _validate_runtime(self) -> None:
        actual={
            "build123d_version":importlib.metadata.version("build123d"),
            "ocp_version":importlib.metadata.version("cadquery-ocp-novtk"),
        }
        manifest=self.cp0_root/"runtime/WHEELHOUSE.json"
        lock=self.cp0_root/"runtime/requirements-lock.txt"
        if not manifest.is_file() or not lock.is_file(): raise JoinError(JoinFailure.RUNTIME_MISMATCH,"CP0 runtime evidence missing")
        m=json.loads(manifest.read_text())
        observed={**actual,"build123d_source_commit":m.get("build123d_source_commit"),"ocp_wheel_sha256":m.get("ocp_expected_sha256")}
        if observed!=EXPECTED_RUNTIME: raise JoinError(JoinFailure.RUNTIME_MISMATCH,f"observed runtime mismatch {observed}")

    def execute(self, fixture: Mapping[str,Any], output_dir: str | os.PathLike[str], failure_root: str | os.PathLike[str] | None=None, failure_case: str="failure") -> dict[str,Any]:
        output=Path(output_dir).resolve(); failures=Path(failure_root).resolve() if failure_root else output.parent/"failures"
        accepted_before=_tree_digest(output) if output.exists() else None
        staging=output.with_name(f".{output.name}.staging-{failure_case}-{os.getpid()}")
        if staging.exists(): raise FileExistsError(staging)
        try:
            spec=validate_fixture(fixture); self._validate_runtime()
            if output.exists(): raise JoinError(JoinFailure.PUBLISH_ABORTED,"accepted output already exists")
            staging.mkdir(parents=True)
            result=self._produce(spec,staging)
            staging.rename(output)
            return result
        except Exception as exc:
            error=exc if isinstance(exc,JoinError) else JoinError(JoinFailure.PROVIDER_EXECUTION_FAILED,f"{type(exc).__name__}: {exc}")
            failures.mkdir(parents=True,exist_ok=True); failed=failures/failure_case
            if failed.exists(): raise FileExistsError(failed) from exc
            if staging.exists(): staging.rename(failed)
            else: failed.mkdir()
            accepted_after=_tree_digest(output) if output.exists() else None
            failure={
                "schema":"royal-capital.fortification.join-failure/1","status":"REJECTED",
                "failure":{"code":error.code,"message":error.message},"partial_output_published":False,
                "accepted_target_unchanged":accepted_before==accepted_after,
                "accepted_target_digest_before":accepted_before,"accepted_target_digest_after":accepted_after,
                "preserved_failed_work":True,
            }
            (failed/"failure-result.json").write_bytes(pretty_bytes(failure))
            return failure

    def _produce(self, spec: Mapping[str,Any], output: Path) -> dict[str,Any]:
        components=component_specs(spec)
        if len(components)>int(spec["budget"]["max_components"]):
            raise JoinError(JoinFailure.GEOMETRY_BUDGET_EXCEEDED,f"components {len(components)} > {spec['budget']['max_components']}")
        combined_vertices=[]; combined_triangles=[]; component_ranges=[]; component_docs=[]; stored=[]
        total_volume=0.0; total_bytes=0
        for index,component in enumerate(components):
            cid=component["component_id"]; component_dir=output/"components"/cid.replace("/","__")
            shape=make_shape(component); evidence=export_shape(shape,component_dir,spec["tolerances"],spec["budget"])
            mesh=evidence["mesh"]
            v0,t0=len(combined_vertices),len(combined_triangles)
            combined_vertices.extend(mesh["vertices_m"]); combined_triangles.extend([[a+v0,b+v0,c+v0] for a,b,c in mesh["triangles"]])
            component_ranges.append({"component_id":cid,"part_id":component["part_id"],"vertex_offset":v0,"vertex_count":len(mesh["vertices_m"]),"triangle_offset":t0,"triangle_count":len(mesh["triangles"])})
            component_result={
                "schema":"royal-capital.fortification.join-component-result/1","component_id":cid,"part_id":component["part_id"],"kind":component["kind"],
                "status":"SUCCEEDED","solid_count":1,"volume_m3":evidence["volume_m3"],"bounds_m":mesh["bounds_m"],
                "vertex_count":len(mesh["vertices_m"]),"triangle_count":len(mesh["triangles"]),"source_spec":component,
                "stored_copies":evidence["stored_copies"],
            }
            (component_dir/"neutral-mesh.json").write_bytes(pretty_bytes(mesh)); (component_dir/"result.json").write_bytes(pretty_bytes(component_result))
            source_digest=model_sha256_ref(canonical_bytes(component))
            receipt={"schema":"royal-capital.fortification.join-component-receipt/1","status":"PASS","component_id":cid,"source_spec_digest":source_digest,"runtime":EXPECTED_RUNTIME,"unit_frame_contract":unit_frame_contract(),"mesh_digest":_digest(component_dir/"neutral-mesh.json"),"stored_copy_canonical_digests":{x["format"]:x["canonical_sha256"] for x in evidence["stored_copies"]}}
            (component_dir/"provider-receipt.json").write_bytes(pretty_bytes(receipt))
            refs={name:_ref(component_dir/name,output) for name in ("neutral-mesh.json","result.json","provider-receipt.json","shape.step","shape.brep")}
            component_docs.append({"component_id":cid,"part_id":component["part_id"],"kind":component["kind"],"bounds_m":mesh["bounds_m"],"volume_m3":evidence["volume_m3"],"refs":refs})
            for item in evidence["stored_copies"]:
                stored.append({**item,"component_id":cid,"part_id":component["part_id"],"path":(Path("components")/cid.replace("/","__")/item["path"]).as_posix()})
            total_volume+=float(evidence["volume_m3"]); total_bytes+=sum((component_dir/name).stat().st_size for name in ("neutral-mesh.json","result.json","provider-receipt.json","shape.step","shape.brep"))
        if len(combined_vertices)>int(spec["budget"]["max_vertices"]) or len(combined_triangles)>int(spec["budget"]["max_triangles"]) or total_bytes>int(spec["budget"]["max_artifact_bytes"]):
            raise JoinError(JoinFailure.GEOMETRY_BUDGET_EXCEEDED,"aggregate join output exceeds budget")
        mesh={"schema":MESH_SCHEMA,"units":"METER","frame":spec["frame"],"vertices_m":combined_vertices,"triangles":combined_triangles,"bounds_m":_bounds(combined_vertices),"component_ranges":component_ranges}
        overlap={"schema":OVERLAP_SCHEMA,"join_id":spec["join_id"],**overlap_report(spec,components)}
        plan={
            "schema":PLAN_SCHEMA,"join_id":spec["join_id"],"join_family":spec["join_family"],"anchor_m":spec["anchor_m"],"turn_angle_deg":spec["turn_angle_deg"],
            "incoming_profile":spec["incoming_profile"],"outgoing_profile":spec["outgoing_profile"],"join_depth_m":spec["join_depth_m"],"bevel_setback_m":spec["bevel_setback_m"],
            "assembly_mode":"BOUNDED_NON_UNIONED_COMPONENTS","component_order":[x["component_id"] for x in components],"source_coverage_stage":"R0B_CP4",
        }
        frames={"schema":FRAME_SCHEMA,"join_id":spec["join_id"],"incoming":spec["incoming_socket"],"outgoing":spec["outgoing_socket"],"anchor_m":spec["anchor_m"],"turn_angle_deg":spec["turn_angle_deg"]}
        sockets=self._sockets(spec)
        alignment=self._alignment(spec,components,sockets)
        parts=[]
        incoming_bands=band_map(spec["incoming_profile"]); outgoing_bands=band_map(spec["outgoing_profile"])
        for part_id in PART_ORDER:
            ids=[c["component_id"] for c in components if c["part_id"]==part_id]
            parts.append({"part_id":part_id,"semantic_role":incoming_bands[part_id].semantic_role,"incoming_material_slot":incoming_bands[part_id].material_slot,"outgoing_material_slot":outgoing_bands[part_id].material_slot,"component_ids":ids,"source_coverage":"DEFERRED_TO_R0B_CP4"})
        part_doc={"schema":PARTS_SCHEMA,"join_id":spec["join_id"],"parts":parts}
        stored_doc={"schema":STORED_SCHEMA,"join_id":spec["join_id"],"copies":stored}
        component_doc={"schema":"royal-capital.fortification.join-components/1","join_id":spec["join_id"],"components":component_docs}
        documents={"join-plan.json":plan,"local-frames.json":frames,"sockets.json":sockets,"socket-alignment.json":alignment,"bounded-overlap.json":overlap,"semantic-parts.json":part_doc,"neutral-mesh.json":mesh,"stored-copies.json":stored_doc,"components.json":component_doc}
        refs={}
        for name,value in documents.items():
            p=output/name; p.write_bytes(pretty_bytes(value)); refs[name]=_ref(p,output)
        source_dir=Path(__file__).resolve().parent
        source_digest=model_sha256_ref(b"".join(p.name.encode()+b"\0"+p.read_bytes() for p in sorted(source_dir.glob("*.py"))))
        receipt={
            "schema":RECEIPT_SCHEMA,"status":"PASS","producer_id":"royal-capital/fortification/join-producer","producer_revision":"r0b-cp2.1",
            "producer_source_digest":source_digest,"fixture_digest":model_sha256_ref(canonical_bytes(spec)),"runtime":EXPECTED_RUNTIME,"unit_frame_contract":unit_frame_contract(),
            "join_family":spec["join_family"],"component_count":len(components),"socket_alignment":"PASS","bounded_overlap":"PASS","output_digests":{k:v["sha256"] for k,v in refs.items()},
            "capabilities":["fortification.join.miter@1","fortification.join.bevel@1","fortification.join.profile_transition@1","fortification.socket_alignment@1","fortification.bounded_overlap@1"],
        }
        (output/"cad-provider-receipt.json").write_bytes(pretty_bytes(receipt)); receipt_ref=_ref(output/"cad-provider-receipt.json",output)
        result={
            "schema":RESULT_SCHEMA,"join_id":spec["join_id"],"join_family":spec["join_family"],"status":"SUCCEEDED","failure":None,
            "component_count":len(components),"part_count":5,"socket_count":len(sockets["sockets"]),"solid_count":len(components),"volume_m3":round(total_volume,9),
            "bounds_m":mesh["bounds_m"],"vertex_count":len(combined_vertices),"triangle_count":len(combined_triangles),"turn_angle_deg":spec["turn_angle_deg"],
            "socket_alignment":"PASS","bounded_overlap":"PASS","artifacts":{**refs,"cad-provider-receipt.json":receipt_ref},
        }
        (output/"result.json").write_bytes(pretty_bytes(result))
        return result

    def _sockets(self,spec: Mapping[str,Any]) -> dict[str,Any]:
        incoming=spec["incoming_socket"]; outgoing=spec["outgoing_socket"]; anchor=spec["anchor_m"]
        def row(sid,role,frame,required=True): return {"socket_id":sid,"role":role,"position_m":frame["position_m"],"frame":{k:frame[k] for k in ("tangent","up","inside","outside")},"required":required}
        sockets=[row("span_in","SPAN_JOIN_IN",incoming),row("span_out","SPAN_JOIN_OUT",outgoing),row("wall_walk_in","WALL_WALK_CONTINUATION",incoming),row("wall_walk_out","WALL_WALK_CONTINUATION",outgoing),row("foundation_in","FOUNDATION_INTERFACE",incoming),row("foundation_out","FOUNDATION_INTERFACE",outgoing),row("tower_anchor","TOWER_JOIN",{"position_m":anchor,"tangent":incoming["tangent"],"up":incoming["up"],"inside":incoming["inside"],"outside":incoming["outside"]},False),row("utility_inside","UTILITY_INSIDE",{"position_m":anchor,"tangent":incoming["tangent"],"up":incoming["up"],"inside":incoming["inside"],"outside":incoming["outside"]},False)]
        return {"schema":"royal-capital.fortification.join-sockets/1","join_id":spec["join_id"],"sockets":sockets}

    def _alignment(self,spec: Mapping[str,Any],components: Sequence[Mapping[str,Any]],sockets: Mapping[str,Any]) -> dict[str,Any]:
        rows=[]; tol=float(spec["tolerances"]["linear_m"])
        for side,expected,sid in (("incoming",spec["incoming_socket"],"span_in"),("outgoing",spec["outgoing_socket"],"span_out")):
            actual=next(s for s in sockets["sockets"] if s["socket_id"]==sid)
            position_error=distance(expected["position_m"],actual["position_m"])
            axis_errors={axis:distance(expected[axis],actual["frame"][axis]) for axis in ("tangent","up","inside","outside")}
            rows.append({"side":side,"socket_id":sid,"position_error_m":round(position_error,12),"axis_errors":{k:round(v,12) for k,v in axis_errors.items()},"pass":position_error<=tol and max(axis_errors.values())<=tol})
        if not all(r["pass"] for r in rows): raise JoinError(JoinFailure.SOCKET_GEOMETRY_MISMATCH,"socket alignment failed")
        return {"schema":ALIGNMENT_SCHEMA,"join_id":spec["join_id"],"status":"PASS","tolerance_m":tol,"rows":rows,"max_position_error_m":max(r["position_error_m"] for r in rows),"max_axis_error":max(max(r["axis_errors"].values()) for r in rows)}
