from __future__ import annotations

from io import BytesIO
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rcf_fortification_cad.canonical import canonicalize_mesh, normalized_step_sha256, project_to_provider

from .model import (
    JoinError, JoinFailure, JoinFamily, PART_ORDER, Band, WallProfile,
    add, band_map, cross, distance, dot, mul, normalize, profile, rounded, sub,
)


def p2(v: Sequence[float]) -> tuple[float, float]: return (float(v[0]), float(v[2]))
def v3(v: Sequence[float], y: float = 0.0) -> tuple[float, float, float]: return (float(v[0]), float(y), float(v[1]))
def add2(a,b): return (a[0]+b[0], a[1]+b[1])
def sub2(a,b): return (a[0]-b[0], a[1]-b[1])
def mul2(a,s): return (a[0]*s, a[1]*s)
def dot2(a,b): return a[0]*b[0]+a[1]*b[1]
def cross2(a,b): return a[0]*b[1]-a[1]*b[0]
def norm2(a): return math.sqrt(dot2(a,a))
def normalize2(a):
    n=norm2(a)
    if n <= 1e-12: raise JoinError(JoinFailure.INVALID_REQUEST,"zero plan vector")
    return (a[0]/n,a[1]/n)


def polygon_signed_area(poly: Sequence[tuple[float,float]]) -> float:
    return 0.5*sum(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))


def ensure_ccw(poly: Sequence[tuple[float,float]]) -> list[tuple[float,float]]:
    out=[(round(float(x),9),round(float(z),9)) for x,z in poly]
    if len(out)<3 or abs(polygon_signed_area(out))<1e-12: raise JoinError(JoinFailure.PROVIDER_EXECUTION_FAILED,"degenerate plan polygon")
    if polygon_signed_area(out)<0: out.reverse()
    return out


def line_intersection(p,d,q,e) -> tuple[float,float]:
    den=cross2(d,e)
    if abs(den)<1e-12: raise JoinError(JoinFailure.PROVIDER_EXECUTION_FAILED,"parallel join construction lines")
    t=cross2(sub2(q,p),e)/den
    return add2(p,mul2(d,t))


def section_points(position: Sequence[float], inside: Sequence[float], offset_min: float, offset_max: float) -> tuple[tuple[float,float],tuple[float,float]]:
    c=p2(position); n=p2(inside)
    return add2(c,mul2(n,offset_min)), add2(c,mul2(n,offset_max))


def _inside_point(p,a,b,eps=1e-10): return cross2(sub2(b,a),sub2(p,a)) >= -eps

def _segment_line_intersection(p1,p2,a,b):
    d=sub2(p2,p1); e=sub2(b,a)
    return line_intersection(p1,d,a,e)


def convex_intersection(subject: Sequence[tuple[float,float]], clip: Sequence[tuple[float,float]]) -> list[tuple[float,float]]:
    output=list(ensure_ccw(subject)); clip_poly=ensure_ccw(clip)
    for i in range(len(clip_poly)):
        a,b=clip_poly[i],clip_poly[(i+1)%len(clip_poly)]
        inp=output; output=[]
        if not inp: break
        s=inp[-1]
        for e in inp:
            ein=_inside_point(e,a,b); sin=_inside_point(s,a,b)
            if ein:
                if not sin: output.append(_segment_line_intersection(s,e,a,b))
                output.append(e)
            elif sin: output.append(_segment_line_intersection(s,e,a,b))
            s=e
    return output


def intersection_area(a,b) -> float:
    poly=convex_intersection(a,b)
    return abs(polygon_signed_area(poly)) if len(poly)>=3 else 0.0


def miter_components(spec: Mapping[str,Any], band: Band) -> list[dict[str,Any]]:
    anchor=p2(spec["anchor_m"]); p0=p2(spec["incoming_socket"]["position_m"]); p1=p2(spec["outgoing_socket"]["position_m"])
    t0=normalize2(p2(spec["incoming_socket"]["tangent"])); t1=normalize2(p2(spec["outgoing_socket"]["tangent"]))
    n0=normalize2(p2(spec["incoming_socket"]["inside"])); n1=normalize2(p2(spec["outgoing_socket"]["inside"]))
    seam_dir=normalize2(add2(n0,n1))
    seam_min_a=line_intersection(add2(p0,mul2(n0,band.offset_min)),t0,anchor,seam_dir)
    seam_min_b=line_intersection(add2(p1,mul2(n1,band.offset_min)),t1,anchor,seam_dir)
    seam_max_a=line_intersection(add2(p0,mul2(n0,band.offset_max)),t0,anchor,seam_dir)
    seam_max_b=line_intersection(add2(p1,mul2(n1,band.offset_max)),t1,anchor,seam_dir)
    if max(norm2(sub2(seam_min_a,seam_min_b)),norm2(sub2(seam_max_a,seam_max_b))) > 1e-7:
        raise JoinError(JoinFailure.SOCKET_GEOMETRY_MISMATCH,f"miter seam mismatch for {band.part_id}")
    seam_min=mul2(add2(seam_min_a,seam_min_b),0.5); seam_max=mul2(add2(seam_max_a,seam_max_b),0.5)
    start_min,start_max=section_points(spec["incoming_socket"]["position_m"],spec["incoming_socket"]["inside"],band.offset_min,band.offset_max)
    end_min,end_max=section_points(spec["outgoing_socket"]["position_m"],spec["outgoing_socket"]["inside"],band.offset_min,band.offset_max)
    return [
        {"component_id":f"{band.part_id}/incoming_miter","part_id":band.part_id,"kind":"PLAN_EXTRUSION","polygon_xz":ensure_ccw([start_min,start_max,seam_max,seam_min]),"y_min":band.y_min,"y_max":band.y_max},
        {"component_id":f"{band.part_id}/outgoing_miter","part_id":band.part_id,"kind":"PLAN_EXTRUSION","polygon_xz":ensure_ccw([seam_min,seam_max,end_max,end_min]),"y_min":band.y_min,"y_max":band.y_max},
    ]


def bevel_components(spec: Mapping[str,Any], band: Band) -> list[dict[str,Any]]:
    anchor=p2(spec["anchor_m"]); p0=p2(spec["incoming_socket"]["position_m"]); p1=p2(spec["outgoing_socket"]["position_m"])
    t0=normalize2(p2(spec["incoming_socket"]["tangent"])); t1=normalize2(p2(spec["outgoing_socket"]["tangent"]))
    n0=normalize2(p2(spec["incoming_socket"]["inside"])); n1=normalize2(p2(spec["outgoing_socket"]["inside"]))
    setback=float(spec["bevel_setback_m"])
    a_center=add2(anchor,mul2(t0,-setback)); b_center=add2(anchor,mul2(t1,setback))
    start_min,start_max=section_points(spec["incoming_socket"]["position_m"],spec["incoming_socket"]["inside"],band.offset_min,band.offset_max)
    end_min,end_max=section_points(spec["outgoing_socket"]["position_m"],spec["outgoing_socket"]["inside"],band.offset_min,band.offset_max)
    a_min,a_max=add2(a_center,mul2(n0,band.offset_min)),add2(a_center,mul2(n0,band.offset_max))
    b_min,b_max=add2(b_center,mul2(n1,band.offset_min)),add2(b_center,mul2(n1,band.offset_max))
    return [
        {"component_id":f"{band.part_id}/incoming_bevel","part_id":band.part_id,"kind":"PLAN_EXTRUSION","polygon_xz":ensure_ccw([start_min,start_max,a_max,a_min]),"y_min":band.y_min,"y_max":band.y_max},
        {"component_id":f"{band.part_id}/bevel_bridge","part_id":band.part_id,"kind":"PLAN_EXTRUSION","polygon_xz":ensure_ccw([a_min,a_max,b_max,b_min]),"y_min":band.y_min,"y_max":band.y_max},
        {"component_id":f"{band.part_id}/outgoing_bevel","part_id":band.part_id,"kind":"PLAN_EXTRUSION","polygon_xz":ensure_ccw([b_min,b_max,end_max,end_min]),"y_min":band.y_min,"y_max":band.y_max},
    ]


def transition_components(spec: Mapping[str,Any], start_band: Band, end_band: Band) -> list[dict[str,Any]]:
    return [{
        "component_id":f"{start_band.part_id}/profile_transition","part_id":start_band.part_id,"kind":"SECTION_LOFT",
        "start_section":{"position_m":spec["incoming_socket"]["position_m"],"inside":spec["incoming_socket"]["inside"],"y_min":start_band.y_min,"y_max":start_band.y_max,"offset_min":start_band.offset_min,"offset_max":start_band.offset_max},
        "end_section":{"position_m":spec["outgoing_socket"]["position_m"],"inside":spec["outgoing_socket"]["inside"],"y_min":end_band.y_min,"y_max":end_band.y_max,"offset_min":end_band.offset_min,"offset_max":end_band.offset_max},
    }]


def component_specs(spec: Mapping[str,Any]) -> list[dict[str,Any]]:
    incoming=band_map(spec["incoming_profile"]); outgoing=band_map(spec["outgoing_profile"]); out=[]
    for part_id in PART_ORDER:
        if spec["join_family"]==JoinFamily.MITER.value: out.extend(miter_components(spec,incoming[part_id]))
        elif spec["join_family"]==JoinFamily.BEVEL.value: out.extend(bevel_components(spec,incoming[part_id]))
        else: out.extend(transition_components(spec,incoming[part_id],outgoing[part_id]))
    return out


def section_wire(b3d, section: Mapping[str,Any]):
    pos=tuple(section["position_m"]); inside=tuple(section["inside"]); points=[]
    for y,o in ((section["y_min"],section["offset_min"]),(section["y_max"],section["offset_min"]),(section["y_max"],section["offset_max"]),(section["y_min"],section["offset_max"])):
        project=add(pos,add(mul((0.0,1.0,0.0),float(y)),mul(inside,float(o))))
        points.append(project_to_provider(project))
    return b3d.Wire.make_polygon(points,close=True)


def make_shape(component: Mapping[str,Any]):
    import build123d as b3d
    if component["kind"]=="PLAN_EXTRUSION":
        points=[project_to_provider((x,float(component["y_min"]),z)) for x,z in component["polygon_xz"]]
        wire=b3d.Wire.make_polygon(points,close=True); face=b3d.Face(wire)
        return b3d.extrude(face,amount=(float(component["y_max"])-float(component["y_min"]))*1000.0,dir=(0.0,0.0,1.0))
    if component["kind"]=="SECTION_LOFT":
        return b3d.Solid.make_loft([section_wire(b3d,component["start_section"]),section_wire(b3d,component["end_section"])],ruled=True)
    raise JoinError(JoinFailure.INVALID_REQUEST,f"unknown component kind {component['kind']}")


def export_shape(shape, output: Path, tolerances: Mapping[str,Any], budget: Mapping[str,Any]) -> dict[str,Any]:
    import build123d as b3d
    provider_vertices,provider_triangles=shape.tessellate(float(tolerances["tessellation_linear_m"])*1000.0,float(tolerances["tessellation_angular_rad"]))
    mesh=canonicalize_mesh(provider_vertices,provider_triangles)
    if len(mesh["vertices_m"])>int(budget["max_vertices"]) or len(mesh["triangles"])>int(budget["max_triangles"]):
        raise JoinError(JoinFailure.GEOMETRY_BUDGET_EXCEEDED,"component tessellation exceeds budget")
    output.mkdir(parents=True,exist_ok=False)
    step_buffer=BytesIO(); brep_buffer=BytesIO()
    if not b3d.export_step(shape,step_buffer,unit=b3d.Unit.MM,timestamp="1970-01-01T00:00:00"): raise JoinError(JoinFailure.STORED_COPY_FAILED,"STEP export failed")
    if not b3d.export_brep(shape,brep_buffer): raise JoinError(JoinFailure.STORED_COPY_FAILED,"BREP export failed")
    step=step_buffer.getvalue(); brep=brep_buffer.getvalue(); (output/"shape.step").write_bytes(step); (output/"shape.brep").write_bytes(brep)
    reopened_step=b3d.import_step(output/"shape.step"); reopened_brep=b3d.import_brep(output/"shape.brep")
    volume=float(shape.volume)/1_000_000_000.0
    if abs(float(reopened_step.volume)/1_000_000_000.0-volume)>1e-8 or abs(float(reopened_brep.volume)/1_000_000_000.0-volume)>1e-8:
        raise JoinError(JoinFailure.STORED_COPY_FAILED,"stored-copy reopen volume mismatch")
    return {
        "mesh":mesh,"volume_m3":round(volume,9),"solid_count":1,
        "stored_copies":[
            {"format":"STEP","path":"shape.step","bytes":len(step),"raw_sha256":"sha256:"+hashlib.sha256(step).hexdigest(),"canonical_sha256":"sha256:"+normalized_step_sha256(step),"reopen_volume_m3":round(float(reopened_step.volume)/1_000_000_000.0,9)},
            {"format":"BREP","path":"shape.brep","bytes":len(brep),"raw_sha256":"sha256:"+hashlib.sha256(brep).hexdigest(),"canonical_sha256":"sha256:"+hashlib.sha256(brep).hexdigest(),"reopen_volume_m3":round(float(reopened_brep.volume)/1_000_000_000.0,9)},
        ],
    }


def overlap_report(spec: Mapping[str,Any], components: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    rows=[]; max_area=0.0
    by_part={part:[] for part in PART_ORDER}
    for c in components:
        if c["kind"]=="PLAN_EXTRUSION": by_part[c["part_id"]].append(c)
    for part_id,items in by_part.items():
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                a,b=items[i],items[j]
                y_overlap=max(0.0,min(float(a["y_max"]),float(b["y_max"]))-max(float(a["y_min"]),float(b["y_min"])))
                area=intersection_area(a["polygon_xz"],b["polygon_xz"])
                volume=area*y_overlap; max_area=max(max_area,area)
                rows.append({"part_id":part_id,"a":a["component_id"],"b":b["component_id"],"plan_overlap_m2":round(area,12),"volume_overlap_m3":round(volume,12)})
    observed=max((math.sqrt(r["plan_overlap_m2"]) for r in rows),default=0.0)
    allowed=float(spec["max_overlap_m"])
    status="PASS" if observed<=allowed+1e-9 else "FAIL"
    if status!="PASS": raise JoinError(JoinFailure.OVERLAP_BUDGET_EXCEEDED,f"observed overlap equivalent {observed} > {allowed}")
    return {"status":status,"policy":"PAIRWISE_CONVEX_PLAN_INTERSECTION","allowed_linear_overlap_m":allowed,"observed_equivalent_linear_overlap_m":round(observed,12),"pairs":rows}
