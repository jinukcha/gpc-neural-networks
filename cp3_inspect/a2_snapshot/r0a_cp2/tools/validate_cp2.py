#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

PARTS = ["foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet"]
SOCKETS = ["span_start", "span_end", "wall_walk_start", "wall_walk_end", "foundation_start", "foundation_end", "tower_start", "tower_end", "utility_inside_01", "utility_inside_02"]
EXPECTED = {
 "foundation": ({"min":[0.0,-2.0,-4.0],"max":[24.0,0.0,4.0]},384.0),
 "wall_body": ({"min":[0.0,0.0,-3.0],"max":[24.0,12.0,3.0]},1728.0),
 "wall_walk": ({"min":[0.0,12.0,-3.2],"max":[24.0,12.6,3.2]},92.16),
 "inner_parapet": ({"min":[0.0,12.6,2.4],"max":[24.0,14.4,3.2]},34.56),
 "outer_parapet": ({"min":[0.0,12.6,-3.2],"max":[24.0,14.4,-2.4]},34.56),
}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def files(root): return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}

p=argparse.ArgumentParser(); p.add_argument('--a',required=True); p.add_argument('--b',required=True); p.add_argument('--report',required=True); a=p.parse_args()
ra,rb=Path(a.a),Path(a.b); checks=[]
def ck(name, ok, detail=None): checks.append({"name":name,"pass":bool(ok),"detail":detail});
required={"wall-span-plan.json","semantic-parts.json","sockets.json","fixed-tessellation.json","neutral-mesh.json","stored-copies.json","cad-provider-receipt.json","result.json"}
ck('root_required_A', required <= {p.name for p in ra.iterdir()}, sorted(required-{p.name for p in ra.iterdir()}))
ck('root_required_B', required <= {p.name for p in rb.iterdir()}, sorted(required-{p.name for p in rb.iterdir()}))
result=json.loads((ra/'result.json').read_text()); mesh=json.loads((ra/'neutral-mesh.json').read_text()); parts=json.loads((ra/'semantic-parts.json').read_text()); sockets=json.loads((ra/'sockets.json').read_text()); tess=json.loads((ra/'fixed-tessellation.json').read_text()); receipt=json.loads((ra/'cad-provider-receipt.json').read_text()); stored=json.loads((ra/'stored-copies.json').read_text())
ck('status',result['status']=='SUCCEEDED',result['status'])
ck('solid_count',result['solid_count']==5,result['solid_count'])
ck('volume',result['volume_m3']==2273.28,result['volume_m3'])
ck('bounds',result['bounds_m']=={'min':[0.0,-2.0,-4.0],'max':[24.0,14.4,4.0]},result['bounds_m'])
ck('mesh_counts',result['vertex_count']==40 and result['triangle_count']==60,[result['vertex_count'],result['triangle_count']])
ck('part_order',mesh['part_order']==PARTS,mesh['part_order'])
ck('part_ranges',len(mesh['part_ranges'])==5 and sum(x['vertex_count'] for x in mesh['part_ranges'])==40 and sum(x['triangle_count'] for x in mesh['part_ranges'])==60,mesh['part_ranges'])
ck('indices_valid',all(0<=i<len(mesh['vertices_m']) for tri in mesh['triangles'] for i in tri))
ck('finite_vertices',all(isinstance(x,(int,float)) and abs(x)<1e9 for v in mesh['vertices_m'] for x in v))
by_part={x['part_id']:x for x in parts['parts']}
ck('semantic_parts',list(by_part)==PARTS,list(by_part))
for part,(bounds,volume) in EXPECTED.items():
    ck(f'{part}_bounds',by_part[part]['bounds_m']==bounds,by_part[part]['bounds_m']); ck(f'{part}_volume',by_part[part]['volume_m3']==volume,by_part[part]['volume_m3']); ck(f'{part}_coverage_deferred',by_part[part]['surface_coverage']=='DEFERRED_TO_R0A_CP3')
ids=[x['socket_id'] for x in sockets['sockets']]; ck('socket_order',ids==SOCKETS,ids); ck('socket_unique',len(ids)==len(set(ids))==10)
ck('tess_linear',tess['linear_deflection_m']==0.05,tess); ck('tess_angular',tess['angular_deflection_rad']==0.1,tess); ck('mesh_round',tess['mesh_round_digits']==9,tess)
ck('stored_copy_count',len(stored['copies'])==10,len(stored['copies']))
ck('part_receipts',len(receipt['part_receipts'])==5,len(receipt['part_receipts']))
ck('cp3_not_claimed',receipt['surface_source_coverage']=='NOT_AUTHORED_CP3')
fa,fb=files(ra),files(rb); ck('file_set_equal',set(fa)==set(fb),{'missing':sorted(set(fa)-set(fb)),'extra':sorted(set(fb)-set(fa))}); ck('clean_replay_byte_identical',fa==fb,[k for k in fa if fb.get(k)!=fa[k]])
report={"schema":"royal-capital.fortification.cp2-validation/1","status":"PASS" if all(x['pass'] for x in checks) else "FAIL","checks":checks,"summary":{"passed":sum(x['pass'] for x in checks),"failed":sum(not x['pass'] for x in checks),"files_compared":len(fa)},"canonical_output_sha256":hashlib.sha256((ra/'result.json').read_bytes()).hexdigest()}
Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True)); raise SystemExit(0 if report['status']=='PASS' else 1)
