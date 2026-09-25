#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--tree',required=True); p.add_argument('--report',required=True); a=p.parse_args()
tree=Path(a.tree); fort=tree/'RC_K0/child_designs/fortification'; cp=fort/'r0b_cp1'; checks=[]
def ck(name,value,detail=None): checks.append({"name":name,"pass":bool(value),"detail":detail})
q=json.loads((cp/'reports/qualification.json').read_text())
ck('qualification',q['status']=='PASS',q)
ck('families',set(q['families'])=={'straight','curved'},sorted(q['families']))
ck('clean_replay',all(v['byte_identical'] for v in q['families'].values()))
ck('negative_gates',q['negative_count']==6,q['negative_count'])
for family in ('straight','curved'):
    root=cp/'outputs'/family/'reference'; result=json.loads((root/'result.json').read_text()); center=json.loads((root/'canonical-centerline.json').read_text()); frames=json.loads((root/'local-frames.json').read_text()); sockets=json.loads((root/'sockets.json').read_text()); parts=json.loads((root/'semantic-parts.json').read_text()); stored=json.loads((root/'stored-copies.json').read_text())
    ck(f'{family}_status',result['status']=='SUCCEEDED',result)
    ck(f'{family}_parts',result['part_count']==5 and len(parts['parts'])==5)
    ck(f'{family}_sockets',result['socket_count']==10 and len({x['socket_id'] for x in sockets['sockets']})==10)
    ck(f'{family}_stored',len(stored['copies'])==10)
    ck(f'{family}_digest',center['canonical_digest']==frames['canonical_centerline_digest'])
    ck(f'{family}_frames',all(abs(x['orientation_determinant']-1.0)<1e-12 for x in frames['frames']))
    ck(f'{family}_geometry',result['vertex_count']>0 and result['triangle_count']>0 and result['volume_m3']>0)
straight=json.loads((cp/'outputs/straight/reference/canonical-centerline.json').read_text()); curved=json.loads((cp/'outputs/curved/reference/canonical-centerline.json').read_text())
ck('straight_length',straight['length_m']==24.0,straight['length_m'])
ck('straight_segments',straight['segment_count']==1,straight['segment_count'])
ck('curved_length',abs(curved['length_m']-48.0*math.radians(30.0))<1e-8,curved['length_m'])
ck('curved_segments',curved['segment_count']==6,curved['segment_count'])
ck('curved_chord_error',curved['maximum_observed_chord_error_m']<=0.05,curved['maximum_observed_chord_error_m'])
ck('curved_bounds',curved['bounds_m']=={'min':[0.0,0.0,0.0],'max':[24.0,0.0,6.430780618]},curved['bounds_m'])
status_rows=list(csv.DictReader((fort/'data/CP_STATUS.csv').open(newline='')))
ck('parent_r0a_complete',any(r['stage']=='R0A' and r['checkpoint']=='CP4' and r['stage_completion']=='R0A_COMPLETE' for r in status_rows))
bad=[p.as_posix() for p in tree.rglob('*') if p.is_file() and (p.suffix in {'.pyc','.pyo'} or '__pycache__' in p.parts or '.godot' in p.parts)]
ck('no_cache',not bad,bad)
report={"schema":"royal-capital.fortification.r0b-cp1-validation/1","status":"PASS" if all(x['pass'] for x in checks) else "FAIL","checks":checks,"summary":{"passed":sum(x['pass'] for x in checks),"failed":sum(not x['pass'] for x in checks)}}
Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2,sort_keys=True)); raise SystemExit(0 if report['status']=='PASS' else 1)
