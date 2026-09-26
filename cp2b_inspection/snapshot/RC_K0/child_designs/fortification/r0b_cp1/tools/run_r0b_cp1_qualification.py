#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser(); p.add_argument('--tree',required=True); p.add_argument('--python',required=True); a=p.parse_args()
tree=Path(a.tree).resolve(); fort=tree/'RC_K0/child_designs/fortification'; cp0=fort/'r0a_cp0'; cp1=fort/'r0a_cp1'; r0b=fort/'r0b_cp1'
sys.path[:0]=[str(cp1/'src'),str(r0b/'src')]
from rcf_fortification_path_span import PathWallSpanProducer
from rcf_fortification_path_span.model import canonicalize_centerline, validate_fixture

outputs=r0b/'outputs'; reports=r0b/'reports'
if outputs.exists(): shutil.rmtree(outputs)
outputs.mkdir(parents=True); reports.mkdir(exist_ok=True)
producer=PathWallSpanProducer(cp0)

def digest_files(root): return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
summary={"schema":"royal-capital.fortification.r0b-cp1-qualification/1","families":{},"negative":[]}
for family,fixture_name in (("straight","straight_path_span.fixture.json"),("curved","curved_path_span.fixture.json")):
    fixture=json.loads((r0b/'fixtures'/fixture_name).read_text())
    family_root=outputs/family; A=family_root/'A'; B=family_root/'B'
    ra=producer.execute(fixture,A); rb=producer.execute(fixture,B)
    fa,fb=digest_files(A),digest_files(B)
    if fa!=fb: raise SystemExit(f"{family} clean replay mismatch: {[k for k in set(fa)|set(fb) if fa.get(k)!=fb.get(k)]}")
    reference=family_root/'reference'; shutil.copytree(A,reference)
    centerline=json.loads((reference/'canonical-centerline.json').read_text()); frames=json.loads((reference/'local-frames.json').read_text()); result=json.loads((reference/'result.json').read_text())
    summary['families'][family]={"status":"PASS","files_compared":len(fa),"byte_identical":True,"centerline_digest":centerline['canonical_digest'],"centerline_length_m":centerline['length_m'],"segment_count":centerline['segment_count'],"sample_count":len(centerline['samples']),"frame_count":len(frames['frames']),"bounds_m":result['bounds_m'],"vertex_count":result['vertex_count'],"triangle_count":result['triangle_count'],"volume_m3":result['volume_m3']}
    shutil.rmtree(A); shutil.rmtree(B)

# bounded contract negatives; no output should be created because validation occurs before execution.
base=json.loads((r0b/'fixtures/curved_path_span.fixture.json').read_text())
mutations=[]
for name,fn in [
    ('zero_length',lambda f:f.update(centerline={"kind":"LINE","start_m":[0,0,0],"end_m":[0,0,0]})),
    ('vertical_grade',lambda f:f.update(centerline={"kind":"LINE","start_m":[0,0,0],"end_m":[24,1,0]})),
    ('tight_radius',lambda f:f['centerline'].__setitem__('radius_m',3.0)),
    ('sweep_out_of_domain',lambda f:f['centerline'].__setitem__('sweep_angle_deg',40.0)),
    ('section_budget',lambda f:f['budget'].__setitem__('max_sections',2)),
    ('runtime_mismatch',lambda f:f['runtime'].__setitem__('ocp_version','0.0.0')),
]:
    value=copy.deepcopy(base); fn(value)
    try:
        # Run the complete project-owned centerline contract, not only static key/domain parsing.
        canonicalize_centerline(value)
    except Exception as exc:
        summary['negative'].append({"case":name,"status":"PASS_EXPECTED_REJECTION","error":type(exc).__name__,"message":str(exc)})
    else:
        raise SystemExit(f"negative fixture accepted: {name}")
summary['negative_count']=len(summary['negative'])
summary['status']='PASS'
(reports/'qualification.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')

# Run both pure contract and generated-output tests.
env=dict(__import__('os').environ); env['PYTHONPATH']=str(cp1/'src')+':'+str(r0b/'src'); env['RCF_R0B_CP1_OUTPUT_ROOT']=str(outputs); env['PYTHONDONTWRITEBYTECODE']='1'
with (reports/'unittest.log').open('wb') as log:
    proc=subprocess.run([a.python,'-m','unittest','discover','-s',str(r0b/'tests'),'-p','test_*.py','-v'],stdout=log,stderr=subprocess.STDOUT,env=env)
(reports/'unittest.json').write_text(json.dumps({"status":"PASS" if proc.returncode==0 else "FAIL","exit_code":proc.returncode},indent=2,sort_keys=True)+'\n')
if proc.returncode: raise SystemExit(proc.returncode)
print(json.dumps(summary,indent=2,sort_keys=True))
