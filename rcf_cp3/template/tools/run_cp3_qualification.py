#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

p=argparse.ArgumentParser()
p.add_argument('--tree',required=True); p.add_argument('--runtime-python',required=True); p.add_argument('--work',required=True)
a=p.parse_args()
tree=Path(a.tree).resolve(); work=Path(a.work).resolve(); work.mkdir(parents=True,exist_ok=True)
fort=tree/'RC_K0/child_designs/fortification'; cp0=fort/'r0a_cp0'; cp1=fort/'r0a_cp1'; cp2=fort/'r0a_cp2'; cp3=fort/'r0a_cp3'
sys.path[:0]=[str(cp1/'src'),str(cp2/'src'),str(cp3/'src')]
from rcf_fortification_cp3 import QualifiedStraightWallSpanPublisher
from rcf_fortification_cp3.coverage import canonical_json_bytes, sha256_bytes, tree_digest

fixture=json.loads((cp2/'fixtures/straight_wall_span.fixture.json').read_text())
publisher=QualifiedStraightWallSpanPublisher(cp0)
out=cp3/'outputs'; failures=cp3/'failures'
if out.exists(): shutil.rmtree(out)
if failures.exists(): shutil.rmtree(failures)
out.mkdir(parents=True); failures.mkdir(parents=True)

A=out/'A'; B=out/'B'
ra=publisher.execute(fixture,A,failures,failure_case='positive_A')
rb=publisher.execute(fixture,B,failures,failure_case='positive_B')
if ra['status']!='SUCCEEDED' or rb['status']!='SUCCEEDED': raise SystemExit('positive run failed')

def files(root):
 return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
fa,fb=files(A),files(B)
clean={"schema":"royal-capital.fortification.cp3-clean-replay/1","status":"PASS" if fa==fb else "FAIL","files_compared":len(fa),"byte_identical":fa==fb,"different":sorted(k for k in set(fa)|set(fb) if fa.get(k)!=fb.get(k))}
(cp3/'reports').mkdir(exist_ok=True)
(cp3/'reports/clean_replay.json').write_text(json.dumps(clean,indent=2,sort_keys=True)+'\n')
if fa!=fb: raise SystemExit('clean replay mismatch')
accepted_before=tree_digest(A)
correct_digest=sha256_bytes(canonical_json_bytes(fixture))

cases=[]
def run_case(name, mutate=None, **kwargs):
 f=copy.deepcopy(fixture)
 if mutate: mutate(f)
 target=work/'negative-targets'/name
 result=publisher.execute(f,target,failures,failure_case=name,**kwargs)
 cases.append({"case":name,"status":result['status'],"code":result.get('failure',{}).get('code'),"partial_output_published":result.get('partial_output_published'),"target_exists":target.exists(),"failure_dir":(failures/name).relative_to(cp3).as_posix(),"accepted_target_unchanged":result.get('accepted_target_unchanged')})
 if result['status']!='REJECTED' or target.exists() or result.get('partial_output_published') is not False: raise SystemExit(f'negative gate failed {name}: {result}')

run_case('invalid_nonfinite',lambda f:f.__setitem__('length_m',float('nan')))
run_case('nonpositive_length',lambda f:f.__setitem__('length_m',0.0))
run_case('invalid_units',lambda f:f.__setitem__('frame','BAD_FRAME'))
run_case('invalid_tolerance',lambda f:f['tolerances'].__setitem__('tessellation_linear_m',0.2))
run_case('stale_input',expected_fixture_digest='sha256:'+'0'*64)
run_case('runtime_mismatch',lambda f:f['runtime'].__setitem__('ocp_version','0.0.0'))
run_case('geometry_budget',lambda f:f['budget'].__setitem__('max_vertices',1))
run_case('output_byte_budget',max_published_bytes=1)
run_case('required_part_failure',fault='REQUIRED_PART_FAILED')
run_case('coverage_gap',fault='COVERAGE_GAP')
run_case('coverage_overlap',fault='COVERAGE_OVERLAP')

# Fail against an already accepted target; accepted bytes must remain unchanged.
result=publisher.execute(fixture,A,failures,failure_case='accepted_target_preservation',fault='REQUIRED_PART_FAILED')
cases.append({"case":"accepted_target_preservation","status":result['status'],"code":result['failure']['code'],"partial_output_published":result['partial_output_published'],"target_exists":A.exists(),"accepted_target_unchanged":result['accepted_target_unchanged']})
if result['status']!='REJECTED' or not A.exists() or not result['accepted_target_unchanged'] or tree_digest(A)!=accepted_before: raise SystemExit('accepted target mutated')

matrix={"schema":"royal-capital.fortification.cp3-negative-gates/1","status":"PASS","case_count":len(cases),"cases":cases,"accepted_output_digest":accepted_before,"all_rejected":all(c['status']=='REJECTED' for c in cases),"no_negative_target_published":all((not c['target_exists']) or c['case']=='accepted_target_preservation' for c in cases)}
(cp3/'reports/negative_gates.json').write_text(json.dumps(matrix,indent=2,sort_keys=True)+'\n')
shutil.copytree(A,cp3/'outputs/reference')
summary={"schema":"royal-capital.fortification.cp3-qualification/1","status":"PASS","clean_replay":clean,"negative_gates":matrix,"coverage":json.loads((A/'source-coverage-map.json').read_text())["summary"],"accepted_reference_digest":tree_digest(cp3/'outputs/reference')}
(cp3/'reports/qualification.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))
