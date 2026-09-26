#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, os, stat
from pathlib import Path
from typing import Any

p=argparse.ArgumentParser(); p.add_argument('--tree',required=True); p.add_argument('--report',required=True); a=p.parse_args()
tree=Path(a.tree).resolve(); fort=tree/'RC_K0/child_designs/fortification'; cp4=fort/'r0b_cp4'; report_path=Path(a.report).resolve()
checks=[]
def ck(name:str, passed:bool, detail:Any=None): checks.append({'name':name,'pass':bool(passed),'detail':detail})
def load(path:Path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()

completion=load(cp4/'reports/cp4_completion.json')
qualification=load(cp4/'reports/qualification.json')
focused=load(cp4/'reports/r0b_cp4_validation.json')
unittest=load(cp4/'reports/unittest.json')
result=load(cp4/'outputs/reference/result.json')
seg=load(cp4/'outputs/reference/segmentation-plan.json')
coverage=load(cp4/'outputs/reference/source-coverage-map.json')
alignment=load(cp4/'outputs/reference/socket-alignment.json')
receipt=load(cp4/'outputs/reference/cad-provider-receipt.json')
runtime=load(cp4/'reports/runtime-reuse-receipt.json')

ck('completion_status',completion['status']=='PASS')
ck('stage_completion',completion['stage_completion']=='R0B_COMPLETE')
ck('r0b_progress',completion['r0b_checkpoint_progress']=='4/4')
ck('next_explicit',completion['next']=='R0C_CP1_BY_EXPLICIT_APPROVAL' and completion['auto_continuation'] is False)
ck('source_preservation',completion['source_preservation']=='PASS')
ck('functional_qualification',completion['functional_qualification']=='PASS')
ck('godot_not_started',completion['godot_product']=='NOT_STARTED')
ck('collision_not_started',completion['collision']=='NOT_STARTED')
ck('navigation_not_started',completion['navigation']=='NOT_STARTED')
ck('product_cook_deferred',completion['product_cook']=='DEFERRED_TO_R0E')

ck('qualification_pass',qualification['status']=='PASS')
ck('focused_validation_pass',focused['status']=='PASS' and focused['summary']['failed']==0 and focused['summary']['passed']==58,focused['summary'])
ck('unittest_pass',unittest['status']=='PASS' and unittest['exit_code']==0)
ck('source_replay',qualification['source_replay']['status']=='PASS' and qualification['source_replay']['case_count']==7 and all(x['byte_identical'] for x in qualification['source_replay']['cases']))
ck('clean_replay',qualification['clean_replay']['status']=='PASS' and qualification['clean_replay']['byte_identical'])
ck('negative_gates',qualification['negative_gates']['status']=='PASS' and qualification['negative_gates']['case_count']==10 and qualification['negative_gates']['all_rejected'])

ck('result_status',result['status']=='SUCCEEDED')
ck('module_counts',result['module_count']==7 and result['span_segment_count']==4 and result['join_count']==3 and result['connection_count']==6)
ck('mesh_counts',result['vertex_count']==544 and result['triangle_count']==908 and result['solid_count']==45)
ck('segmentation_digest',result['segmentation_digest']==seg['segmentation_digest']==receipt['segmentation_digest'])
ck('segmentation_order_not_identity',seg['input_array_order_is_identity'] is False)
ck('segmentation_chain',seg['chain_order']==['span/straight','join/miter','span/curved','join/bevel','span/stepped','span/retaining','join/transition'])
ck('coverage_complete',coverage['summary']['mesh_triangle_count']==coverage['summary']['covered_triangle_count']==coverage['summary']['unique_triangle_count']==908)
ck('coverage_no_gap_overlap',coverage['summary']['gap_count']==0 and coverage['summary']['overlap_count']==0)
ck('coverage_surfaces',coverage['summary']['surface_count']==270)
ck('provider_face_not_identity',coverage['provider_face_identity_used'] is False and receipt['provider_face_identity_used'] is False)
ck('socket_alignment',alignment['status']=='PASS' and alignment['alignment_count']==6)
ck('global_boolean_false',receipt['global_boolean_used'] is False)

ck('runtime_reuse',runtime['status']=='PASS' and runtime['python']=='3.13.5' and runtime['build123d']=='0.13.1.dev12+ge22d34dae' and runtime['cadquery_ocp_novtk']=='8.0.1.0.0')
ck('runtime_wheels',len(list((fort/'r0a_cp0/runtime/wheelhouse').glob('*.whl')))==58)
ck('runtime_lock',digest(fort/'r0a_cp0/runtime/requirements-lock.txt')=='59b8802bc325c94b4d4e94060e4ed9490ce738fff6ecb8726f86eadb779744bf')
ck('runtime_manifest',digest(fort/'r0a_cp0/runtime/WHEELHOUSE.json')=='f6e9908f02cbd6b689ff0448a32ca460e374afcbeb6c66d34129ec74f07c51cd')

rows=list(csv.DictReader((fort/'data/CP_STATUS.csv').open(encoding='utf-8',newline='')))
expected={(f'R0A',f'CP{i}') for i in range(5)}|{('R0B',f'CP{i}') for i in range(1,5)}
observed={(row['stage'],row['checkpoint']) for row in rows}
ck('status_rows_complete',expected<=observed,{'missing':sorted(expected-observed)})
r0b4=next((row for row in rows if row['stage']=='R0B' and row['checkpoint']=='CP4'),None)
ck('status_row_cp4',bool(r0b4) and r0b4['stage_completion']=='R0B_COMPLETE' and r0b4['next']=='START_R0C_CP1_BY_EXPLICIT_APPROVAL',r0b4)
status_text=(fort/'docs/00_STATUS.md').read_text(encoding='utf-8')
ck('status_text_r0b_closed','R0B_COMPLETE / CLOSED' in status_text)
ck('status_text_no_false_product','Godot product                 NOT_STARTED' in status_text and 'collision                     NOT_STARTED' in status_text and 'navigation                    NOT_STARTED' in status_text)
roadmap=(fort/'docs/06_ROADMAP.md').read_text(encoding='utf-8')
ck('roadmap_next_explicit','CURRENT                 RC-FORT-R0B COMPLETE / CLOSED' in roadmap and 'NEXT                    RC-FORT-R0C-CP1 — EXPLICIT APPROVAL REQUIRED' in roadmap)
ck('no_r0c_implementation',not (fort/'r0c_cp1').exists())

required=[
 cp4/'README.md',cp4/'docs/CP4_REPORT.md',cp4/'docs/R0B_CLOSEOUT.md',cp4/'reports/cp4_completion.json',cp4/'reports/cp4_completion.txt',
 cp4/'reports/qualification.json',cp4/'reports/r0b_cp4_validation.json',cp4/'reports/unittest.json',cp4/'outputs/reference/result.json',
]
ck('required_closeout_files',all(x.is_file() for x in required),[str(x) for x in required if not x.is_file()])

parse_errors=[]
for path in fort.rglob('*.json'):
 try: json.loads(path.read_text(encoding='utf-8'))
 except Exception as exc: parse_errors.append([path.relative_to(tree).as_posix(),str(exc)])
for path in fort.rglob('*.csv'):
 try: list(csv.reader(path.open(encoding='utf-8',newline='')))
 except Exception as exc: parse_errors.append([path.relative_to(tree).as_posix(),str(exc)])
ck('machine_parse',not parse_errors,parse_errors[:20])

cache=[]; nested=[]; symlinks=[]; unsafe=[]
for path in tree.rglob('*'):
 rel=path.relative_to(tree).as_posix()
 if path.is_symlink(): symlinks.append(rel)
 if path.is_file():
  if path.suffix in {'.pyc','.pyo'} or '__pycache__' in path.parts or '.godot' in path.parts: cache.append(rel)
  if path.suffix.lower()=='.zip': nested.append(rel)
  if rel.startswith('/') or '..' in Path(rel).parts: unsafe.append(rel)
ck('no_cache',not cache,cache)
ck('no_nested_bundle',not nested,nested)
ck('no_symlink',not symlinks,symlinks)
ck('no_unsafe_path',not unsafe,unsafe)

# Source and license/provenance retention.
ck('licenses_retained',all((fort/'licenses'/name).is_file() for name in ['build123d-LICENSE.txt','build123d-NOTICE.txt','cadquery-LICENSE.txt','building_tools-LICENSE.txt']))
ck('provenance_retained',(fort/'provenance/SOURCE_ARCHIVE_RECEIPTS.csv').is_file() and (cp4/'provenance/CHANGESET.json').is_file())

report={'schema':'royal-capital.fortification.r0b-closeout-validation/1','status':'PASS' if all(x['pass'] for x in checks) else 'FAIL','checks':checks,'summary':{'passed':sum(x['pass'] for x in checks),'failed':sum(not x['pass'] for x in checks),'total':len(checks)},'r0b_complete':all(x['pass'] for x in checks),'next':'R0C_CP1_BY_EXPLICIT_APPROVAL'}
report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2,sort_keys=True))
raise SystemExit(0 if report['status']=='PASS' else 1)
