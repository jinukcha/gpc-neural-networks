#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, shutil, sys
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('--tree',required=True); ap.add_argument('--work',required=True); args=ap.parse_args()
tree=Path(args.tree).resolve(); work=Path(args.work).resolve(); work.mkdir(parents=True,exist_ok=True)
fort=tree/'RC_K0/child_designs/fortification'; cp0=fort/'r0a_cp0'; cp1=fort/'r0a_cp1'; cp2=fort/'r0b_cp2'
sys.path[:0]=[str(cp1/'src'),str(cp2/'src')]
from rcf_fortification_joins import FortificationJoinProducer, JoinError
from rcf_fortification_joins.geometry import overlap_report

outputs=cp2/'outputs'; failures=cp2/'failures'; reports=cp2/'reports'
for d in (outputs,failures,reports): d.mkdir(parents=True,exist_ok=True)
producer=FortificationJoinProducer(cp0)
fixtures={'miter':cp2/'fixtures/miter_12deg.fixture.json','bevel':cp2/'fixtures/bevel_30deg.fixture.json','profile_transition':cp2/'fixtures/profile_transition.fixture.json'}
expected={'miter':10,'bevel':15,'profile_transition':5}
def fmap(root): return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
qualified=[]
for name,path in fixtures.items():
    fixture=json.loads(path.read_text()); aout=outputs/'runs'/name/'A'; bout=outputs/'runs'/name/'B'
    ra=producer.execute(fixture,aout,failures,failure_case=f'{name}_A'); rb=producer.execute(fixture,bout,failures,failure_case=f'{name}_B')
    if ra.get('status')!='SUCCEEDED' or rb.get('status')!='SUCCEEDED': raise SystemExit(f'{name} positive failure')
    fa,fb=fmap(aout),fmap(bout); result=json.loads((aout/'result.json').read_text()); align=json.loads((aout/'socket-alignment.json').read_text()); overlap=json.loads((aout/'bounded-overlap.json').read_text()); parts=json.loads((aout/'semantic-parts.json').read_text()); stored=json.loads((aout/'stored-copies.json').read_text())
    checks={'clean_replay':fa==fb,'component_count':result['component_count']==expected[name],'part_count':len(parts['parts'])==5,'socket_alignment':align['status']=='PASS','bounded_overlap':overlap['status']=='PASS','geometry_nonempty':result['vertex_count']>0 and result['triangle_count']>0 and result['volume_m3']>0,'stored_copies':len(stored['copies'])==2*expected[name]}
    if not all(checks.values()): raise SystemExit(f'{name} checks failed {checks}')
    ref=outputs/'reference'/name; ref.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(aout,ref)
    clean={'schema':'royal-capital.fortification.r0b-cp2-clean-replay/1','status':'PASS','family':fixture['join_family'],'files_compared':len(fa),'byte_identical':fa==fb,'different':sorted(k for k in set(fa)|set(fb) if fa.get(k)!=fb.get(k))}
    (reports/f'clean_replay_{name}.json').write_text(json.dumps(clean,indent=2,sort_keys=True)+'\n')
    qualified.append({'name':name,'result':result,'checks':checks,'clean_replay':clean})

negative=[]
def reject(name,fixture,code):
    target=work/'negative'/name; result=producer.execute(fixture,target,failures,failure_case=name); observed=result.get('failure',{}).get('code')
    ok=result.get('status')=='REJECTED' and observed==code and not target.exists() and result.get('partial_output_published') is False
    negative.append({'case':name,'expected':code,'observed':observed,'pass':ok})
    if not ok: raise SystemExit(f'negative failed {name}: {result}')
m=json.loads(fixtures['miter'].read_text()); b=json.loads(fixtures['bevel'].read_text()); t=json.loads(fixtures['profile_transition'].read_text())
x=copy.deepcopy(m); x['incoming_socket']['inside']=[0,0,-1]; x['incoming_socket']['outside']=[0,0,1]; reject('invalid_frame',x,'JoinFailure.INVALID_FRAME')
x=copy.deepcopy(b); x['join_family']='MITER'; reject('miter_turn_too_large',x,'JoinFailure.TURN_OUT_OF_DOMAIN')
x=copy.deepcopy(m); x['join_family']='BEVEL'; x['bevel_setback_m']=0.5; reject('bevel_turn_too_small',x,'JoinFailure.TURN_OUT_OF_DOMAIN')
x=copy.deepcopy(t); x['outgoing_profile']=x['incoming_profile']; reject('transition_same_profile',x,'JoinFailure.PROFILE_TRANSITION_REQUIRED')
x=copy.deepcopy(m); x['outgoing_socket']['position_m'][0]+=0.25; reject('socket_mismatch',x,'JoinFailure.SOCKET_GEOMETRY_MISMATCH')
x=copy.deepcopy(m); x['budget']['max_components']=1; reject('component_budget',x,'JoinFailure.GEOMETRY_BUDGET_EXCEEDED')
x=copy.deepcopy(m); x['runtime']['ocp_version']='0.0.0'; reject('runtime_mismatch',x,'JoinFailure.RUNTIME_MISMATCH')
x=copy.deepcopy(m); x['join_family']='UNKNOWN'; reject('unknown_family',x,'JoinFailure.INVALID_REQUEST')
accepted=outputs/'reference/miter'; before=fmap(accepted); x=copy.deepcopy(m); x['runtime']['ocp_version']='0.0.0'; result=producer.execute(x,accepted,failures,failure_case='accepted_target_preservation'); ok=result.get('status')=='REJECTED' and before==fmap(accepted) and result.get('accepted_target_unchanged') is True
negative.append({'case':'accepted_target_preservation','expected':'unchanged','observed':result.get('accepted_target_unchanged'),'pass':ok})
if not ok: raise SystemExit('accepted target changed')
fake_spec=copy.deepcopy(m); fake_spec['max_overlap_m']=0.01; square=[[-1,-1],[1,-1],[1,1],[-1,1]]; fake=[{'component_id':'a','part_id':'wall_body','kind':'PLAN_EXTRUSION','polygon_xz':square,'y_min':0,'y_max':1},{'component_id':'b','part_id':'wall_body','kind':'PLAN_EXTRUSION','polygon_xz':square,'y_min':0,'y_max':1}]
try: overlap_report(fake_spec,fake); overlap_ok=False; overlap_code='NONE'
except JoinError as exc: overlap_ok=exc.code=='JoinFailure.OVERLAP_BUDGET_EXCEEDED'; overlap_code=exc.code
negative.append({'case':'bounded_overlap_exceeded','expected':'JoinFailure.OVERLAP_BUDGET_EXCEEDED','observed':overlap_code,'pass':overlap_ok})
if not overlap_ok: raise SystemExit('overlap gate failed')
neg={'schema':'royal-capital.fortification.r0b-cp2-negative-gates/1','status':'PASS','case_count':len(negative),'cases':negative,'all_pass':all(x['pass'] for x in negative)}
(reports/'negative_gates.json').write_text(json.dumps(neg,indent=2,sort_keys=True)+'\n')
summary={'schema':'royal-capital.fortification.r0b-cp2-qualification/1','status':'PASS','qualified_families':qualified,'negative_gates':neg}
(reports/'qualification.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','families':list(fixtures),'negative_cases':len(negative)},indent=2))
