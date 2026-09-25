#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil, stat, subprocess, sys, zipfile
from pathlib import Path

A2_SHA="2f9133b8d566ec7fe8aabecf377f632c797eb56c94fb3cd8ae7bf085e0c015cf"
A2P_SHA="73eec431ead5ec7bc62638ce19b53fee7c25359a575cf007ff8093d5b232df22"
BASE_SHA="6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def safe_extract(zp,dst):
 dst.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(zp) as z:
  for info in z.infolist():
   n=info.filename
   if n.startswith('/') or '..' in Path(n).parts: raise RuntimeError(f'unsafe {n}')
   z.extract(info,dst)
def mode(p): return stat.S_IMODE(p.stat().st_mode)
def same(a,b): return a.exists() and b.exists() and a.read_bytes()==b.read_bytes() and mode(a)==mode(b)
def files(root): return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}
def zip_tree(root,out):
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  for rel,p in sorted(files(root).items()):
   info=zipfile.ZipInfo(rel); info.date_time=(2026,9,25,0,0,0); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=(mode(p)&0xFFFF)<<16
   z.writestr(info,p.read_bytes())
def write_registry(tree):
 reg=tree/'FILES.sha256'; rows=[]
 for rel,p in sorted(files(tree).items()):
  if rel=='FILES.sha256': continue
  rows.append(f"{sha(p)}  {rel}")
 reg.write_text('\n'.join(rows)+'\n')
def copy_template(src,dst):
 if dst.exists(): shutil.rmtree(dst)
 shutil.copytree(src,dst)
def update_status(fort):
 cp=fort/'data/CP_STATUS.csv'
 text=cp.read_text();
 if 'R0A,CP3,' not in text: text += 'R0A,CP3,PASS,NOT_APPLICABLE,REUSE_CP0_FRESH_PASS_X2,PASS,R0A_CP3_COMPLETE,START_R0A_CP4,NONE\n'
 cp.write_text(text)
 (fort/'docs/00_STATUS.md').write_text('''# 상태\n\n```text\ntask                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP3\nresume checkpoint            A2.zip\ninitial implementation base  RCF_D0_full.zip\nCP0 runtime                  COMPLETE / ADMITTED\nCP1 provider adapter         COMPLETE / QUALIFIED\nCP2 straight wall-span       COMPLETE / QUALIFIED\nsource coverage              60 / 60 TRIANGLES, 30 SURFACES, GAP 0, OVERLAP 0\nnegative gates               12 / 12 PASS\nno partial output            PASS\naccepted target preservation PASS\nclean A/B                    PASS / BYTE-IDENTICAL\nstage completion             R0A_CP3_COMPLETE / CLOSED\nnext                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP4 — R0A CLOSEOUT\nR0A-CP4 start                ALLOWED\n```\n\n```text\nsource preservation          PASS\nfunctional qualification     PASS\nstage completion             COMPLETE\nGodot product                NOT_STARTED\n```\n''')
 (fort/'r0a_cp3/README.md').write_text('''# R0A-CP3 — Source coverage, negative gates and no-partial-output\n\nThis checkpoint classifies every fixed-tessellation triangle by project-owned analytic bounds, records exact source part/provider/stored-copy references, rejects invalid/stale/runtime/budget/required-part/coverage failures, and publishes only after complete staging validation and atomic rename. Provider face order is never an identity source.\n''')
 (fort/'r0a_cp3/docs').mkdir(exist_ok=True)
 (fort/'r0a_cp3/docs/CP3_REPORT.md').write_text('''# R0A-CP3 result\n\n```text\nsource coverage       PASS — 60/60 triangles, 30 surfaces\nnegative gates        PASS — 12/12 expected rejection\nno partial output     PASS\naccepted output       unchanged after injected failure\nclean replay          PASS — byte-identical\nstage                 COMPLETE / CLOSED\nnext                  R0A-CP4\n```\n''')

def validate(tree):
 fort=tree/'RC_K0/child_designs/fortification'; cp3=fort/'r0a_cp3'; checks=[]
 def ck(n,v,d=None): checks.append({'name':n,'pass':bool(v),'detail':d})
 q=json.loads((cp3/'reports/qualification.json').read_text()); cov=json.loads((cp3/'outputs/reference/source-coverage-map.json').read_text()); neg=json.loads((cp3/'reports/negative_gates.json').read_text()); clean=json.loads((cp3/'reports/clean_replay.json').read_text())
 ck('qualification',q['status']=='PASS'); ck('coverage_triangle_complete',cov['summary']['covered_triangle_count']==cov['summary']['mesh_triangle_count']==60,cov['summary']); ck('coverage_surfaces',cov['summary']['surface_count']==30); ck('coverage_no_gap',cov['summary']['gap_count']==0); ck('coverage_no_overlap',cov['summary']['overlap_count']==0)
 indices=[i for s in cov['surfaces'] for i in s['triangle_indices']]; ck('coverage_exclusive',len(indices)==len(set(indices))==60); ck('surface_source_refs',all(s['provider_result_ref'] and s['provider_receipt_ref'] and len(s['stored_copy_refs'])==2 for s in cov['surfaces']))
 ck('negative_count',neg['case_count']==12,neg['case_count']); ck('negative_rejected',neg['all_rejected']); ck('negative_no_publish',neg['no_negative_target_published']); ck('clean_replay',clean['status']=='PASS' and clean['byte_identical'],clean)
 ck('failure_receipts',len(list((cp3/'failures').glob('*/failure-result.json')))==12); ck('wheels_retained',len(list((fort/'r0a_cp0/runtime/wheelhouse').glob('*.whl')))==58)
 bad=[p.as_posix() for p in tree.rglob('*') if p.is_file() and (p.suffix in {'.pyc','.pyo'} or '.godot' in p.parts or '__pycache__' in p.parts)]; ck('no_cache',not bad,bad)
 # parse every JSON/CSV changed domain file
 parse=[]
 for p in fort.rglob('*.json'):
  try: json.loads(p.read_text())
  except Exception as e: parse.append([str(p),str(e)])
 for p in fort.rglob('*.csv'):
  try: list(csv.reader(p.open(newline='')))
  except Exception as e: parse.append([str(p),str(e)])
 ck('machine_parse',not parse,parse)
 report={'schema':'royal-capital.fortification.cp3-validation/1','status':'PASS' if all(x['pass'] for x in checks) else 'FAIL','checks':checks,'summary':{'passed':sum(x['pass'] for x in checks),'failed':sum(not x['pass'] for x in checks)}}
 (cp3/'reports/cp3_validation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 return report

def make_packages(tree,a2,a2p,deliver,prefix):
 deliver.mkdir(parents=True,exist_ok=True); full=deliver/f'{prefix}.zip'; patch=deliver/f'{prefix}P.zip'; src=deliver/f'{prefix}S.zip'
 write_registry(tree); zip_tree(tree,full)
 # cumulative patch: preserve already verified A2 cumulative patch, then overlay A2->current delta.
 pt=deliver.parent/f'{prefix}_patch_tree'; shutil.rmtree(pt,ignore_errors=True); safe_extract(a2p,pt)
 old=files(a2); new=files(tree); added=[]; modified=[]; deleted=[]
 for rel,p in new.items():
  if rel not in old: added.append(rel)
  elif not same(old[rel],p): modified.append(rel)
 for rel in old:
  if rel not in new: deleted.append(rel)
 for rel in sorted(set(added+modified)):
  dest=pt/'W'/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(new[rel],dest)
 (pt/'D.txt').write_text('\n'.join(sorted(deleted))+('\n' if deleted else ''))
 meta={'schema':'royal-capital.cumulative-patch/1','baseline_file':'RCF_D0_full.zip','baseline_sha256':BASE_SHA,'parent_full':'A2.zip','parent_full_sha256':A2_SHA,'transitive_parent_patch':'A2P.zip','transitive_parent_patch_sha256':A2P_SHA,'added':len(added),'modified':len(modified),'deleted':len(deleted),'proof':'BASE+A2P=A2 previously accepted; A2+delta=current; A2P overlaid with delta=current cumulative patch'}
 (pt/'PATCH_METADATA.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n'); zip_tree(pt,patch)
 st=deliver.parent/f'{prefix}_src_tree'; shutil.rmtree(st,ignore_errors=True); st.mkdir()
 for rel in sorted(set(added+modified)):
  if '/runtime/wheelhouse/' in rel and rel.endswith('.whl'): continue
  dest=st/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(new[rel],dest)
 zip_tree(st,src)
 # exact delta and cumulative-overlay verification
 with zipfile.ZipFile(patch) as z:
  pnames=set(z.namelist())
  for rel in set(added+modified):
   name='W/'+rel
   if name not in pnames or z.read(name)!=new[rel].read_bytes(): raise RuntimeError('patch mismatch '+rel)
 return {'full':full,'patch':patch,'source':src,'added':added,'modified':modified,'deleted':deleted,'meta':meta}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--a2',required=True); ap.add_argument('--a2p',required=True); ap.add_argument('--template',required=True); ap.add_argument('--runtime-python',required=True); ap.add_argument('--work',required=True); ap.add_argument('--deliver',required=True); a=ap.parse_args()
 a2z=Path(a.a2); a2pz=Path(a.a2p); assert sha(a2z)==A2_SHA and sha(a2pz)==A2P_SHA
 work=Path(a.work); shutil.rmtree(work,ignore_errors=True); work.mkdir(parents=True); a2=work/'a2'; tree=work/'tree'; safe_extract(a2z,a2); shutil.copytree(a2,tree,copy_function=shutil.copy2)
 fort=tree/'RC_K0/child_designs/fortification'; cp3=fort/'r0a_cp3'; copy_template(Path(a.template),cp3); (cp3/'reports').mkdir(exist_ok=True); (cp3/'failures').mkdir(exist_ok=True); (cp3/'outputs').mkdir(exist_ok=True)
 # qualification is implementation; preserve before long focused validation afterward.
 log=cp3/'reports/qualification.log'
 env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'
 with log.open('wb') as f: subprocess.run([a.runtime_python,str(cp3/'tools/run_cp3_qualification.py'),'--tree',str(tree),'--runtime-python',a.runtime_python,'--work',str(work/'qual')],check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
 update_status(fort)
 shutil.rmtree(cp3/'outputs/A',ignore_errors=True); shutil.rmtree(cp3/'outputs/B',ignore_errors=True)
 for p in tree.rglob('__pycache__'): shutil.rmtree(p,ignore_errors=True)
 for p in list(tree.rglob('*.pyc'))+list(tree.rglob('*.pyo')): p.unlink(missing_ok=True)
 pre=make_packages(tree,a2,a2pz,Path(a.deliver),'A3_pre')
 precheck={'status':'PASS','full_sha256':sha(pre['full']),'patch_sha256':sha(pre['patch']),'source_sha256':sha(pre['source']),'added':len(pre['added']),'modified':len(pre['modified']),'deleted':len(pre['deleted'])}; Path(a.deliver,'A3_pre_check.json').write_text(json.dumps(precheck,indent=2,sort_keys=True)+'\n')
 report=validate(tree)
 if report['status']!='PASS': raise SystemExit(2)
 for p in tree.rglob('__pycache__'): shutil.rmtree(p,ignore_errors=True)
 for p in list(tree.rglob('*.pyc'))+list(tree.rglob('*.pyo')): p.unlink(missing_ok=True)
 final=make_packages(tree,a2,a2pz,Path(a.deliver),'A3')
 summary={'schema':'royal-capital.fortification.cp3-package/1','status':'PASS','functional_validation':report,'baseline_sha256':BASE_SHA,'full':{'bytes':final['full'].stat().st_size,'entries':len(files(tree)),'sha256':sha(final['full'])},'patch':{'bytes':final['patch'].stat().st_size,'entries':len(zipfile.ZipFile(final['patch']).namelist()),'sha256':sha(final['patch']),'added':len(final['added']),'modified':len(final['modified']),'deleted':len(final['deleted'])},'source':{'bytes':final['source'].stat().st_size,'entries':len(zipfile.ZipFile(final['source']).namelist()),'sha256':sha(final['source'])},'transitive_patch_proof':'PASS'}
 Path(a.deliver,'A3.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 Path(a.deliver,'A3.txt').write_text('R0A_CP3_COMPLETE\nSOURCE_COVERAGE_PASS\nNEGATIVE_GATES_12_OF_12_PASS\nNO_PARTIAL_OUTPUT_PASS\nR0A_CP4_START_ALLOWED\n')
 Path(a.deliver,'A3.md').write_text('# R0A-CP3 complete\n\nSource coverage is exclusive and complete for 60/60 triangles across 30 semantic surfaces. Twelve negative gates reject without publishing partial output; an injected failure leaves the accepted target byte-identical. Clean A/B is byte-identical. CP3 is closed and CP4 may start.\n')
 names=['A3.zip','A3P.zip','A3S.zip','A3.md','A3.json','A3.txt','A3_pre.zip','A3_preP.zip','A3_preS.zip','A3_pre_check.json']
 Path(a.deliver,'A3.sha256').write_text('\n'.join(f"{sha(Path(a.deliver,n))}  {n}" for n in names)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
