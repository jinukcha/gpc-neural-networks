#!/usr/bin/env python3
"""Continue the available RCF tree and publish ONE resumable working directory.
No old baseline replay, no mandatory patch, no full CP2 requalification.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

FORT = Path('RC_K0/child_designs/fortification')
CP3 = FORT / 'r0c_cp3'
RUNTIME = FORT / 'r0a_cp0/runtime'

PACKAGER = '''#!/usr/bin/env python3
"""Make one current RCF.zip, replacing the previous ZIP only after successful reopen."""
from __future__ import annotations
import argparse, os, tempfile, zipfile
from pathlib import Path
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', 'node_modules'}
def package(root: Path, output: Path) -> dict:
    root=root.resolve(); output=output.resolve()
    if not (root/'RC_K0').is_dir() or not (root/'STATUS.json').is_file():
        raise ValueError('Expected the current integrated source root, not an empty or old patch directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.rcf-',suffix='.tmp',dir=output.parent);os.close(fd)
    temp=Path(name);count=0
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(root.rglob('*')):
                if p.is_symlink(): raise ValueError(f'Symlink is not a self-contained source file: {p}')
                if not p.is_file() or p.resolve() in (output,temp): continue
                rel=p.relative_to(root)
                if any(x in SKIP for x in rel.parts): continue
                if p.suffix in ('.pyc','.pyo','.log','.patch'): continue
                # Delivery ZIPs are never inputs to the next integrated ZIP. Runtime .whl files remain.
                if p.suffix.lower()=='.zip': continue
                info=zipfile.ZipInfo(rel.as_posix(),date_time=(2026,9,26,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9);count+=1
        with zipfile.ZipFile(temp) as z:
            bad=z.testzip()
            if bad: raise ValueError(f'CRC failure: {bad}')
            if 'STATUS.json' not in z.namelist(): raise ValueError('Missing resume status')
        os.replace(temp,output)
        return {'file':str(output),'bytes':output.stat().st_size,'files':count,'reopen':'PASS'}
    finally:
        temp.unlink(missing_ok=True)
if __name__=='__main__':
    import json
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path)
    args=ap.parse_args();print(json.dumps(package(args.root,args.output or args.root.parent/'RCF.zip'),indent=2))
'''

SMOKE = '''import json, tempfile
from pathlib import Path
from build123d import Box, export_step, export_brep, import_step, import_brep
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp); shape=Box(2,3,4)
    export_step(shape,str(d/'probe.step'));export_brep(shape,str(d/'probe.brep'))
    a=import_step(str(d/'probe.step'));b=import_brep(str(d/'probe.brep'))
    volumes=[float(x.volume) for x in (shape,a,b)]
    assert all(abs(v-24)<1e-7 for v in volumes),volumes
    assert all(x.is_valid for x in (shape,a,b))
    print(json.dumps({'status':'PASS','operation':'build123d Box + STEP/BREP reopen','volumes':volumes}))
'''

POLICY = '''# Current working delivery rule — 2026-09-26

The current integrated source tree and its single `RCF.zip` are the working and handoff authority.
This operational rule supersedes older instructions requiring C3A_preS/P/full, C3AS/AP/R,
byte-identical reconstruction of old C2 ZIPs, or Library publication before implementation.

Do not reconstruct historical ZIPs to start CP3-B. Do not make a patch or receipt archive mandatory.
Keep source edits in this tree. Repackage to the SAME output filename with tools/package_current.py.
A failed optional validation records its real result; it does not erase source or require a restart.
Checkpoints are emergency recovery only. Do not nest old deliveries, failed-output trees, or full logs.

Retain current implementation, design/roadmap, fixtures, actual CAD outputs, licences, reference art,
and the complete pinned Linux CPython 3.13 wheelhouse. Never shrink by silently removing CAD dependencies.
The system CPython interpreter itself is not bundled; use Python 3.13.5 for the inherited exact runtime.
Only a short affected smoke test is expected per change. No Windows or Godot run is required here.

Historical reports are evidence of past attempts, not today's execution prerequisites.
STATUS.json at the integrated root gives the current continuation state. FILES.sha256 is transport integrity,
not a reason to reproduce a previous archive's bytes. CP3 scope, host immutability, finite instance budgets,
shared definitions and no-partial-output still apply. Publishing the scope files does not qualify geometry.
'''

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')

def dump(p: Path, obj: object) -> None:
    write(p,json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def is_generated_evidence(rel: Path) -> bool:
    parts=rel.parts
    return ('reports' in parts and ('history' in parts or any(x.startswith('resume') for x in parts))) or 'failures' in parts

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--tree',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); tree=a.tree;out=a.output
    if not (tree/FORT).is_dir(): raise ValueError('No existing fortification tree; refusing a replacement rebuild')
    source_files=[p for p in tree.rglob('*') if p.is_file()]
    before_bytes=sum(p.stat().st_size for p in source_files)
    # Protect all implementation plus every non-historical CAD output and every pinned wheel.
    protected={p.relative_to(tree).as_posix():sha(p) for p in source_files if p.suffix in ('.py','.gd','.whl') or (p.suffix.lower() in ('.step','.stp','.brep') and not is_generated_evidence(p.relative_to(tree)))}
    overlay=Path('cp3a_source')
    if not (overlay/CP3).is_dir(): raise ValueError('Existing CP3 source overlay missing')
    shutil.copytree(overlay,tree,dirs_exist_ok=True)
    removed=[]
    for p in list(tree.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(tree); reason=None
        if any(x in ('.git','__pycache__','.pytest_cache','.mypy_cache') for x in rel.parts): reason='generated cache'
        elif p.suffix.lower() in ('.log','.pyc','.pyo'): reason='expendable log/cache'
        elif p.suffix.lower()=='.zip' and 'runtime' not in rel.parts and 'upstream_reference' not in rel.parts: reason='nested delivery archive'
        elif is_generated_evidence(rel) and p.suffix.lower() in ('.step','.stp','.brep','.obj','.stl','.ply','.gltf','.glb','.bin'): reason='historical generated geometry; current fixtures and outputs retained'
        elif is_generated_evidence(rel) and p.name in ('neutral-mesh.json','source-coverage-map.json','mesh.json'): reason='historical generated mesh evidence'
        if reason:
            removed.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'reason':reason});p.unlink()
    for p in sorted(tree.rglob('*'),key=lambda x:len(x.parts),reverse=True):
        if p.is_dir() and not any(p.iterdir()):p.rmdir()
    for rel,digest in protected.items():
        if not (tree/rel).is_file() or sha(tree/rel)!=digest: raise RuntimeError('Protected current source changed: '+rel)
    write(tree/'tools/package_current.py',PACKAGER)
    compile(PACKAGER,'package_current.py','exec')
    write(tree/'tools/smoke_cad.py',SMOKE)
    compile(SMOKE,'smoke_cad.py','exec')
    write(tree/CP3/'docs/CURRENT_DELIVERY_RULE.md',POLICY)
    # Amend active entry points instead of pretending old publication gates still apply.
    notice='> Current execution/delivery rule: use one integrated RCF.zip. Historical patch/prevalidation/baseline-replay publication requirements are superseded by r0c_cp3/docs/CURRENT_DELIVERY_RULE.md; implementation may continue from this tree.\n\n'
    amended=[]
    for p in tree.rglob('*.md'):
        rel=p.relative_to(tree)
        if ('roadmap' in p.name.lower() or p==(tree/CP3/'README.md') or p==(tree/CP3/'docs/CP3_SCOPE_FREEZE.md')) and not is_generated_evidence(rel) and 'upstream_reference' not in rel.parts:
            write(p,notice+p.read_text(encoding='utf-8'));amended.append(rel.as_posix())
    contract=json.loads((tree/CP3/'contracts/BATTLEMENT_RHYTHM_CONTRACT.json').read_text())
    with (tree/CP3/'data/CP3_FIXTURE_MATRIX.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if len(rows)!=4: raise ValueError('CP3 fixture matrix must remain exactly four rows')
    if contract.get('schema')!='royal-capital.fortification.battlement-rhythm/1':raise ValueError('Unexpected CP3 contract')
    matrix_text=(tree/CP3/'data/CP3_FIXTURE_MATRIX.csv').read_text()
    for fixture in ('R0C_CP3_FX01','R0C_CP3_FX02','R0C_CP3_FX03','R0C_CP3_FX04'):
        if fixture not in matrix_text:raise ValueError('Missing fixture '+fixture)
    inv=json.loads(Path('single_result/inventory.json').read_text())
    intake={k:inv[k] for k in ('selected_source','source_zip_bytes','source_zip_sha256')}
    smoke={'status':'NOT_RUN'}
    with tempfile.TemporaryDirectory(prefix='rcf-smoke-') as tmp:
        env=Path(tmp)/'runtime'
        try:
            subprocess.run([sys.executable,'-m','venv',str(env)],check=True,capture_output=True,text=True,timeout=60)
            py=env/'bin/python'
            cmd=[str(py),'-m','pip','install','--no-index','--no-cache-dir','--find-links',str(tree/RUNTIME/'wheelhouse'),'--require-hashes','-r',str(tree/RUNTIME/'requirements-lock.txt')]
            install=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=240)
            check=subprocess.run([str(py),'-m','pip','check'],check=True,capture_output=True,text=True,timeout=30)
            run=subprocess.run([str(py),str(tree/'tools/smoke_cad.py')],check=True,capture_output=True,text=True,timeout=60)
            smoke=json.loads(run.stdout.strip().splitlines()[-1]);smoke['offline_install']='PASS';smoke['pip_check']=check.stdout.strip();smoke['python']=sys.version.split()[0]
        except (subprocess.SubprocessError, ValueError) as e:
            smoke={'status':'FAILED','error':str(e),'stderr':str(getattr(e,'stderr',''))[-3500:],'cp3_source_retained':True}
    policy_receipt={'status':'SOURCE_INTEGRATED','scope_contract_matrix':'PASS','fixture_count':4,'geometry_qualification':'NOT_STARTED','delivery':'SINGLE_INTEGRATED_FILE','historical_publication_gate':'SUPERSEDED_BY_USER_REQUEST','actual_intake':intake,'accepted_C2_zip_byte_identity':'NOT_CLAIMED','notes':'Continued the latest available full working source. No historical C2 replay or patch reconstruction was performed.'}
    dump(tree/CP3/'reports/CP3A_SOURCE_RECEIPT.json',policy_receipt)
    readme='''# RCF — start here

This is the single integrated continuation tree. Do not request C2P, C2S, C2R,
a fixed baseline, or an old recovery ZIP to continue work.

## Current work
CP3-A scope, rhythm contract and four-row fixture matrix are integrated into the actual source tree.
CP3 geometry/module instancing is not implemented or qualified by this delivery.
The next implementation task is **CP3-B / R0C_CP3_FX01 — linear battlement pilot**.
The round, square and polygonal perimeter fixtures remain later CP3 work; socket redesign remains CP4.
Godot product work is NOT_STARTED.

## Continue
Read STATUS.json, then RC_K0/child_designs/fortification/r0c_cp3/README.md and its contract/matrix.
Edit the existing tree. Do not reimplement or replay ancestors merely to restore old archive hashes.
The intake is the latest full available working checkpoint, C2_pre.zip, not a falsely relabelled exact C2.zip.
Historical accepted C2 evidence remains historical; this delivery does not revoke or re-certify CP2.

## Runtime (Linux x86_64)
Python 3.13.5 is required for the inherited exact runtime. The interpreter itself is not included.
All original pinned wheels, including materials data, are retained unchanged for offline package installation.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --no-index --require-hashes \\
  --find-links RC_K0/child_designs/fortification/r0a_cp0/runtime/wheelhouse \\
  -r RC_K0/child_designs/fortification/r0a_cp0/runtime/requirements-lock.txt
.venv/bin/python tools/smoke_cad.py
python3 tools/package_current.py --output ../RCF.zip
```

Only the final command packages a handoff. It atomically replaces ONE RCF.zip after one CRC reopen.
The integration handoff used here is a GitHub artifact ZIP of this directory, not a ZIP containing another ZIP.
Old failure geometry and text logs were omitted; original source, current CAD outputs, fixture inputs,
licences and reference images were retained. Do not run historical all-stage audits as a new start gate.

A failed smoke does not delete this source. Check the actual smoke status in STATUS.json before making
runtime claims. Current delivery rules are in r0c_cp3/docs/CURRENT_DELIVERY_RULE.md under the fortification tree.
'''
    write(tree/'START_HERE.md',readme)
    runtime_files=list((tree/RUNTIME/'wheelhouse').glob('*.whl'))
    status={'project':'ROYAL-CAPITAL-FORTIFICATION-CAD','work':'R0C-CP3 / single integrated source consolidation','source_publication':'COMPLETE','cp3_scope_contract_fixture_files':'INTEGRATED','cp3_functional_qualification':'NOT_STARTED','r0c_cp3':'OPEN','godot_product':'NOT_STARTED','next_task':'CP3-B — R0C_CP3_FX01 LINEAR BATTLEMENT PILOT','actual_source_intake':intake,'accepted_C2_byte_identity':'NOT_CLAIMED_OR_REQUIRED_FOR_CONTINUATION','protected_current_source_files':len(protected),'protected_current_source_unchanged':True,'runtime_smoke':smoke,'delivery_policy':{'integrated_files':1,'patch_required':False,'baseline_replay_required':False,'prevalidation_zip_required':False,'library_publication_is_start_gate':False},'size':{'input_uncompressed_bytes':before_bytes,'removed_files':len(removed),'removed_uncompressed_bytes':sum(r['bytes'] for r in removed),'runtime_wheel_count':len(runtime_files),'runtime_wheel_bytes':sum(p.stat().st_size for p in runtime_files),'runtime_wheels_removed':0},'amended_active_docs':amended,'removed_by_reason':{r:sum(x['bytes'] for x in removed if x['reason']==r) for r in sorted({x['reason'] for x in removed})},'validation_scope':'four-row contract parsing; protected-current-source comparison; one offline CAD STEP/BREP smoke; no CP2 requalification'}
    dump(tree/'STATUS.json',status)
    # Keep an existing conventional transport inventory, with no replay chain or new manifest layer.
    manifest=tree/'FILES.sha256'
    entries=[p for p in tree.rglob('*') if p.is_file() and p!=manifest]
    write(manifest,''.join(sha(p)+'  '+p.relative_to(tree).as_posix()+'\n' for p in sorted(entries)))
    files=[p for p in tree.rglob('*') if p.is_file()]
    status_summary=dict(status);status_summary['size']=dict(status['size'],output_files=len(files),output_uncompressed_bytes=sum(p.stat().st_size for p in files))
    dump(Path('single_result/result.json'),status_summary)
    # This is the only materialized deliverable directory. Actions makes the single final ZIP.
    if out.exists():shutil.rmtree(out)
    shutil.copytree(tree,out)
    print(json.dumps(status_summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
