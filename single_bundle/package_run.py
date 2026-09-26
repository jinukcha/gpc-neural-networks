#!/usr/bin/env python3
"""Continue the available RCF source; no ancestor replay or mandatory patch chain."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
FORT=Path('RC_K0/child_designs/fortification')
CP3=FORT/'r0c_cp3'
RUNTIME=FORT/'r0a_cp0/runtime'
PACKAGER=r'''#!/usr/bin/env python3
"""Replace ONE RCF.zip only after a successful reopen; refresh transport inventory."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile, zipfile
from pathlib import Path
SKIP={'.git','.venv','__pycache__','.pytest_cache','.mypy_cache','node_modules'}
def package(root: Path, output: Path) -> dict:
    root=root.resolve();output=output.resolve()
    if not (root/'RC_K0').is_dir() or not (root/'STATUS.json').is_file():
        raise ValueError('Expected the current integrated source root, not a patch or empty directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.rcf-',suffix='.tmp',dir=output.parent);os.close(fd)
    temp=Path(name)
    try:
        payload={}
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if any(x in SKIP for x in rel.parts):continue
            if p.is_symlink():raise ValueError(f'Symlink is not self-contained: {rel}')
            if not p.is_file() or p.resolve() in (output,temp):continue
            if p.suffix.lower() in ('.pyc','.pyo','.log','.patch','.zip'):continue
            if rel.as_posix()=='FILES.sha256':continue
            payload[rel.as_posix()]=p
        entries=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+rel+'\n' for rel,p in sorted(payload.items()))
        inventory=root/'FILES.sha256'
        inventory.write_text(entries,encoding='utf-8')
        payload['FILES.sha256']=inventory
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for rel,p in sorted(payload.items()):
                info=zipfile.ZipInfo(rel,date_time=(2026,9,26,0,0,0));info.external_attr=0o100644<<16
                z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
        with zipfile.ZipFile(temp) as z:
            bad=z.testzip()
            if bad:raise ValueError('CRC failure: '+bad)
            if 'STATUS.json' not in z.namelist():raise ValueError('Missing resume status')
        os.replace(temp,output)
        return {'file':str(output),'bytes':output.stat().st_size,'files':len(payload),'reopen':'PASS'}
    finally:temp.unlink(missing_ok=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path)
    a=ap.parse_args();print(json.dumps(package(a.root,a.output or a.root.parent/'RCF.zip'),indent=2))
'''
SMOKE='''import json,tempfile
from pathlib import Path
from build123d import Box,export_step,export_brep,import_step,import_brep
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);shape=Box(2,3,4)
    export_step(shape,str(d/'probe.step'));export_brep(shape,str(d/'probe.brep'))
    a=import_step(str(d/'probe.step'));b=import_brep(str(d/'probe.brep'))
    volumes=[float(x.volume) for x in (shape,a,b)]
    assert all(abs(v-24)<1e-7 for v in volumes),volumes
    assert all(x.is_valid for x in (shape,a,b))
    print(json.dumps({'status':'PASS','operation':'build123d Box + STEP/BREP reopen','volumes':volumes}))
'''
POLICY='''# Current delivery rule — 2026-09-26

The current integrated source tree and its ONE RCF.zip are the working and handoff authority.
This supersedes old requirements for preS/preP/pre-full, S/P/R delivery sets,
byte-identical reconstruction of old C2 ZIPs, or Library publication before implementation.
Do not reconstruct ancestors to start CP3-B. Keep source edits here and replace the same RCF.zip.
Emergency checkpoints are not mandatory gates; do not nest previous deliveries or full logs.
A failed optional check records its actual result and does not erase source or restart the project.

Retain implementation, design/roadmap, fixtures, current CAD outputs, licences, reference art and
all pinned runtime wheels. Never shrink by silently deleting CAD dependencies or changing artwork.
Use the inherited Linux x86_64 CPython 3.13.5 runtime. The interpreter itself is not bundled.
One affected smoke test is sufficient for this consolidation; no CP2 requalification or Windows run.

STATUS.json at the integrated root gives today's continuation state. FILES.sha256 is transport
integrity only, not a reason to reproduce previous archive bytes. Old reports are historical evidence.
CP3 scope, shared definitions, finite budgets, host immutability and no-partial-output still apply.
Publishing scope files is not geometry qualification. CP3-B can proceed from this tree now.
'''
README='''# RCF — start here

This is the single integrated continuation tree. Do not request old C2P/C2S/C2R,
a fixed baseline, or a recovery ZIP to continue implementation.

## Current state
CP3-A scope, rhythm contract and its four-row fixture matrix are integrated in the source tree.
CP3 geometry/module instancing is NOT implemented or qualified by this consolidation.
Next: **CP3-B / R0C_CP3_FX01 — linear battlement pilot**. Closed-perimeter fixtures follow within CP3;
socket redesign and the second profile remain CP4. Godot product is NOT_STARTED.

Read STATUS.json and RC_K0/child_designs/fortification/r0c_cp3/README.md, contract and matrix.
Edit this tree rather than reimplementing ancestors. The actual intake is the latest available
full working checkpoint C2_pre.zip, NOT a falsely relabelled exact accepted C2.zip.
This delivery neither revokes nor re-certifies the historical CP2 acceptance.

## Linux runtime
Python 3.13.5 is required for the inherited exact runtime; the interpreter itself is not bundled.
Every original pinned wheel, including materials data, is unchanged and retained for offline install.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --no-index --require-hashes --find-links RC_K0/child_designs/fortification/r0a_cp0/runtime/wheelhouse -r RC_K0/child_designs/fortification/r0a_cp0/runtime/requirements-lock.txt
.venv/bin/python tools/smoke_cad.py
python3 tools/package_current.py --output ../RCF.zip
```

The last command replaces ONE RCF.zip after one CRC reopen and refreshes FILES.sha256.
It never requires a baseline or patch and never nests an earlier ZIP or .venv.
This handoff is a ZIP of the integrated directory itself, not a ZIP containing another ZIP.
Historical failure geometry and disposable logs were omitted; fixture inputs and current outputs remain.
Do not rerun all old qualification stages as an admission requirement. Actual smoke result is in STATUS.json.
Current delivery rules are in r0c_cp3/docs/CURRENT_DELIVERY_RULE.md under the fortification directory.
'''
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def write(p:Path,s:str)->None:
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def dump(p:Path,obj:object)->None:write(p,json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def historical(rel:Path)->bool:
    return ('reports' in rel.parts and ('history' in rel.parts or any(x.startswith('resume') for x in rel.parts))) or 'failures' in rel.parts

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--tree',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();tree=a.tree
    if not (tree/FORT).is_dir():raise ValueError('Existing fortification source missing; refusing a replacement rebuild')
    original=[p for p in tree.rglob('*') if p.is_file()]
    before_bytes=sum(p.stat().st_size for p in original)
    protected={p.relative_to(tree).as_posix():sha(p) for p in original if p.suffix in ('.py','.gd','.whl') or (p.suffix.lower() in ('.step','.stp','.brep') and not historical(p.relative_to(tree)))}
    overlay=Path('cp3a_source')
    if not (overlay/CP3).is_dir():raise ValueError('Existing CP3 source overlay missing')
    shutil.copytree(overlay,tree,dirs_exist_ok=True)
    removed=[]
    for p in list(tree.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(tree);reason=None
        if any(x in ('.git','__pycache__','.pytest_cache','.mypy_cache') for x in rel.parts) or p.suffix.lower() in ('.log','.pyc','.pyo'):reason='disposable log/cache'
        elif p.suffix.lower()=='.zip' and 'runtime' not in rel.parts and 'upstream_reference' not in rel.parts:reason='nested old delivery'
        elif historical(rel) and p.suffix.lower() in ('.step','.stp','.brep','.obj','.stl','.ply','.gltf','.glb','.bin'):reason='historical generated geometry'
        elif historical(rel) and p.name in ('neutral-mesh.json','source-coverage-map.json','mesh.json'):reason='historical generated mesh'
        if reason:removed.append((rel.as_posix(),p.stat().st_size,reason));p.unlink()
    for rel,digest in protected.items():
        if not (tree/rel).is_file() or sha(tree/rel)!=digest:raise RuntimeError('Protected current source changed: '+rel)
    for name,body in (('package_current.py',PACKAGER),('smoke_cad.py',SMOKE)):
        compile(body,name,'exec');write(tree/'tools'/name,body)
    # A tiny packaging regression, not a project-wide qualification stage.
    ns={};exec(compile(PACKAGER,'package_current.py','exec'),ns)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)/'source';(root/'RC_K0').mkdir(parents=True);write(root/'STATUS.json','{}');write(root/'sample.txt','one')
        ns['package'](root,Path(t)/'RCF.zip');first=(root/'FILES.sha256').read_text()
        write(root/'sample.txt','two');ns['package'](root,Path(t)/'RCF.zip')
        assert first!=(root/'FILES.sha256').read_text()
    write(tree/'START_HERE.md',README);write(tree/CP3/'docs/CURRENT_DELIVERY_RULE.md',POLICY)
    notice='> Current rule: ONE integrated RCF.zip. Old patch/prevalidation/ancestor-replay publication gates are superseded by fortification/r0c_cp3/docs/CURRENT_DELIVERY_RULE.md. Continue implementation from this tree.\n\n'
    amended=[]
    for p in tree.rglob('*.md'):
        rel=p.relative_to(tree)
        if ('roadmap' in p.name.lower() or p in (tree/CP3/'README.md',tree/CP3/'docs/CP3_SCOPE_FREEZE.md')) and not historical(rel) and 'upstream_reference' not in rel.parts:
            write(p,notice+p.read_text(encoding='utf-8'));amended.append(rel.as_posix())
    contract=json.loads((tree/CP3/'contracts/BATTLEMENT_RHYTHM_CONTRACT.json').read_text())
    with (tree/CP3/'data/CP3_FIXTURE_MATRIX.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==4 and contract['schema']=='royal-capital.fortification.battlement-rhythm/1'
    matrix=(tree/CP3/'data/CP3_FIXTURE_MATRIX.csv').read_text()
    assert all('R0C_CP3_FX0'+str(i) in matrix for i in range(1,5))
    inv=json.loads(Path('single_result/inventory.json').read_text());intake={k:inv[k] for k in ('selected_source','source_zip_bytes','source_zip_sha256')}
    with tempfile.TemporaryDirectory(prefix='rcf-smoke-') as t:
        env=Path(t)/'venv'
        try:
            subprocess.run([sys.executable,'-m','venv',str(env)],check=True,capture_output=True,text=True,timeout=60)
            py=env/'bin/python'
            subprocess.run([str(py),'-m','pip','install','--no-index','--no-cache-dir','--find-links',str(tree/RUNTIME/'wheelhouse'),'--require-hashes','-r',str(tree/RUNTIME/'requirements-lock.txt')],check=True,capture_output=True,text=True,timeout=240)
            check=subprocess.run([str(py),'-m','pip','check'],check=True,capture_output=True,text=True,timeout=30)
            run=subprocess.run([str(py),str(tree/'tools/smoke_cad.py')],check=True,capture_output=True,text=True,timeout=60)
            smoke=json.loads(run.stdout.strip().splitlines()[-1]);smoke.update(offline_install='PASS',pip_check=check.stdout.strip(),python=sys.version.split()[0])
        except (subprocess.SubprocessError,ValueError) as e:
            smoke={'status':'FAILED','error':str(e),'stderr':str(getattr(e,'stderr',''))[-3500:],'source_retained':True}
    receipt={'schema':'royal-capital.fortification.cp3a-source-receipt/1','status':'SOURCE_INTEGRATED','scope_contract_matrix':'PASS','fixture_count':4,'geometry_qualification':'NOT_STARTED','delivery':'SINGLE_INTEGRATED_FILE','historical_publication_gate':'SUPERSEDED_BY_USER_REQUEST','actual_intake':intake,'accepted_C2_zip_byte_identity':'NOT_CLAIMED'}
    dump(tree/CP3/'reports/CP3A_SOURCE_RECEIPT.json',receipt)
    wheels=list((tree/RUNTIME/'wheelhouse').glob('*.whl'))
    status={'project':'ROYAL-CAPITAL-FORTIFICATION-CAD','work':'R0C-CP3 single integrated source consolidation','source_publication':'COMPLETE','cp3_scope_contract_fixture_files':'INTEGRATED','cp3_functional_qualification':'NOT_STARTED','r0c_cp3':'OPEN','godot_product':'NOT_STARTED','next_task':'CP3-B — R0C_CP3_FX01 LINEAR BATTLEMENT PILOT','actual_source_intake':intake,'accepted_C2_byte_identity':'NOT_CLAIMED_OR_REQUIRED_FOR_CONTINUATION','protected_current_source_files':len(protected),'protected_current_source_unchanged':True,'runtime_smoke':smoke,'packager_manifest_refresh_test':'PASS','delivery_policy':{'integrated_files':1,'patch_required':False,'baseline_replay_required':False,'prevalidation_zip_required':False,'library_publication_is_start_gate':False},'size':{'input_uncompressed_bytes':before_bytes,'removed_files':len(removed),'removed_uncompressed_bytes':sum(x[1] for x in removed),'runtime_wheel_count':len(wheels),'runtime_wheel_bytes':sum(p.stat().st_size for p in wheels),'runtime_wheels_removed':0},'amended_active_docs':amended,'removed_by_reason':{r:sum(x[1] for x in removed if x[2]==r) for r in sorted({x[2] for x in removed})},'validation_scope':'contract/matrix parse; current source preservation; one packaging edit-refresh regression; one offline CAD STEP/BREP smoke; no CP2 requalification'}
    dump(tree/'STATUS.json',status)
    manifest=tree/'FILES.sha256';files=sorted(p for p in tree.rglob('*') if p.is_file() and p!=manifest)
    write(manifest,''.join(sha(p)+'  '+p.relative_to(tree).as_posix()+'\n' for p in files))
    files=[p for p in tree.rglob('*') if p.is_file()]
    summary=dict(status);summary['size']=dict(status['size'],output_files=len(files),output_uncompressed_bytes=sum(p.stat().st_size for p in files))
    dump(Path('single_result/result.json'),summary)
    if a.output.exists():shutil.rmtree(a.output)
    shutil.copytree(tree,a.output)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
