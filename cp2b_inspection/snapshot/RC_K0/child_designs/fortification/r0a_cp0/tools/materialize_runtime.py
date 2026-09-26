#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, venv
from pathlib import Path

def sha(path: Path):
    h=hashlib.sha256();
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--python',required=True); ap.add_argument('--wheelhouse',required=True); ap.add_argument('--lock',required=True); ap.add_argument('--destination',required=True); ap.add_argument('--receipt',required=True)
    ns=ap.parse_args(); wheelhouse=Path(ns.wheelhouse); lock=Path(ns.lock); dest=Path(ns.destination); receipt=Path(ns.receipt)
    wheels=sorted(wheelhouse.glob('*.whl'))
    lock_text=lock.read_text(encoding='utf-8') if lock.is_file() else ''
    complete=bool(wheels) and '--hash=sha256:' in lock_text and 'INCOMPLETE' not in lock_text
    if not complete:
        result={'schema':'royal-capital.fortification.runtime-admission/1','status':'BLOCKED_MISSING_EXACT_WHEELHOUSE','wheel_count':len(wheels),'lock_present':lock.is_file(),'lock_complete':False,'global_install':False}
        receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2)); return 2
    if dest.exists(): raise SystemExit('destination must be fresh')
    subprocess.run([ns.python,'-m','venv',str(dest)],check=True)
    py=dest/'bin/python'
    install=subprocess.run([str(py),'-m','pip','install','--no-index','--find-links',str(wheelhouse),'--require-hashes','-r',str(lock)],capture_output=True,text=True)
    pipcheck=subprocess.run([str(py),'-m','pip','check'],capture_output=True,text=True) if install.returncode==0 else None
    result={'schema':'royal-capital.fortification.runtime-admission/1','status':'ADMITTED' if install.returncode==0 and pipcheck and pipcheck.returncode==0 else 'FAILED','global_install':False,'python':str(py),'wheelhouse':[{'name':p.name,'size':p.stat().st_size,'sha256':sha(p)} for p in wheels],'install':{'returncode':install.returncode,'stdout':install.stdout,'stderr':install.stderr},'pip_check':None if pipcheck is None else {'returncode':pipcheck.returncode,'stdout':pipcheck.stdout,'stderr':pipcheck.stderr}}
    receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2)); return 0 if result['status']=='ADMITTED' else 1
if __name__=='__main__': raise SystemExit(main())
