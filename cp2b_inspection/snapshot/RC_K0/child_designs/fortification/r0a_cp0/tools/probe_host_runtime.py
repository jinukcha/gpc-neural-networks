#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib, importlib.metadata as md, json, platform, subprocess, sys, sysconfig, traceback
from pathlib import Path

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
    return h.hexdigest()

def dist_version(name: str):
    try: return md.version(name)
    except md.PackageNotFoundError: return None

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--exact-source-root', required=True)
    ap.add_argument('--report', required=True)
    ns=ap.parse_args()
    before=subprocess.run([sys.executable,'-m','pip','list','--format=freeze'],capture_output=True,text=True,check=False)
    packages={n:dist_version(n) for n in [
        'build123d','cadquery-ocp','cadquery-ocp-novtk','cadquery','numpy','typing_extensions','svgpathtools','anytree','ezdxf','fonttools','ipython','ocpsvg','ocp_gordon','trianglesolver','sympy','scipy','scikit-learn','webcolors','requests','lib3mf','bd_materials','threejs-materials']}
    module_checks={}
    for module in ['OCP','OCP.collections','build123d']:
        try:
            importlib.import_module(module)
            module_checks[module]={'imported':True,'error':None}
        except Exception as exc:
            module_checks[module]={'imported':False,'error':f'{type(exc).__name__}: {exc}'}
    exact_import={}
    code='import build123d; print(build123d.__version__)'
    env=dict(__import__('os').environ)
    env['PYTHONPATH']=str(Path(ns.exact_source_root)/'src')
    proc=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env,check=False)
    exact_import={'returncode':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip()}
    after=subprocess.run([sys.executable,'-m','pip','list','--format=freeze'],capture_output=True,text=True,check=False)
    before_hash=hashlib.sha256(before.stdout.encode()).hexdigest()
    after_hash=hashlib.sha256(after.stdout.encode()).hexdigest()
    ocp_ok=packages.get('cadquery-ocp-novtk')=='8.0.1.0.0' and module_checks['OCP.collections']['imported']
    result={
      'schema':'royal-capital.fortification.host-preflight/1',
      'python':{'version':platform.python_version(),'cache_tag':sys.implementation.cache_tag,'soabi':sysconfig.get_config_var('SOABI'),'platform':sysconfig.get_platform(),'executable':sys.executable,'executable_sha256':digest(Path(sys.executable))},
      'packages':packages,
      'module_checks':module_checks,
      'exact_source_import':exact_import,
      'global_environment_unchanged':before_hash==after_hash,
      'pip_list_before_sha256':before_hash,
      'pip_list_after_sha256':after_hash,
      'exact_runtime_compatible':ocp_ok,
      'status':'BOOTSTRAP_REJECTED_VERSION_MISMATCH' if not ocp_ok else 'HOST_EXACT_RUNTIME_PRESENT',
    }
    Path(ns.report).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0
if __name__=='__main__': raise SystemExit(main())
