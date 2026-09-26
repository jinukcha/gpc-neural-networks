#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--cp0-root',required=True); p.add_argument('--cp1-root',required=True); p.add_argument('--r0b-root',required=True); p.add_argument('--fixture',required=True); p.add_argument('--output',required=True); a=p.parse_args()
sys.path[:0]=[str(Path(a.cp1_root).resolve()/'src'),str(Path(a.r0b_root).resolve()/'src')]
from rcf_fortification_path_span import PathWallSpanProducer
fixture=json.loads(Path(a.fixture).read_text())
result=PathWallSpanProducer(a.cp0_root).execute(fixture,a.output)
print(json.dumps(result,indent=2,sort_keys=True)); raise SystemExit(0 if result['status']=='SUCCEEDED' else 1)
