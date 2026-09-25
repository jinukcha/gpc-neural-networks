#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--cp0-root", required=True)
p.add_argument("--cp1-root", required=True)
p.add_argument("--cp2-root", required=True)
p.add_argument("--fixture", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
sys.path.insert(0, str(Path(a.cp1_root).resolve() / "src"))
sys.path.insert(0, str(Path(a.cp2_root).resolve() / "src"))
from rcf_fortification_wall_span import StraightWallSpanProducer
fixture = json.loads(Path(a.fixture).read_text(encoding="utf-8"))
result = StraightWallSpanProducer(a.cp0_root).execute(fixture, a.output)
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "SUCCEEDED" else 1)
