#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--fortification-root", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.fortification_root).resolve()
sys.path.insert(0, str(root / "r0b_cp4/src"))
from rcf_fortification_mixed_span import MixedSpanAssemblyProducer
fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
result = MixedSpanAssemblyProducer(root).execute(fixture, args.output)
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "SUCCEEDED" else 1)
