#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cp0-root", required=True)
parser.add_argument("--cp1-root", required=True)
parser.add_argument("--cp2-root", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

sys.path.insert(0, str(Path(args.cp1_root).resolve() / "src"))
sys.path.insert(0, str(Path(args.cp2_root).resolve() / "src"))
from rcf_fortification_join import WallJoinProducer

fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
result = WallJoinProducer(args.cp0_root).execute(fixture, args.output)
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "SUCCEEDED" else 1)
