#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--cp0-root", required=True)
parser.add_argument("--adapter-root", required=True)
parser.add_argument("--path-span-root", required=True)
parser.add_argument("--cp3-root", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

sys.path.insert(0, str(Path(args.adapter_root).resolve() / "src"))
sys.path.insert(0, str(Path(args.path_span_root).resolve() / "src"))
sys.path.insert(0, str(Path(args.cp3_root).resolve() / "src"))

from rcf_fortification_terrain_span import TerrainWallSpanProducer

fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
result = TerrainWallSpanProducer(args.cp0_root).execute(fixture, args.output)
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "SUCCEEDED" else 1)
