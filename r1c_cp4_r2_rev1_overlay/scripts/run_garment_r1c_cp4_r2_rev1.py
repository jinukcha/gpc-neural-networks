#!/usr/bin/env python3
"""Run Blender-free CP4-R2-R1-REV1 metric repair and direct GLB build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.metric_fidelity import run_metric_fidelity_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_metric_fidelity_pipeline(args.root.resolve())
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
