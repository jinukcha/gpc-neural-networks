#!/usr/bin/env python3
"""Run GARMENT-CAD-PRO-R1C CP4-R1 materialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.materialization import run_materialization


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_materialization(args.root.resolve())
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
