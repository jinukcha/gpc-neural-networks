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
    root = args.root.resolve()
    receipt = run_materialization(root)
    technical_path = root / "build/r1c_cp4_r1/technical_qualification_receipt.json"
    technical = json.loads(technical_path.read_text(encoding="utf-8"))
    print(json.dumps({"preliminary": receipt, "technical": technical}, sort_keys=True))
    if technical["technical_pass"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
