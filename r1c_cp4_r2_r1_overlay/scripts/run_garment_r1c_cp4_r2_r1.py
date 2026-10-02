#!/usr/bin/env python3
"""Run CP4-R2-R1 metric decomposition and local arrangement repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.materialization.metric_fidelity.pipeline import run_metric_fidelity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    receipt = run_metric_fidelity(root)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
