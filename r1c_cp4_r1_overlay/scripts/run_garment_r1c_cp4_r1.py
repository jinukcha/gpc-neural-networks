#!/usr/bin/env python3
"""Run GARMENT-CAD-PRO-R1C CP4-R1 materialization."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path


def _apply_source_patch(root: Path) -> None:
    package_dir = root / "source/wuxia_garment_oss/materialization"
    path = package_dir / "contact_patch.py"
    spec = importlib.util.spec_from_file_location("r1c_cp4_r1_contact_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load CP4-R1 contact patch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_contact_owner_patch(package_dir)
    importlib.invalidate_caches()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    _apply_source_patch(root)
    from wuxia_garment_oss.materialization import run_materialization
    receipt = run_materialization(root)
    technical_path = root / "build/r1c_cp4_r1/technical_qualification_receipt.json"
    technical = json.loads(technical_path.read_text(encoding="utf-8"))
    print(json.dumps({"preliminary": receipt, "technical": technical}, sort_keys=True))
    return 0 if technical["technical_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
