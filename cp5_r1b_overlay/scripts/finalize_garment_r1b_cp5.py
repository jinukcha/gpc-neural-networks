#!/usr/bin/env python3
"""Close CP5 after exact Godot 4.7.2 import and product inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.rig.library.model import canonical_sha256
from wuxia_garment_oss.rig.sleeve_generalization.evidence import render_cp5_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--godot-receipt", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _product_receipts(root: Path) -> tuple[dict, dict]:
    base = root / "build/rig_cp5"
    tunic = load_json(base / "sleeved_tunic/rigged_garment_product.json")
    robe = load_json(base / "straight_sleeve_robe/rigged_garment_product.json")
    return tunic, robe


def _write_report(root: Path, receipt: dict, runtime: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1B / CP5 실행 보고서

## Terminal decision

```text
terminal decision                 {receipt['terminal_decision']}
CP5 acceptance                    {receipt['cp5_acceptance']}
direct arm measurement            {receipt['direct_arm_measurement_pass']}
sleeve cap / armhole              {receipt['sleeve_cap_armhole_pass']}
upper-arm / forearm transfer      {receipt['upper_arm_forearm_transfer_pass']}
corrective generalization         {receipt['corrective_generalization_pass']}
Godot 4.7.2 consumer              {receipt['godot_consumer_pass']}
secondary motion                  NOT EXECUTED
rig-aware LOD                     NOT EXECUTED
```

CP4 registry와 CP3 rigged tunic은 immutable input으로 소비했다. CP5는 직접 팔 치수,
sleeve-cap/armhole construction, underarm seam, upper-arm/forearm weight field,
shoulder/underarm/elbow sparse corrective와 두 새 GLB 제품만 소유한다.

```text
sleeved tunic meshes       {runtime['sleeved_tunic']['mesh_instance_count']}
sleeved tunic vertices     {runtime['sleeved_tunic']['vertex_count']}
straight robe meshes       {runtime['straight_robe']['mesh_instance_count']}
straight robe vertices     {runtime['straight_robe']['vertex_count']}
```
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1B_CP5_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    next_task = (
        "# Next task\n\n"
        "**`GARMENT-CAD-PRO-R1B / CP6 — SECONDARY MOTION / RIG-AWARE LOD / MULTI-DISTANCE CLOSEOUT`**\n\n"
        "Use the accepted CP5 products and registry as immutable inputs. Add bounded cloth secondary-motion domains, "
        "corrective-preserving rig-aware LOD transfer, and exact Godot 4.7.2 multi-distance captures.\n"
    )
    (root / "NEXT_TASK.md").write_text(next_task, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    build = root / "build/rig_cp5"
    runtime = load_json(args.godot_receipt)
    preliminary = load_json(build / "cp5_preliminary_receipt.json")
    if runtime.get("consumer_pass") is not True:
        raise RuntimeError("Godot CP5 consumer did not pass")
    version = runtime["godot_version"]
    if (int(version["major"]), int(version["minor"]), int(version["patch"])) != (4, 7, 2):
        raise RuntimeError("unexpected Godot runtime")
    receipt = dict(preliminary)
    receipt["phase"] = "CLOSED"
    receipt["godot_consumer_pass"] = True
    receipt["godot_version"] = "4.7.2"
    gates = (
        "direct_arm_measurement_pass",
        "sleeve_cap_armhole_pass",
        "upper_arm_forearm_transfer_pass",
        "corrective_generalization_pass",
        "incompatible_atomic_rejection_pass",
        "godot_consumer_pass",
    )
    receipt["cp5_acceptance"] = all(bool(receipt[name]) for name in gates)
    receipt["terminal_decision"] = "CP5_COMPLETE_SLEEVE_RIG_GENERALIZATION" if receipt["cp5_acceptance"] else "HOLD_CP5"
    receipt["next_checkpoint"] = "GARMENT_CAD_PRO_R1B_CP6"
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(build / "godot_product/godot_sleeve_runtime_receipt.json", runtime)
    write_json(build / "cp5_receipt.json", receipt)
    write_json(root / "R1B_STATUS.json", receipt)
    measurement = load_json(build / "arm_measurements.json")
    products = _product_receipts(root)
    render_cp5_evidence(root, measurement, products)
    _write_report(root, receipt, runtime)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
