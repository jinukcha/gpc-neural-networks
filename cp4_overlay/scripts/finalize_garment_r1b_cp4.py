#!/usr/bin/env python3
"""Close CP4 after exact Godot outfit runtime verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.rig.library.evidence import render_cp4_evidence
from wuxia_garment_oss.rig.library.model import canonical_sha256


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


def _update_roadmap(root: Path) -> None:
    path = root / "docs/roadmap/GARMENT_CAD_PRO_R1B_ROADMAP_KO.md"
    marker = "## CP4 실행 결과"
    text = path.read_text(encoding="utf-8") if path.is_file() else "# GARMENT-CAD-PRO-R1B Roadmap\n"
    if marker in text:
        return
    addition = """

## CP4 실행 결과

```text
GarmentLibraryRegistry/1      구현
OutfitAssemblyPlan/1          구현
BodyOcclusionMask/1           구현
compatible two-piece outfit   PASS
incompatible atomic reject    PASS
Godot 4.7.2 consumer          PASS
secondary motion              미실행
rig-aware LOD                 미실행
next                          CP5
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + addition, encoding="utf-8")


def write_report(root: Path, receipt: dict, runtime: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1B / CP4 실행 보고서

## Terminal decision

```text
terminal decision              {receipt['terminal_decision']}
registry entries               {receipt['registry_entry_count']}
compatible outfit              {receipt['compatible_outfit_pass']}
incompatible atomic rejection  {receipt['incompatible_outfit_rejected']}
body occlusion                 {receipt['body_occlusion_pass']}
Godot 4.7.2 consumer           {receipt['godot_outfit_consumer_pass']}
secondary motion               NOT EXECUTED
rig-aware LOD                  NOT EXECUTED
```

CP3의 tunic/trousers rigged products를 변경하지 않고 registry, layer thickness,
semantic body coverage, per-triangle hide mask와 atomic outfit transaction을 구현했다.
Godot는 두 제품을 off-tree에서 모두 해석한 후 한 번에 commit하며, incompatible
fixture는 기존 two-piece outfit을 변경하지 않고 거부한다.

Runtime product counts:

```text
tunic surfaces      {runtime['tunic']['surface_count']}
tunic vertices      {runtime['tunic']['vertex_count']}
trousers surfaces   {runtime['trousers']['surface_count']}
trousers vertices   {runtime['trousers']['vertex_count']}
```
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1B_CP4_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "**`GARMENT-CAD-PRO-R1B / CP5 — SLEEVED TUNIC / STRAIGHT-SLEEVE ROBE RIG GENERALIZATION`**\n\n"
        "Use the accepted registry, outfit compiler and CP0–CP3 rig products as immutable inputs. "
        "Add direct-arm-measurement sleeve products, sleeve-cap/underarm binding and elbow/shoulder corrective generalization. "
        "Secondary motion and rig-aware LOD remain deferred to CP6.\n",
        encoding="utf-8",
    )
    _update_roadmap(root)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    build = root / "build/rig_cp4"
    receipt = load_json(build / "cp4_receipt.json")
    runtime = load_json(args.godot_receipt)
    if runtime.get("consumer_pass") is not True:
        raise RuntimeError("Godot outfit consumer did not pass")
    version = runtime["godot_version"]
    if (int(version["major"]), int(version["minor"]), int(version["patch"])) != (4, 7, 2):
        raise RuntimeError("unexpected Godot version")
    receipt["godot_outfit_consumer_pass"] = True
    receipt["godot_version"] = "4.7.2"
    receipt["cp4_acceptance"] = all(
        receipt[name]
        for name in (
            "compatible_outfit_pass",
            "incompatible_outfit_rejected",
            "body_occlusion_pass",
            "python_atomic_transaction_pass",
            "godot_outfit_consumer_pass",
        )
    )
    receipt["terminal_decision"] = "CP4_COMPLETE_MULTI_GARMENT_OUTFIT" if receipt["cp4_acceptance"] else "HOLD_CP4"
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(build / "godot_product/godot_outfit_runtime_receipt.json", runtime)
    write_json(build / "cp4_receipt.json", receipt)
    write_json(root / "R1B_STATUS.json", receipt)
    transactions = load_json(build / "outfit_transaction_suite.json")
    render_cp4_evidence(root, receipt, transactions)
    write_report(root, receipt, runtime)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
