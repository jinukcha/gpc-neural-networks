#!/usr/bin/env python3
"""Close CP4-R1 after exact Blender visual evidence has been published."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = parse_args().root.resolve()
    build = root / "build/r1c_cp4_r1"
    preliminary = load(build / "cp4_r1_preliminary_receipt.json")
    technical = load(build / "technical_qualification_receipt.json")
    visual = load(build / "visual_evidence_receipt.json")
    blender = load(build / "blender_visual_product_receipt.json")
    version_pass = blender["blender_version"] == "5.2.2"
    glb = build / blender["glb_path"]
    blend = build / blender["blend_path"]
    product_acceptance = bool(technical["technical_pass"] and visual["visual_review"] == "PASS" and version_pass and glb.is_file() and blend.is_file())
    product_receipt = load(build / "product/materialized_product_receipt.json")
    product_receipt.update({
        "glb_pending_blender_export": False,
        "glb_path": glb.relative_to(build).as_posix(),
        "glb_sha256": sha256(glb),
        "blend_path": blend.relative_to(build).as_posix(),
        "blend_sha256": sha256(blend),
    })
    write(build / "product/materialized_product_receipt.json", product_receipt)
    receipt = dict(preliminary)
    receipt.update({
        "phase": "CLOSED",
        "terminal_decision": "CP4_R1_ACCEPTED_PATTERN_DRIVEN_TUNIC" if product_acceptance else "CP4_R1_NOT_ACCEPTED",
        "technical_pass": technical["technical_pass"],
        "visual_review": visual["visual_review"],
        "product_acceptance": product_acceptance,
        "blender_visual_review_executed": True,
        "blender_version": blender["blender_version"],
        "blender_exact_version_pass": version_pass,
        "required_view_count": visual["view_count"],
        "contact_sheet_path": "build/r1c_cp4_r1/cp4_r1_body_visible_contact_sheet.png",
        "glb_sha256": product_receipt["glb_sha256"],
        "next_checkpoint": "GARMENT_CAD_PRO_R1C_CP5" if product_acceptance else "GARMENT_CAD_PRO_R1C_CP4_R1_REPAIR",
    })
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(build / "cp4_r1_receipt.json", receipt)
    write(root / "R1C_STATUS.json", receipt)
    _write_docs(root, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _write_docs(root: Path, receipt: dict) -> None:
    if receipt["product_acceptance"]:
        next_task = "**`GARMENT-CAD-PRO-R1C / CP5 — FULL-LENGTH STRAIGHT-ROBE PANEL / SIDE-GORE MATERIALIZATION`**"
    else:
        next_task = "**`GARMENT-CAD-PRO-R1C / CP4-R1 — BOUNDED VISUAL OR TECHNICAL REPAIR`**"
    (root / "NEXT_TASK.md").write_text(f"# Next task\n\n{next_task}\n", encoding="utf-8")
    report = f"""# GARMENT-CAD-PRO-R1C / CP4-R1 실행 보고서

```text
terminal decision      {receipt['terminal_decision']}
technical pass         {receipt['technical_pass']}
visual review          {receipt['visual_review']}
product acceptance     {receipt['product_acceptance']}
Blender                {receipt['blender_version']}
required views         {receipt['required_view_count']}
```

CP3 repaired pattern snapshot을 immutable 입력으로 사용했다. Orientation-aware seam map,
set-in sleeve cap patch arrangement, arrangement-space rest metric, exact wool-twill Warp settling,
body-visible neutral-gray Blender evidence를 발행했다. Post-settle vertex repair는 수행하지 않았다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1C_CP4_R1_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
