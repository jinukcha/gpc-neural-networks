#!/usr/bin/env python3
"""Finalize CP4-R2-R1-REV1 from validator, Godot, and direct art-review receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validator-report", type=Path, required=True)
    parser.add_argument("--art-review", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def art_disposition(review: dict) -> tuple[str, int]:
    views = review.get("views", [])
    if not views or any(item.get("disposition") == "NOT_REVIEWABLE" for item in views):
        return "PENDING", len(review.get("blocking_defects", []))
    blocking = len(review.get("blocking_defects", []))
    if blocking or any(item.get("disposition") == "BLOCKING_DEFECT" for item in views):
        return "FAIL", max(blocking, 1)
    return "PASS", 0


def terminal_receipt(root: Path, validator: dict, art: dict) -> dict:
    build = root / "build/r1c_cp4_r2_rev1"
    preliminary = load_json(build / "cp4_r2_rev1_preliminary_receipt.json")
    technical = load_json(build / "technical_qualification_receipt.json")
    godot = load_json(build / "godot_consumer_receipt.json")
    evidence = load_json(build / "godot_visual_evidence_receipt.json")
    disposition, blocking = art_disposition(art)
    validator_errors = int(validator.get("issues", {}).get("numErrors", 999999))
    validator_pass = validator_errors == 0
    godot_pass = bool(godot.get("consumer_pass")) and bool(evidence.get("all_views_reviewable"))
    accepted = bool(technical["technical_pass"]) and validator_pass and godot_pass and disposition == "PASS"
    if accepted:
        terminal = "CP4_R2_R1_REV1_ACCEPTED_BLENDER_FREE_PRODUCT"
    elif disposition == "PENDING":
        terminal = "CP4_R2_R1_REV1_HOLD_DIRECT_ART_AUDIT"
    else:
        terminal = "CP4_R2_R1_REV1_HOLD_DEFECT_REPAIR"
    payload = dict(preliminary)
    payload.update({
        "phase": "CLOSED" if disposition != "PENDING" else "PENDING_DIRECT_ART_AUDIT",
        "terminal_decision": terminal,
        "technical_pass": bool(technical["technical_pass"]),
        "khronos_validator_pass": validator_pass,
        "khronos_validator_error_count": validator_errors,
        "godot_consumer_pass": godot_pass,
        "direct_visual_art_review": disposition,
        "blocking_visual_defect_count": blocking,
        "product_acceptance": accepted,
        "next_checkpoint": "GARMENT_CAD_PRO_R1C_CP5" if accepted else "GARMENT_CAD_PRO_R1C_CP4_R2_R1_REV1",
        "blender_executed": False,
        "blend_generated": False,
        "cp4_r1_predecessor_mutated": False,
    })
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_report(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1C / CP4-R2-R1-REV1

```text
terminal decision          {receipt['terminal_decision']}
technical pass             {receipt['technical_pass']}
Khronos validator          {receipt['khronos_validator_pass']}
Godot consumer             {receipt['godot_consumer_pass']}
direct visual art review   {receipt['direct_visual_art_review']}
product acceptance         {receipt['product_acceptance']}
Blender executed           false
```

The CP4-R1 product remains immutable. The REV1 product uses component-boundary metric decomposition,
bounded arrangement repair, Warp resettling, deterministic direct GLB, Khronos validation, and exact
Godot 4.7.2 same-camera before/after evidence.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1C_CP4_R2_R1_REV1_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_next_task(root: Path, accepted: bool) -> None:
    if accepted:
        text = "# Next task\n\n**`GARMENT-CAD-PRO-R1C / CP5 — FULL-LENGTH STRAIGHT ROBE PATTERN MATERIALIZATION`**\n"
    else:
        text = "# Next task\n\nResume **`CP4-R2-R1-REV1`** from the single blocking art-audit or metric item recorded in the receipt.\n"
    (root / "NEXT_TASK.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    validator = load_json(args.validator_report)
    art = load_json(args.art_review)
    build = root / "build/r1c_cp4_r2_rev1"
    write_json(build / "khronos_gltf_validation_report.json", validator)
    write_json(build / "direct_visual_art_review_receipt.json", art)
    receipt = terminal_receipt(root, validator, art)
    write_json(build / "cp4_r2_rev1_receipt.json", receipt)
    write_json(root / "R1C_STATUS.json", receipt)
    write_report(root, receipt)
    write_next_task(root, bool(receipt["product_acceptance"]))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
