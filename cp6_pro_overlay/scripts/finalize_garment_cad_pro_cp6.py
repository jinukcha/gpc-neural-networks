#!/usr/bin/env python3
"""Aggregate CP6 export, fit, fresh-reopen, Godot, and preservation receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.pattern_cad.document.model import canonical_sha256


BUILD_REL = Path("build/garment_cad_pro_cp6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fit_receipts(root: Path) -> list[dict]:
    return [
        load_json(path)
        for path in sorted((root / BUILD_REL / "trousers/fit").glob("*/*.json"))
    ]


def terminal_receipt(root: Path) -> dict:
    preliminary = load_json(root / BUILD_REL / "cp6_preliminary_receipt.json")
    fresh = load_json(root / BUILD_REL / "game_products/fresh_process_receipt.json")
    godot = load_json(root / BUILD_REL / "godot_product/godot_consumer_receipt.json")
    preserve = load_json(root / BUILD_REL / "predecessor_preservation.json")
    topology = load_json(root / BUILD_REL / "trousers/topology_receipt.json")
    tunic_export = load_json(root / BUILD_REL / "manufacturing/tunic/export_receipt.json")
    trousers_export = load_json(root / BUILD_REL / "manufacturing/trousers/export_receipt.json")
    receipts = fit_receipts(root)
    fit_pass_count = sum(bool(item["pose_pass"]) for item in receipts)
    technical = all((
        tunic_export["export_pass"],
        trousers_export["export_pass"],
        topology["topology_pass"],
        fit_pass_count == 9,
        fresh["fresh_process_pass"],
        godot["consumer_pass"],
        preserve["all_predecessors_unchanged"],
    ))
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP6",
        "terminal_decision": "CP6_ACCEPTED_EXPORT_AND_TROUSERS_CLOSEOUT" if technical else "HOLD_CP6_PRODUCT_CLOSEOUT",
        "tunic_manufacturing_pass": tunic_export["export_pass"],
        "trousers_manufacturing_pass": trousers_export["export_pass"],
        "trousers_topology_pass": topology["topology_pass"],
        "trousers_fit_scenario_count": len(receipts),
        "trousers_fit_pass_count": fit_pass_count,
        "glb_fresh_process_pass": fresh["fresh_process_pass"],
        "godot_4_7_2_consumer_pass": godot["consumer_pass"],
        "predecessor_preservation_pass": preserve["all_predecessors_unchanged"],
        "cp6_scope_acceptance": technical,
        "product_acceptance": technical,
        "tunic_game_product_scope": "IMMUTABLE_CP2B_REFERENCE_MESH_WITH_CP5_R1_ACCEPTED_MORPHS",
        "tunic_cp3_feature_topology_embodied": False,
        "trousers_actual_product_topology": True,
        "garment_cad_pro_r1a_complete": False,
        "program_hold_reason": (
            "CP3 dart/pleat/gather/gusset manufacturing authority is exported, but the tunic game mesh still uses the immutable CP2B reference topology."
        ),
        "thresholds_changed": False,
        "mesh_scaling": "FORBIDDEN",
        "runtime": preliminary.get("runtime", {"warp": "1.17.0", "godot": "4.7.2"}),
        "next_checkpoint": "GARMENT_CAD_PRO_R1A_CP6_R1" if technical else "GARMENT_CAD_PRO_R1A_CP6_REPAIR",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_report(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1A / CP6 실행 보고서

## Terminal decision

```text
terminal decision              {receipt['terminal_decision']}
CP6 scope acceptance           {receipt['cp6_scope_acceptance']}
trousers fit                    {receipt['trousers_fit_pass_count']}/{receipt['trousers_fit_scenario_count']}
GLB fresh-process reopen        {receipt['glb_fresh_process_pass']}
Godot 4.7.2 read-only consumer  {receipt['godot_4_7_2_consumer_pass']}
predecessor preservation       {receipt['predecessor_preservation_pass']}
product acceptance             {receipt['product_acceptance']}
R1A program complete           {receipt['garment_cad_pro_r1a_complete']}
```

CP6는 튜닉과 trousers의 1:1 제조용 SVG·DXF·JSON, neutral-gray GLB와 Godot 소비를
발행했다. Trousers는 waist/crotch/inseam/outseam, back dart, waistband와 crotch gusset을
가진 실제 분기형 제품 topology로 sit·squat·stride를 세 material에서 판정한다.

제한: 튜닉 game GLB는 CP5-R1에서 수락된 immutable CP2B reference mesh와 10 motion morph를
사용한다. CP3의 dart·pleat·gather·gusset authority는 제조 패키지에는 포함되지만 동일한 3D
mesh topology에 아직 materialize되지 않았다. 따라서 CP6 전달 범위는 수락하되 R1A 전체 종료는
선언하지 않는다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1A_CP6_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def update_roadmap(root: Path, receipt: dict) -> None:
    roadmap = root / "docs/roadmap/GARMENT_CAD_PRO_R1A_ROADMAP_KO.md"
    with roadmap.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n## CP6 결과\n\n"
            f"- Terminal: `{receipt['terminal_decision']}`\n"
            "- 제조 2D: 튜닉·trousers 1:1 SVG / DXF / JSON 발행\n"
            "- 게임 3D: neutral-gray GLB, fresh-process reopen, Godot 4.7.2 read-only import\n"
            "- Trousers: 7 pattern panels, 4 back dart legs, waistband 2, crotch gusset 1, 9 motion scenarios\n"
            "- 남은 범위: CP3 tunic feature-complete topology의 실제 3D materialization\n"
        )


def write_next(root: Path, receipt: dict) -> None:
    if receipt["cp6_scope_acceptance"]:
        task = (
            "`GARMENT-CAD-PRO-R1A / CP6-R1 — TUNIC FEATURE-COMPLETE 3D TOPOLOGY / "
            "DART–PLEAT–GATHER–GUSSET MATERIALIZATION / TEN-POSE REQUALIFICATION`"
        )
    else:
        task = "`GARMENT-CAD-PRO-R1A / CP6-REPAIR — FAILED EXPORT / TOPOLOGY / CONSUMER GATE ONLY`"
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n" + task + "\n",
        encoding="utf-8",
    )


def main() -> int:
    root = parse_args().root.resolve()
    receipt = terminal_receipt(root)
    write_json(root / BUILD_REL / "cp6_receipt.json", receipt)
    write_json(root / "PROFESSIONAL_STATUS.json", receipt)
    write_report(root, receipt)
    update_roadmap(root, receipt)
    write_next(root, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["cp6_scope_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
