#!/usr/bin/env python3
"""Finalize CP3 after fresh GLB reopen and exact Godot consumer execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wuxia_garment_oss.rig.game_product.evidence import render_evidence


BUILD_REL = Path("build/rig_cp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_runtime_contracts(root: Path, godot: dict, glb: dict) -> tuple[dict, dict]:
    package = {
        "contract": "GodotGarmentEquipPackage/1",
        "engine": "Godot 4.7.2 Linux",
        "runtime_script": "godot/garment_r1b_cp3/garment_runtime.gd",
        "products": {
            "UPPER": "build/rig_cp3/products/feature_complete_tunic_rigged.glb",
            "LOWER": "build/rig_cp3/products/trousers_rigged.glb",
        },
        "adapter_ids": ["CANONICAL_EXACT_V1", "MIXAMO_STYLE_V1"],
        "atomic_policy": {
            "instantiate_off_state": True,
            "validate_before_attach": True,
            "preserve_previous_target_on_failed_swap": True,
            "reject_occupied_slot": True,
        },
        "glb_fresh_reopen_pass": glb["fresh_process_pass"],
        "godot_consumer_pass": godot["consumer_pass"],
        "secondary_motion": "NOT_EXECUTED",
        "lod": "NOT_EXECUTED",
        "outfit_composition": "NOT_EXECUTED",
    }
    package["package_sha256"] = canonical_sha256(package)
    swap = {
        "contract": "CharacterSwapContract/1",
        "compatible_adapter": "MIXAMO_STYLE_V1",
        "incompatible_fixture": "MIXAMO_MISSING_R_CALF",
        "validate_before_commit": True,
        "failed_swap_preserves_current_target": godot["failed_swap_preserved"],
        "compatible_swap_pass": godot["character_swap_pass"],
    }
    swap["contract_sha256"] = canonical_sha256(swap)
    write_json(root / BUILD_REL / "godot_garment_equip_package.json", package)
    write_json(root / BUILD_REL / "character_swap_contract.json", swap)
    return package, swap


def write_report(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1B / CP3 실행 보고서

## 판정

```text
terminal decision          {receipt['terminal_decision']}
CP3 scope acceptance       {str(receipt['cp3_scope_acceptance']).lower()}
rigged GLB fresh reopen    {str(receipt['glb_fresh_reopen_pass']).lower()}
Godot 4.7.2 consumer       {str(receipt['godot_consumer_pass']).lower()}
atomic equip               {str(receipt['atomic_equip_pass']).lower()}
character swap             {str(receipt['character_swap_pass']).lower()}
failed swap preservation   {str(receipt['failed_swap_preserved']).lower()}
```

로컬 실행면의 지속적인 ClientError와 공개 CP2 체크포인트 부재 때문에 CP2 ZIP의 바이트 연속성은
확인하지 못했다. R1A accepted game products와 R1B 설계 계약에서 CP3에 필요한 skeleton, semantic
weight field와 corrective metadata만 제한적으로 복원했다. 이 사실은 restored_input_provenance.json에
기록되며, 기존 R1A 제품은 변경하지 않았다.

CP3는 두 GLB에 23-bone skin, JOINTS_0, WEIGHTS_0와 inverse-bind matrices를 발행했다. Godot에서는
import 이후 topology나 weight를 수정하지 않고 내부 garment Skeleton3D를 target character Skeleton3D로
구동한다. Equip, duplicate-slot rejection, unequip/re-equip, compatible character swap과 incompatible swap의
원자적 보존을 실행했다. Secondary motion, LOD와 outfit composition은 범위 밖이다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1B_CP3_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`GARMENT-CAD-PRO-R1B / CP4 — MULTI-GARMENT REGISTRY / OUTFIT LAYERING / BODY OCCLUSION`\n\n"
        "Register tunic and trousers as family products, compile semantic coverage and layer thickness, "
        "generate body occlusion masks, admit a compatible two-garment outfit, and reject incompatible "
        "slot/coverage combinations atomically. Secondary motion and rig-aware LOD remain deferred.\n",
        encoding="utf-8",
    )


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    preliminary = load_json(build / "cp3_preliminary_receipt.json")
    glb = load_json(build / "glb_fresh_reopen_receipt.json")
    godot = load_json(build / "godot_product/godot_equip_receipt.json")
    provenance = load_json(build / "restored_input_provenance.json")
    skeleton = load_json(build / "canonical_skeleton.json")
    package, swap = write_runtime_contracts(root, godot, glb)
    accepted = bool(
        glb["fresh_process_pass"]
        and godot["consumer_pass"]
        and godot["atomic_equip_pass"]
        and godot["character_swap_pass"]
        and godot["failed_swap_preserved"]
        and preliminary["glb_products_written"]
    )
    receipt = {
        "checkpoint": "GARMENT_CAD_PRO_R1B_CP3",
        "terminal_decision": "CP3_COMPLETE_GODOT_EQUIP_PRODUCT" if accepted else "HOLD_CP3_GODOT_EQUIP",
        "cp3_scope_acceptance": accepted,
        "input_continuity_mode": "RESTORED_CONTRACT_EQUIVALENCE",
        "exact_cp2_byte_continuity_verified": provenance["exact_cp2_byte_continuity_verified"],
        "tunic": preliminary["tunic"],
        "trousers": preliminary["trousers"],
        "glb_fresh_reopen_pass": glb["fresh_process_pass"],
        "godot_consumer_pass": godot["consumer_pass"],
        "atomic_equip_pass": godot["atomic_equip_pass"],
        "character_swap_pass": godot["character_swap_pass"],
        "failed_swap_preserved": godot["failed_swap_preserved"],
        "unequip_re_equip_pass": godot["unequip_re_equip_pass"],
        "final_equipped_state_count": godot["final_state_count"],
        "godot_version": godot["godot_version"],
        "equip_package_sha256": package["package_sha256"],
        "character_swap_contract_sha256": swap["contract_sha256"],
        "r1a_products_mutated": False,
        "secondary_motion_executed": False,
        "lod_executed": False,
        "outfit_composition_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "next_checkpoint": "GARMENT_CAD_PRO_R1B_CP4",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(build / "cp3_receipt.json", receipt)
    write_json(root / "R1B_STATUS.json", receipt)
    evidence_path = build / "cp3_godot_equip_evidence.png"
    render_evidence(
        evidence_path,
        build / "products/feature_complete_tunic_rigged.glb",
        build / "products/trousers_rigged.glb",
        skeleton,
        godot,
        receipt,
    )
    write_report(root, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if accepted else 4


if __name__ == "__main__":
    raise SystemExit(main())
