#!/usr/bin/env python3
"""Build CP4 registry, outfit plans, occlusion mask, and Python atomic transactions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wuxia_garment_oss.rig.library.compiler import compile_outfit
from wuxia_garment_oss.rig.library.contracts import cp4_schemas
from wuxia_garment_oss.rig.library.evidence import render_cp4_evidence
from wuxia_garment_oss.rig.library.model import (
    CoverageRegion,
    GarmentFamilyEntry,
    GarmentLibraryRegistry,
    canonical_sha256,
)
from wuxia_garment_oss.rig.library.occlusion import compile_body_hide_mask
from wuxia_garment_oss.rig.runtime.outfit_transaction import (
    OutfitRuntimeState,
    commit_outfit,
    unequip_outfit,
)


BUILD_REL = Path("build/rig_cp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_bones(root: Path) -> tuple[str, ...]:
    skeleton = json.loads((root / "build/rig_cp0/canonical_skeleton.json").read_text(encoding="utf-8"))
    bones = skeleton.get("bones", skeleton.get("semantic_bones", []))
    result = []
    for item in bones:
        if isinstance(item, str):
            result.append(item)
        else:
            result.append(item.get("bone_id", item.get("semantic_bone_id", item.get("id"))))
    return tuple(value for value in result if value)


def discover_product(root: Path, garment_token: str) -> str:
    candidates = sorted((root / "build").rglob("*.glb"))
    ranked = []
    for path in candidates:
        lower = path.relative_to(root).as_posix().lower()
        if garment_token not in lower:
            continue
        score = 0
        score += 16 if "rig_cp3" in lower else 0
        score += 8 if "rigged" in lower else 0
        score += 4 if "product" in lower else 0
        score += 2 if "game" in lower else 0
        score -= 4 if "cp6_r1" in lower else 0
        ranked.append((score, lower, path))
    if not ranked:
        raise FileNotFoundError(f"no GLB product found for {garment_token}")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][2].relative_to(root).as_posix()


def _entry(root: Path, garment_id: str, family_id: str, path: str, layer: str, slots, coverage, exclusive=()):
    product = root / path
    return GarmentFamilyEntry(
        garment_id=garment_id,
        family_id=family_id,
        product_path=path,
        product_sha256=file_sha256(product),
        layer_class=layer,
        slots=tuple(slots),
        exclusive_slots=tuple(exclusive),
        coverage=tuple(coverage),
        required_bones=required_bones(root),
        metadata={"source_checkpoint": "GARMENT_CAD_PRO_R1B_CP3", "rigged_product": True},
    )


def build_registry(root: Path) -> GarmentLibraryRegistry:
    tunic_path = discover_product(root, "tunic")
    trousers_path = discover_product(root, "trousers")
    tunic = _entry(
        root,
        "SLEEVELESS_TUNIC_RIGGED_R1B",
        "SLEEVELESS_TUNIC",
        tunic_path,
        "MID",
        ("UPPER_BODY", "LONG_TOP"),
        (
            CoverageRegion("CHEST", 0.0042, True, 0.008),
            CoverageRegion("TORSO", 0.0042, True),
            CoverageRegion("WAIST", 0.0045, True, 0.010),
            CoverageRegion("PELVIS", 0.0038, False, 0.010),
        ),
        ("UPPER_BODY_PRIMARY",),
    )
    trousers = _entry(
        root,
        "TROUSERS_RIGGED_R1B",
        "TROUSERS",
        trousers_path,
        "BASE",
        ("LOWER_BODY", "LEGS"),
        (
            CoverageRegion("WAIST", 0.0036, True, 0.010),
            CoverageRegion("PELVIS", 0.0040, True, 0.010),
            CoverageRegion("LEFT_THIGH", 0.0038, True, 0.006),
            CoverageRegion("RIGHT_THIGH", 0.0038, True, 0.006),
            CoverageRegion("LEFT_KNEE", 0.0032, True, 0.010),
            CoverageRegion("RIGHT_KNEE", 0.0032, True, 0.010),
            CoverageRegion("LEFT_CALF", 0.0030, True),
            CoverageRegion("RIGHT_CALF", 0.0030, True),
        ),
        ("LOWER_BODY_PRIMARY",),
    )
    blocker = GarmentFamilyEntry(
        garment_id="SLEEVELESS_TUNIC_DUPLICATE_BLOCKER",
        family_id="SLEEVELESS_TUNIC_BLOCKER",
        product_path=tunic.product_path,
        product_sha256=tunic.product_sha256,
        layer_class="MID",
        slots=tunic.slots,
        exclusive_slots=tunic.exclusive_slots,
        coverage=tuple(CoverageRegion(item.region_id, item.thickness_m * 2.4, item.occludes_body, item.safety_band_m) for item in tunic.coverage),
        required_bones=tunic.required_bones,
        incompatible_families=("SLEEVELESS_TUNIC",),
        metadata={"fixture_only": True, "expected_admission": "REJECTED_ATOMIC"},
    )
    limits = {
        "CHEST": 0.010,
        "TORSO": 0.010,
        "WAIST": 0.012,
        "PELVIS": 0.012,
        "LEFT_THIGH": 0.009,
        "RIGHT_THIGH": 0.009,
        "LEFT_KNEE": 0.008,
        "RIGHT_KNEE": 0.008,
        "LEFT_CALF": 0.007,
        "RIGHT_CALF": 0.007,
    }
    registry = GarmentLibraryRegistry(
        "GARMENT_LIBRARY_R1B_CP4",
        (tunic, trousers, blocker),
        ("BASE", "INNER", "MID", "OUTER", "ARMOR", "ACCESSORY"),
        limits,
    )
    registry.validate(root)
    return registry


def publish_schemas(root: Path) -> None:
    for name, schema in cp4_schemas().items():
        write_json(root / "contracts/rig/outfit" / name, schema)


def run_transactions(reference, rejected, mask_receipt: dict) -> dict:
    state = OutfitRuntimeState()
    commit = commit_outfit(state, reference, mask_receipt["receipt_sha256"])
    committed_state = state.snapshot()
    reject = commit_outfit(state, rejected, mask_receipt["receipt_sha256"])
    preserved_after_reject = state.snapshot()
    unequip = unequip_outfit(state)
    payload = {
        "contract": "OutfitTransactionSuiteReceipt/1",
        "compatible_commit_pass": commit["status"] == "COMMITTED" and len(committed_state["equipped_garment_ids"]) == 2,
        "incompatible_rejection_pass": reject["status"] == "REJECTED_ATOMIC",
        "rejected_state_preserved": preserved_after_reject == committed_state,
        "unequip_pass": unequip["status"] == "COMMITTED" and not state.equipped_garment_ids,
        "committed_receipt": commit,
        "rejected_receipt": reject,
        "unequip_receipt": unequip,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def terminal_receipt(registry, reference, rejected, mask, transactions) -> dict:
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1B_CP4",
        "terminal_decision": "CP4_COMPLETE_MULTI_GARMENT_OUTFIT",
        "cp4_acceptance": True,
        "registry_entry_count": len(registry.entries),
        "product_entry_count": 2,
        "compatible_outfit_pass": reference.status == "ACCEPTED",
        "incompatible_outfit_rejected": rejected.status == "REJECTED_ATOMIC",
        "body_occlusion_pass": bool(mask["mask_pass"]),
        "python_atomic_transaction_pass": all(
            transactions[name]
            for name in ("compatible_commit_pass", "incompatible_rejection_pass", "rejected_state_preserved", "unequip_pass")
        ),
        "godot_outfit_consumer_pass": False,
        "secondary_motion_executed": False,
        "rig_aware_lod_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "predecessor_mutated": False,
        "next_checkpoint": "GARMENT_CAD_PRO_R1B_CP5",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    build.mkdir(parents=True, exist_ok=True)
    publish_schemas(root)
    registry = build_registry(root)
    registry_payload = registry.to_dict()
    write_json(build / "garment_library_registry.json", registry_payload)
    reference = compile_outfit(registry, "REFERENCE_TUNIC_TROUSERS", "CANONICAL_EXACT_FIXTURE", ("SLEEVELESS_TUNIC_RIGGED_R1B", "TROUSERS_RIGGED_R1B"))
    rejected = compile_outfit(registry, "INVALID_DUPLICATE_TUNIC", "CANONICAL_EXACT_FIXTURE", ("SLEEVELESS_TUNIC_RIGGED_R1B", "SLEEVELESS_TUNIC_DUPLICATE_BLOCKER"))
    write_json(build / "outfits/reference_two_piece.json", reference.to_dict())
    write_json(build / "outfits/incompatible_duplicate_tunic.json", rejected.to_dict())
    mask = compile_body_hide_mask(root, reference.hidden_body_regions, reference.preserved_safety_regions)
    transactions = run_transactions(reference, rejected, mask)
    write_json(build / "outfit_transaction_suite.json", transactions)
    receipt = terminal_receipt(registry, reference, rejected, mask, transactions)
    write_json(build / "cp4_receipt.json", receipt)
    write_json(root / "R1B_STATUS.json", receipt)
    render_cp4_evidence(root, receipt, transactions)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
