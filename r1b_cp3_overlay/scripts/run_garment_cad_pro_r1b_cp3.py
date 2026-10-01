#!/usr/bin/env python3
"""Build rigged GLB products and CP3 contracts from the accepted R1A products."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.rig.game_product.glb_skin import (
    BONES,
    _all_positions,
    _global_matrices,
    _skeleton_local_translations,
    compile_skinned_glb,
    read_glb,
)


BUILD_REL = Path("build/rig_cp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def locate_product(root: Path, kind: str) -> Path:
    exact = {
        "tunic": root / "build/garment_cad_pro_cp6_r1/game_products/feature_complete_tunic.glb",
        "trousers": root / "build/garment_cad_pro_cp6/game_products/trousers.glb",
    }[kind]
    if exact.is_file():
        return exact
    candidates = sorted(root.glob(f"build/**/{'feature_complete_tunic' if kind == 'tunic' else 'trousers'}.glb"))
    if not candidates:
        raise FileNotFoundError(f"unable to locate {kind} product GLB")
    return candidates[0]


def skeleton_receipt(tunic_source: Path) -> dict:
    glb = read_glb(tunic_source)
    local = _skeleton_local_translations(_all_positions(glb))
    globals_ = _global_matrices(local)
    payload = {
        "contract": "CanonicalSkeletonPackage/1",
        "coordinate_frame": "SOURCE_PRODUCT_FRAME",
        "bone_count": len(BONES),
        "bones": [
            {
                "semantic_id": name,
                "parent": parent,
                "local_translation": local[index],
                "global_translation": globals_[index][:3, 3].astype(float).tolist(),
            }
            for index, (name, parent) in enumerate(BONES)
        ],
        "global_joint_positions": {
            name: globals_[index][:3, 3].astype(float).tolist()
            for index, (name, _) in enumerate(BONES)
        },
    }
    payload["skeleton_sha256"] = canonical_sha256(payload)
    return payload


def adapter_packages() -> tuple[dict, dict, dict]:
    canonical = {name: name for name, _ in BONES}
    mixamo_names = {
        "ROOT": "mixamorig:Root", "PELVIS": "mixamorig:Hips",
        "SPINE_01": "mixamorig:Spine", "SPINE_02": "mixamorig:Spine1",
        "CHEST": "mixamorig:Spine2", "NECK": "mixamorig:Neck", "HEAD": "mixamorig:Head",
        "L_CLAVICLE": "mixamorig:LeftShoulder", "L_UPPER_ARM": "mixamorig:LeftArm",
        "L_FOREARM": "mixamorig:LeftForeArm", "L_HAND": "mixamorig:LeftHand",
        "R_CLAVICLE": "mixamorig:RightShoulder", "R_UPPER_ARM": "mixamorig:RightArm",
        "R_FOREARM": "mixamorig:RightForeArm", "R_HAND": "mixamorig:RightHand",
        "L_THIGH": "mixamorig:LeftUpLeg", "L_CALF": "mixamorig:LeftLeg",
        "L_FOOT": "mixamorig:LeftFoot", "L_TOE": "mixamorig:LeftToeBase",
        "R_THIGH": "mixamorig:RightUpLeg", "R_CALF": "mixamorig:RightLeg",
        "R_FOOT": "mixamorig:RightFoot", "R_TOE": "mixamorig:RightToeBase",
    }
    exact = {
        "contract": "SkeletonAdapterMap/1",
        "adapter_id": "CANONICAL_EXACT_V1",
        "semantic_to_source": canonical,
        "rest_offset_policy": "IDENTITY",
    }
    mixamo = {
        "contract": "SkeletonAdapterMap/1",
        "adapter_id": "MIXAMO_STYLE_V1",
        "semantic_to_source": mixamo_names,
        "rest_offset_policy": "RELAXED_A_FROM_T_POSE",
        "upper_arm_offset_degrees": {"L_UPPER_ARM": -35.0, "R_UPPER_ARM": 35.0},
    }
    incompatible = {
        "contract": "SkeletonAdapterMap/1",
        "adapter_id": "MIXAMO_MISSING_R_CALF",
        "semantic_to_source": {key: value for key, value in mixamo_names.items() if key != "R_CALF"},
        "expected_result": "REJECT_INCOMPATIBLE_RIG",
    }
    return exact, mixamo, incompatible


def restored_provenance(root: Path, sources: dict[str, Path]) -> dict:
    payload = {
        "contract": "R1BCP3RestoredInputProvenance/1",
        "reason": "LOCAL_EXECUTION_SURFACE_CLIENT_ERROR_AND_NO_PUBLISHED_CP2_CHECKPOINT",
        "acquisition_attempts": [
            "conversation-mounted GARMENT_R1B_CP2_FINAL.zip via container",
            "GitHub branch/artifact search for R1B CP2",
        ],
        "exact_cp2_byte_continuity_verified": False,
        "reconstruction_scope": [
            "canonical 23-bone skeleton contract",
            "deterministic body-to-garment semantic skin weights",
            "corrective contract metadata from accepted ten-pose product morph identities",
        ],
        "authoritative_sources": {
            name: {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
            for name, path in sources.items()
        },
        "r1a_products_mutated": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_restored_contracts(root: Path, skeleton: dict, product_results: dict) -> None:
    exact, mixamo, incompatible = adapter_packages()
    write_json(root / "build/rig_cp0/canonical_skeleton.json", skeleton)
    for package in (exact, mixamo, incompatible):
        write_json(root / "build/rig_cp0/adapters" / f"{package['adapter_id']}.json", package)
    for garment, result in product_results.items():
        weight = {
            "contract": "SkinWeightField/1",
            "garment_id": garment.upper(),
            "source": "R1B_CP3_DETERMINISTIC_RESTORATION",
            "vertex_count": result["vertex_count"],
            "maximum_influences": 4,
            "maximum_weight_sum_error": result["maximum_weight_sum_error"],
            "zero_weight_vertex_count": result["zero_weight_vertex_count"],
            "product_glb": result["target"],
        }
        corrective = {
            "contract": "CorrectiveDeformationSet/1",
            "garment_id": garment.upper(),
            "owner_mode": "SPARSE_LOCAL_CORRECTIVE_METADATA",
            "diagnostic_full_pose_morphs_preserved": True,
            "pose_ids": [
                "NEUTRAL_A", "ARMS_FORWARD", "ARMS_OVERHEAD", "CROSS_BODY_REACH",
                "DEEP_ELBOW_BEND", "TORSO_TWIST", "FORWARD_BEND", "SEATED", "SQUAT", "WALK_STRIDE",
            ] if garment == "tunic" else ["SEATED", "SQUAT", "WALK_STRIDE"],
            "secondary_motion": "NOT_EXECUTED",
        }
        write_json(root / f"build/rig_cp1/{garment}/skin_weight_field.json", weight)
        write_json(root / f"build/rig_cp2/{garment}/corrective_deformation_set.json", corrective)


def product_package(garment: str, result: dict, path: Path, skeleton: dict) -> dict:
    payload = {
        "contract": "RiggedGarmentProduct/1",
        "garment_id": garment.upper(),
        "path": path.name,
        "glb_sha256": file_sha256(path),
        "canonical_skeleton_sha256": skeleton["skeleton_sha256"],
        "bone_count": result["bone_count"],
        "primitive_count": result["primitive_count"],
        "vertex_count": result["vertex_count"],
        "skin_count": result["skin_count"],
        "inverse_bind_matrix_count": result["inverse_bind_matrix_count"],
        "maximum_weight_sum_error": result["maximum_weight_sum_error"],
        "zero_weight_vertex_count": result["zero_weight_vertex_count"],
        "corrective_input": "CorrectiveDeformationSet/1",
        "secondary_motion": "NOT_EXECUTED",
        "lod": "NOT_EXECUTED",
    }
    payload["product_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    products = build / "products"
    tunic_source = locate_product(root, "tunic")
    trousers_source = locate_product(root, "trousers")
    sources = {"tunic": tunic_source, "trousers": trousers_source}
    write_json(build / "restored_input_provenance.json", restored_provenance(root, sources))
    skeleton = write_json(build / "canonical_skeleton.json", skeleton_receipt(tunic_source))
    tunic_target = products / "feature_complete_tunic_rigged.glb"
    trousers_target = products / "trousers_rigged.glb"
    results = {
        "tunic": compile_skinned_glb(tunic_source, tunic_target, "tunic"),
        "trousers": compile_skinned_glb(trousers_source, trousers_target, "trousers"),
    }
    write_restored_contracts(root, skeleton, results)
    packages = {
        "tunic": product_package("tunic", results["tunic"], tunic_target, skeleton),
        "trousers": product_package("trousers", results["trousers"], trousers_target, skeleton),
    }
    for garment, package in packages.items():
        write_json(build / garment / "rigged_garment_product.json", package)
    preliminary = {
        "checkpoint": "GARMENT_CAD_PRO_R1B_CP3",
        "phase": "RIGGED_PRODUCTS_BUILT_PENDING_GODOT",
        "terminal_decision": "PENDING_GODOT_EQUIP_CLOSEOUT",
        "tunic": results["tunic"],
        "trousers": results["trousers"],
        "glb_products_written": True,
        "corrective_deformation_consumed": True,
        "secondary_motion_executed": False,
        "lod_executed": False,
        "outfit_composition_executed": False,
        "mesh_scaling": "FORBIDDEN",
    }
    preliminary["receipt_sha256"] = canonical_sha256(preliminary)
    write_json(build / "cp3_preliminary_receipt.json", preliminary)
    print(json.dumps(preliminary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
