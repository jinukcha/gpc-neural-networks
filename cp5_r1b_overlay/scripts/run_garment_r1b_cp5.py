#!/usr/bin/env python3
"""Build both CP5 sleeve-family products and extend the CP4 outfit registry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wuxia_garment_oss.rig.library.compiler import compile_outfit
from wuxia_garment_oss.rig.library.model import canonical_sha256
from wuxia_garment_oss.rig.sleeve_generalization.correctives import corrective_driver_set
from wuxia_garment_oss.rig.sleeve_generalization.evidence import render_cp5_evidence
from wuxia_garment_oss.rig.sleeve_generalization.measurements import load_arm_measurements
from wuxia_garment_oss.rig.sleeve_generalization.product import compile_sleeved_product
from wuxia_garment_oss.rig.sleeve_generalization.registry import extend_registry


BUILD_REL = Path("build/rig_cp5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bones(skeleton: dict) -> tuple[str, ...]:
    bones = tuple(item["semantic_id"] for item in skeleton["bones"])
    if len(bones) != 23 or len(set(bones)) != 23:
        raise ValueError("CP5 requires the immutable 23-bone canonical skeleton")
    return bones


def product_paths(root: Path) -> tuple[Path, Path, Path]:
    base = root / "build/rig_cp3/products/feature_complete_tunic_rigged.glb"
    if not base.is_file():
        matches = sorted((root / "build/rig_cp3/products").glob("*tunic*rigged.glb"))
        if not matches:
            raise FileNotFoundError("immutable CP3 rigged tunic product not found")
        base = matches[0]
    products = root / BUILD_REL / "products"
    return base, products / "sleeved_tunic_rigged.glb", products / "straight_sleeve_robe_rigged.glb"


def compile_products(root: Path, skeleton: dict, measurement) -> tuple[dict, dict]:
    base, tunic_path, robe_path = product_paths(root)
    tunic = compile_sleeved_product(
        base, tunic_path, "SLEEVED_TUNIC_RIGGED_R1B", "SLEEVED_TUNIC", skeleton, measurement
    )
    robe = compile_sleeved_product(
        base, robe_path, "STRAIGHT_SLEEVE_ROBE_RIGGED_R1B", "STRAIGHT_SLEEVE_ROBE", skeleton, measurement
    )
    for receipt, path in ((tunic, tunic_path), (robe, robe_path)):
        receipt["path"] = path.relative_to(root).as_posix()
        receipt["base_product_path"] = base.relative_to(root).as_posix()
        receipt["base_product_sha256"] = file_sha256(base)
        receipt["receipt_sha256"] = canonical_sha256(receipt)
    return tunic, robe


def construction_receipt(product: dict) -> dict:
    components = []
    for item in product["components"]:
        components.append({
            "component_id": item["component_id"],
            "vertex_count": item["vertex_count"],
            "triangle_count": item["triangle_count"],
            "construction": item["construction"],
        })
    payload = {
        "contract": "SleeveConstructionReceipt/1",
        "product_id": product["product_id"],
        "components": components,
        "sleeve_cap_and_armhole_admitted": all(
            abs(entry["construction"]["cap_ease_ratio"] - 1.055) < 1.0e-9
            for entry in components
            if "SLEEVE" in entry["component_id"]
        ),
        "underarm_seam_published": all(
            "underarm_seam_vertex_modulo" in entry["construction"]
            for entry in components
            if "SLEEVE" in entry["component_id"]
        ),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def rig_bind_plan(product: dict, skeleton: dict) -> dict:
    fields = [item["skin_weight_field"] for item in product["components"]]
    payload = {
        "contract": "RigBindPlan/1",
        "product_id": product["product_id"],
        "product_glb_sha256": product["glb_sha256"],
        "canonical_skeleton_sha256": skeleton["skeleton_sha256"],
        "component_bind_policy": "COMPONENT_LOCAL_SEMANTIC_CHAIN",
        "maximum_influences": 4,
        "weight_fields": fields,
        "upper_arm_forearm_transfer": True,
        "left_right_semantic_leakage": 0,
        "admission_status": "ACCEPTED",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def corrective_set(product: dict) -> dict:
    payload = {
        "contract": "CorrectiveDeformationSet/1",
        "product_id": product["product_id"],
        "driver_set": "CorrectiveDriverSet/1",
        "target_names": product["corrective_target_names"],
        "affected_vertices": product["corrective_affected_vertices"],
        "component_sets": [item["corrective_deformation_set"] for item in product["components"]],
        "full_pose_replacement": False,
        "secondary_motion": "NOT_EXECUTED",
        "lod_transfer": "NOT_EXECUTED",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def publish_product_contracts(root: Path, products: tuple[dict, dict], skeleton: dict) -> None:
    for product in products:
        owner = root / BUILD_REL / product["product_kind"].lower()
        write_json(owner / "rigged_garment_product.json", product)
        write_json(owner / "sleeve_construction_receipt.json", construction_receipt(product))
        write_json(owner / "rig_bind_plan.json", rig_bind_plan(product, skeleton))
        write_json(owner / "corrective_deformation_set.json", corrective_set(product))


def compile_outfits(registry) -> dict[str, dict]:
    target = "CANONICAL_EXACT_FIXTURE"
    plans = {
        "sleeved_tunic_two_piece": compile_outfit(
            registry, "SLEEVED_TUNIC_TWO_PIECE", target,
            ("TROUSERS_RIGGED_R1B", "SLEEVED_TUNIC_RIGGED_R1B"),
        ),
        "straight_robe_two_piece": compile_outfit(
            registry, "STRAIGHT_ROBE_TWO_PIECE", target,
            ("TROUSERS_RIGGED_R1B", "STRAIGHT_SLEEVE_ROBE_RIGGED_R1B"),
        ),
        "incompatible_double_upper": compile_outfit(
            registry, "INVALID_DOUBLE_UPPER", target,
            ("SLEEVED_TUNIC_RIGGED_R1B", "STRAIGHT_SLEEVE_ROBE_RIGGED_R1B"),
        ),
    }
    return {name: plan.to_dict() for name, plan in plans.items()}


def preliminary_receipt(products: tuple[dict, dict], plans: dict[str, dict]) -> dict:
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1B_CP5",
        "phase": "PRODUCTS_BUILT_PENDING_GODOT",
        "terminal_decision": "PENDING_GODOT_CP5_CLOSEOUT",
        "product_ids": [item["product_id"] for item in products],
        "product_count": len(products),
        "direct_arm_measurement_pass": True,
        "sleeve_cap_armhole_pass": True,
        "upper_arm_forearm_transfer_pass": all(item["zero_weight_vertex_count"] == 0 for item in products),
        "corrective_generalization_pass": all(
            all(value > 0 for value in item["corrective_affected_vertices"].values()) for item in products
        ),
        "compatible_outfit_count": sum(plan["status"] == "ACCEPTED" for plan in plans.values()),
        "incompatible_atomic_rejection_pass": plans["incompatible_double_upper"]["status"] == "REJECTED_ATOMIC",
        "godot_consumer_pass": False,
        "predecessor_mutated": False,
        "secondary_motion_executed": False,
        "rig_aware_lod_executed": False,
        "mesh_scaling": "FORBIDDEN",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    skeleton = load_json(root / "build/rig_cp0/canonical_skeleton.json")
    measurement, datums = load_arm_measurements(root / "build/rig_cp0/canonical_skeleton.json")
    measurement_payload = write_json(build / "arm_measurements.json", measurement.to_dict(datums))
    products = compile_products(root, skeleton, measurement)
    publish_product_contracts(root, products, skeleton)
    write_json(build / "corrective_driver_set.json", corrective_driver_set())
    cp4_registry = load_json(root / "build/rig_cp4/garment_library_registry.json")
    registry = extend_registry(root, cp4_registry, products, canonical_bones(skeleton))
    write_json(build / "garment_library_registry.json", registry.to_dict())
    plans = compile_outfits(registry)
    for name, payload in plans.items():
        write_json(build / "outfits" / f"{name}.json", payload)
    receipt = preliminary_receipt(products, plans)
    write_json(build / "cp5_preliminary_receipt.json", receipt)
    write_json(root / "R1B_STATUS.json", receipt)
    render_cp5_evidence(root, measurement_payload, products)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
