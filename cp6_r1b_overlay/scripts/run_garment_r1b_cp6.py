#!/usr/bin/env python3
"""Compile CP6 secondary motion and rig-aware LOD products from immutable CP5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wuxia_garment_oss.rig.library.model import canonical_sha256
from wuxia_garment_oss.rig.runtime_product.lod import (
    compile_lod_glb,
    inspect_rigged_glb,
    qualify_lod_set,
)
from wuxia_garment_oss.rig.runtime_product.secondary_motion import (
    compile_secondary_motion_glb,
    product_profile,
)


BUILD_REL = Path("build/rig_cp6")
PRODUCTS = (
    {
        "product_id": "SLEEVED_TUNIC_RIGGED_R1B",
        "product_kind": "SLEEVED_TUNIC",
        "source": "build/rig_cp5/products/sleeved_tunic_rigged.glb",
        "owner": "sleeved_tunic",
    },
    {
        "product_id": "STRAIGHT_SLEEVE_ROBE_RIGGED_R1B",
        "product_kind": "STRAIGHT_SLEEVE_ROBE",
        "source": "build/rig_cp5/products/straight_sleeve_robe_rigged.glb",
        "owner": "straight_robe",
    },
)


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


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _lod0_receipt(path: Path) -> dict:
    return {
        "contract": "RigAwareLODProduct/1",
        "lod_id": "LOD0",
        "target_ratio": 1.0,
        "source_path": path.name,
        "target_path": path.name,
        "changed_primitive_count": 0,
        "primitive_receipts": [],
        "metrics": inspect_rigged_glb(path),
    }


def _compile_product(root: Path, spec: dict) -> dict:
    source = root / spec["source"]
    if not source.is_file():
        raise FileNotFoundError(source)
    owner = root / BUILD_REL / "products" / spec["owner"]
    lod0 = owner / "lod0.glb"
    lod1 = owner / "lod1.glb"
    lod2 = owner / "lod2.glb"
    secondary = compile_secondary_motion_glb(source, lod0, spec["product_kind"])
    lod_receipts = (
        _lod0_receipt(lod0),
        compile_lod_glb(lod0, lod1, "LOD1"),
        compile_lod_glb(lod0, lod2, "LOD2"),
    )
    metrics = tuple(item["metrics"] for item in lod_receipts)
    qualification = qualify_lod_set(metrics)
    if not qualification["accepted"]:
        raise ValueError(f"LOD qualification failed: {spec['product_id']}")
    profile = product_profile(spec["product_id"], spec["product_kind"]).to_dict()
    write_json(owner / "secondary_motion_profile.json", profile)
    write_json(owner / "secondary_motion_morph_set.json", secondary)
    write_json(owner / "lod_qualification.json", qualification)
    for receipt in lod_receipts:
        write_json(owner / f"{receipt['lod_id'].lower()}_receipt.json", receipt)
    package = {
        "contract": "RuntimeGarmentLODSet/1",
        "product_id": spec["product_id"],
        "product_kind": spec["product_kind"],
        "immutable_cp5_source": spec["source"],
        "immutable_cp5_sha256": file_sha256(source),
        "secondary_motion_profile_sha256": profile["profile_sha256"],
        "variants": {
            "LOD0": _relative(root, lod0),
            "LOD1": _relative(root, lod1),
            "LOD2": _relative(root, lod2),
        },
        "distance_thresholds_m": {"LOD0_MAX": 5.0, "LOD1_MAX": 14.0},
        "state_transfer_key": "DOMAIN_ID_AND_BLEND_SHAPE_NAME",
        "corrective_target_names": secondary["original_target_names"],
        "secondary_target_names": secondary["secondary_target_names"],
        "lod_qualification": qualification,
        "secondary_motion_profile": profile,
    }
    package["package_sha256"] = canonical_sha256(package)
    return write_json(owner / "runtime_garment_lod_set.json", package)


def _runtime_registry(root: Path, packages: tuple[dict, ...], cp5_registry: dict) -> dict:
    payload = {
        "contract": "RuntimeGarmentLODRegistry/1",
        "registry_id": "GARMENT_RUNTIME_R1B_CP6",
        "immutable_cp5_registry_sha256": cp5_registry["registry_sha256"],
        "distance_policy": {
            "near": {"lod": "LOD0", "maximum_distance_m": 5.0},
            "mid": {"lod": "LOD1", "maximum_distance_m": 14.0},
            "far": {"lod": "LOD2", "maximum_distance_m": None},
            "hysteresis_m": 0.75,
        },
        "products": [
            {
                "product_id": item["product_id"],
                "product_kind": item["product_kind"],
                "package_path": _relative(
                    root,
                    root / BUILD_REL / "products" / (
                        "sleeved_tunic" if item["product_kind"] == "SLEEVED_TUNIC" else "straight_robe"
                    ) / "runtime_garment_lod_set.json",
                ),
                "package_sha256": item["package_sha256"],
                "variants": item["variants"],
            }
            for item in packages
        ],
        "corrective_state_transfer": "REQUIRED",
        "secondary_state_transfer": "REQUIRED",
        "atomic_variant_swap": True,
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    return payload


def _preliminary_receipt(packages: tuple[dict, ...], registry: dict) -> dict:
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1B_CP6",
        "phase": "PRODUCTS_COMPILED_PENDING_GODOT_CAPTURE",
        "terminal_decision": "PENDING_GODOT_MULTI_DISTANCE_CLOSEOUT",
        "product_count": len(packages),
        "lod_variant_count": sum(len(item["variants"]) for item in packages),
        "secondary_motion_domain_count": sum(
            len(item["secondary_motion_profile"]["domains"]) for item in packages
        ),
        "lod_qualification_pass": all(item["lod_qualification"]["accepted"] for item in packages),
        "skin_continuity_pass": all(item["lod_qualification"]["skin_preserved"] for item in packages),
        "corrective_continuity_pass": all(
            item["lod_qualification"]["active_morph_targets_preserved"] for item in packages
        ),
        "runtime_registry_sha256": registry["registry_sha256"],
        "godot_runtime_pass": False,
        "multi_distance_capture_pass": False,
        "cp5_predecessor_mutated": False,
        "mesh_scaling": "FORBIDDEN",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write_report_stub(root: Path) -> None:
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1B_CP6_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# GARMENT-CAD-PRO-R1B / CP6\n\n"
        "Secondary-motion and rig-aware LOD products are compiled. "
        "Exact Godot 4.7.2 captures and terminal closeout are pending.\n",
        encoding="utf-8",
    )


def main() -> int:
    root = parse_args().root.resolve()
    cp5_registry = load_json(root / "build/rig_cp5/garment_library_registry.json")
    packages = tuple(_compile_product(root, spec) for spec in PRODUCTS)
    registry = _runtime_registry(root, packages, cp5_registry)
    write_json(root / BUILD_REL / "runtime_garment_lod_registry.json", registry)
    receipt = _preliminary_receipt(packages, registry)
    write_json(root / BUILD_REL / "cp6_preliminary_receipt.json", receipt)
    write_json(root / "R1B_STATUS.json", receipt)
    _write_report_stub(root)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
