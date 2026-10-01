#!/usr/bin/env python3
"""Build CP6 manufacturing exports, game products, and trousers closeout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.export.contracts import cp6_schemas
from wuxia_garment_oss.export.evidence import (
    render_game_product_board,
    render_manufacturing_board,
    render_trousers_fit_board,
)
from wuxia_garment_oss.export.glb import write_glb
from wuxia_garment_oss.export.manufacturing import compile_manufacturing_package, write_package
from wuxia_garment_oss.garments.trousers.fit import POSE_IDS, pose_target, run_fit_suite
from wuxia_garment_oss.garments.trousers.mesh import build_trousers_mesh, topology_receipt
from wuxia_garment_oss.garments.trousers.pattern import build_trousers_pattern
from wuxia_garment_oss.pattern_cad.document.model import PatternDocument, canonical_sha256
from wuxia_garment_oss.pattern_cad.document.resolver import require_resolved
from wuxia_garment_oss.qualification.motion_fit.archive import write_canonical_npz


BUILD_REL = Path("build/garment_cad_pro_cp6")
TUNIC_POSES = (
    "NEUTRAL_A", "ARMS_FORWARD", "ARMS_OVERHEAD", "CROSS_BODY_REACH",
    "DEEP_ELBOW_BEND", "TORSO_TWIST", "FORWARD_BEND", "SEATED", "SQUAT", "WALK_STRIDE",
)
REPAIRED_POSES = {"ARMS_OVERHEAD", "CROSS_BODY_REACH", "FORWARD_BEND", "SEATED", "SQUAT", "WALK_STRIDE"}
REFERENCE_MATERIAL = "WOOL_TWILL_MEDIUM_REFERENCE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_schemas(root: Path) -> None:
    directory = root / "contracts/export"
    for filename, schema in cp6_schemas().items():
        write_json(directory / filename, schema)


def load_material_profiles(root: Path) -> dict[str, dict]:
    directory = root / "build/material_calibration_cp4/profiles"
    profiles = {}
    for item in sorted(path for path in directory.iterdir() if path.is_dir()):
        warp = load_json(item / "warp_material_profile.json")
        sizing = load_json(item / "material_sizing_profile.json")
        profiles[item.name] = {
            **warp,
            "thickness_m": sizing["thickness_m"],
            "areal_density_kg_m2": sizing["areal_density_kg_m2"],
        }
    if len(profiles) != 3:
        raise ValueError(f"CP6 requires 3 calibrated materials: {sorted(profiles)}")
    return profiles


def _first_array(data, names: tuple[str, ...], dimensions: int) -> np.ndarray:
    for name in names:
        if name in data.files and np.asarray(data[name]).ndim == dimensions:
            return np.asarray(data[name])
    raise KeyError(f"missing array {names}; available={sorted(data.files)}")


def load_tunic_mesh(root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = root / "build/tunic_pilot/warp_cp3/final_simulation_mesh.npz"
    with np.load(path, allow_pickle=False) as data:
        positions = _first_array(data, ("positions", "final_positions", "vertices", "x"), 2).astype(np.float64)
        triangles = _first_array(data, ("triangles", "faces", "indices"), 2).astype(np.int64)
    if positions.shape[1] != 3 or triangles.shape[1] != 3:
        raise ValueError("invalid tunic mesh arrays")
    return positions, triangles


def _scenario_path(root: Path, pose_id: str) -> Path:
    if pose_id in REPAIRED_POSES:
        return root / f"build/motion_fit_cp5_r1/scenarios/{REFERENCE_MATERIAL}/{pose_id}.npz"
    return root / f"build/motion_fit_cp5/scenarios/{REFERENCE_MATERIAL}/{pose_id}.npz"


def tunic_morph_targets(root: Path, base_positions: np.ndarray) -> dict[str, np.ndarray]:
    targets = {}
    for pose_id in TUNIC_POSES:
        with np.load(_scenario_path(root, pose_id), allow_pickle=False) as data:
            positions = np.asarray(data["positions"], dtype=np.float64)
        if positions.shape != base_positions.shape:
            raise ValueError(f"tunic morph shape mismatch: {pose_id}")
        targets[pose_id] = positions - base_positions
    return targets


def tunic_manufacturing(root: Path) -> tuple[dict, dict]:
    pattern_path = root / "build/pattern_cad_cp2/features/composite_pattern_document.json"
    construction_path = root / "build/construction_cp3/construction_package.json"
    document = PatternDocument.from_dict(load_json(pattern_path))
    resolved = require_resolved(document)
    construction = load_json(construction_path)
    source = {
        "pattern_document_path": str(pattern_path.relative_to(root)),
        "pattern_document_sha256": sha256_file(pattern_path),
        "construction_package_path": str(construction_path.relative_to(root)),
        "construction_package_sha256": sha256_file(construction_path),
    }
    package = compile_manufacturing_package("SLEEVELESS_TUNIC_PRO_R1A", resolved, construction, source)
    paths = write_package(package, root / BUILD_REL / "manufacturing/tunic")
    receipt = {
        "contract": "ManufacturingExportReceipt/1",
        "garment_id": package["garment_id"],
        "panel_count": len(package["panels"]),
        "notch_count": len(package["notches"]),
        "svg_sha256": sha256_file(paths["svg"]),
        "dxf_sha256": sha256_file(paths["dxf"]),
        "json_sha256": sha256_file(paths["json"]),
        "scale": "1:1",
        "export_pass": len(package["panels"]) >= 5 and len(package["notches"]) >= 8,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(root / BUILD_REL / "manufacturing/tunic/export_receipt.json", receipt)
    return package, receipt


def trousers_manufacturing(root: Path) -> tuple[dict, dict, dict]:
    profile_path = root / "build/pattern_cad_cp1/anthropometry_v2_profile.json"
    authority = build_trousers_pattern(load_json(profile_path))
    write_json(root / BUILD_REL / "trousers/pattern_authority.json", authority)
    source = {
        "anthropometry_path": str(profile_path.relative_to(root)),
        "anthropometry_sha256": sha256_file(profile_path),
        "trousers_pattern_authority_sha256": authority["authority_sha256"],
    }
    package = compile_manufacturing_package(
        "TROUSERS_PRO_REFERENCE_M",
        authority["resolved"],
        authority["construction"],
        source,
    )
    paths = write_package(package, root / BUILD_REL / "manufacturing/trousers")
    receipt = {
        "contract": "ManufacturingExportReceipt/1",
        "garment_id": package["garment_id"],
        "panel_count": len(package["panels"]),
        "notch_count": len(package["notches"]),
        "dart_count": authority["dart_count"],
        "gusset_panel_count": authority["gusset_panel_count"],
        "waistband_panel_count": authority["waistband_panel_count"],
        "svg_sha256": sha256_file(paths["svg"]),
        "dxf_sha256": sha256_file(paths["dxf"]),
        "json_sha256": sha256_file(paths["json"]),
        "scale": "1:1",
        "export_pass": len(package["panels"]) == 7 and authority["dart_count"] == 4,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(root / BUILD_REL / "manufacturing/trousers/export_receipt.json", receipt)
    return authority, package, receipt


def archive_trousers_results(root: Path, results) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    morphs: dict[str, dict[str, np.ndarray]] = {}
    receipts = []
    for result in results:
        relative = BUILD_REL / "trousers/fit" / result.material_id / f"{result.pose_id}.npz"
        arrays = {
            "positions": result.positions.astype(np.float32),
            "target_positions": result.target_positions.astype(np.float32),
            "frame_motion_m": result.frame_motion.astype(np.float32),
            **{name: values.astype(np.float32) for name, values in result.maps.items()},
        }
        write_canonical_npz(root / relative, arrays)
        receipt = {
            **result.receipt,
            "archive_path": relative.as_posix(),
            "archive_sha256": sha256_file(root / relative),
        }
        receipt["receipt_sha256"] = canonical_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
        write_json(root / relative.with_suffix(".json"), receipt)
        receipts.append(receipt)
        morphs.setdefault(result.material_id, {})[result.pose_id] = result.positions
    return morphs, receipts


def write_products(root: Path, trousers, authority, fit_morphs) -> tuple[dict, dict]:
    products = root / BUILD_REL / "game_products"
    tunic_positions, tunic_triangles = load_tunic_mesh(root)
    tunic_targets = tunic_morph_targets(root, tunic_positions)
    tunic_receipt = write_glb(
        products / "sleeveless_tunic.glb",
        "SLEEVELESS_TUNIC_GAME_PRODUCT_R1A",
        ({
            "name": "tunic_shell",
            "positions": tunic_positions,
            "triangles": tunic_triangles,
            "morph_targets": tunic_targets,
        },),
        TUNIC_POSES,
        {"motion_fit_authority": "CP5_R1_ACCEPTED", "construction_authority": "CP3_READ_ONLY"},
    )
    write_json(products / "sleeveless_tunic_product_receipt.json", tunic_receipt)
    pom = authority["points_of_measure"]
    reference_positions = fit_morphs[REFERENCE_MATERIAL]
    shell_targets = {name: reference_positions[name] - trousers.positions for name in POSE_IDS}
    waistband_targets = {}
    gusset_targets = {}
    for pose_id in POSE_IDS:
        waistband_target, _ = pose_target(trousers.waistband_positions, pose_id, pom)
        gusset_target, _ = pose_target(trousers.gusset_positions, pose_id, pom)
        waistband_targets[pose_id] = waistband_target - trousers.waistband_positions
        gusset_targets[pose_id] = gusset_target - trousers.gusset_positions
    trousers_receipt = write_glb(
        products / "trousers.glb",
        "TROUSERS_GAME_PRODUCT_R1A",
        (
            {"name": "trousers_shell", "positions": trousers.positions, "triangles": trousers.triangles, "morph_targets": shell_targets},
            {"name": "waistband", "positions": trousers.waistband_positions, "triangles": trousers.waistband_triangles, "morph_targets": waistband_targets},
            {"name": "crotch_gusset", "positions": trousers.gusset_positions, "triangles": trousers.gusset_triangles, "morph_targets": gusset_targets},
        ),
        POSE_IDS,
        {"fit_authority": "CP6_TROUSERS_9_SCENARIO", "pattern_authority_sha256": authority["authority_sha256"]},
    )
    write_json(products / "trousers_product_receipt.json", trousers_receipt)
    return tunic_receipt, trousers_receipt


def write_godot_project(root: Path, tunic_receipt: dict, trousers_receipt: dict) -> None:
    project = root / BUILD_REL / "godot_product"
    project.mkdir(parents=True, exist_ok=True)
    expected = {
        "tunic": {
            "path": "res://products/sleeveless_tunic.glb",
            "primitive_count": tunic_receipt["primitive_count"],
            "blend_shape_count": len(tunic_receipt["morph_target_names"]),
        },
        "trousers": {
            "path": "res://products/trousers.glb",
            "primitive_count": trousers_receipt["primitive_count"],
            "blend_shape_count": len(trousers_receipt["morph_target_names"]),
        },
    }
    write_json(project / "expected.json", expected)


def write_preliminary_receipt(
    root: Path,
    tunic_export: dict,
    trousers_export: dict,
    topology: dict,
    fit_receipts: list[dict],
    tunic_product: dict,
    trousers_product: dict,
) -> dict:
    fit_pass = all(item["pose_pass"] for item in fit_receipts)
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP6",
        "phase": "PRODUCTS_BUILT_PENDING_FRESH_CONSUMER",
        "tunic_manufacturing_pass": tunic_export["export_pass"],
        "trousers_manufacturing_pass": trousers_export["export_pass"],
        "trousers_topology_pass": topology["topology_pass"],
        "trousers_fit_scenario_count": len(fit_receipts),
        "trousers_fit_pass_count": sum(bool(item["pose_pass"]) for item in fit_receipts),
        "trousers_fit_pass": fit_pass,
        "tunic_game_product_written": tunic_product["byte_length"] > 0,
        "trousers_game_product_written": trousers_product["byte_length"] > 0,
        "cp5_r1_predecessor_mutated": False,
        "thresholds_changed": False,
        "mesh_scaling": "FORBIDDEN",
        "product_acceptance": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    write_json(root / BUILD_REL / "cp6_preliminary_receipt.json", payload)
    return payload


def main() -> int:
    root = parse_args().root.resolve()
    publish_schemas(root)
    tunic_package, tunic_export = tunic_manufacturing(root)
    authority, trousers_package, trousers_export = trousers_manufacturing(root)
    trousers = build_trousers_mesh(authority)
    topology = topology_receipt(trousers)
    write_json(root / BUILD_REL / "trousers/topology_receipt.json", topology)
    write_canonical_npz(root / BUILD_REL / "trousers/trousers_product_mesh.npz", {
        "positions": trousers.positions.astype(np.float32),
        "triangles": trousers.triangles.astype(np.int32),
        "waistband_positions": trousers.waistband_positions.astype(np.float32),
        "waistband_triangles": trousers.waistband_triangles.astype(np.int32),
        "gusset_positions": trousers.gusset_positions.astype(np.float32),
        "gusset_triangles": trousers.gusset_triangles.astype(np.int32),
    })
    profiles = load_material_profiles(root)
    results = run_fit_suite(trousers, authority["points_of_measure"], profiles)
    fit_morphs, fit_receipts = archive_trousers_results(root, results)
    tunic_product, trousers_product = write_products(root, trousers, authority, fit_morphs)
    write_godot_project(root, tunic_product, trousers_product)
    tunic_positions, tunic_triangles = load_tunic_mesh(root)
    render_manufacturing_board(root / BUILD_REL / "cp6_manufacturing_2d_evidence.png", tunic_package, trousers_package)
    render_game_product_board(root / BUILD_REL / "cp6_game_products_evidence.png", tunic_positions, tunic_triangles, trousers, len(TUNIC_POSES), len(POSE_IDS))
    render_trousers_fit_board(root / BUILD_REL / "cp6_trousers_fit_evidence.png", results)
    receipt = write_preliminary_receipt(root, tunic_export, trousers_export, topology, fit_receipts, tunic_product, trousers_product)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
