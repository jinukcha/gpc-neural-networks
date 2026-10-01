#!/usr/bin/env python3
"""Re-run only the eighteen failed CP5 pose/material scenarios."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.pattern_cad.document.model import canonical_sha256
from wuxia_garment_oss.qualification.motion_fit.archive import write_canonical_npz
from wuxia_garment_oss.qualification.motion_fit.body import BodyEnvelope
from wuxia_garment_oss.qualification.motion_fit.mesh import load_motion_mesh
from wuxia_garment_oss.qualification.motion_fit.poses import professional_tunic_suite
from wuxia_garment_oss.qualification.motion_fit.qualification import qualify_pose, qualify_suite
from wuxia_garment_oss.qualification.motion_fit_repair.maps import compute_repair_maps, repair_metrics
from wuxia_garment_oss.qualification.motion_fit_repair.render import (
    FAILED_POSES,
    render_before_after,
    render_maps,
    render_overview,
)
from wuxia_garment_oss.qualification.motion_fit_repair.solver import solve_failed_pose


SOURCE_REL = Path("build/motion_fit_cp5")
BUILD_REL = Path("build/motion_fit_cp5_r1")
MAP_KEYS = (
    "stress_n_m",
    "strain_ratio",
    "pressure_pa",
    "clearance_m",
    "seam_tension_n_m",
    "contact_persistence",
    "mobility_restriction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
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


def hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def load_envelope(root: Path) -> BodyEnvelope:
    payload = read_json(root / SOURCE_REL / "body_envelope.json")
    return BodyEnvelope(
        np.asarray(payload["center_xy"], dtype=np.float64),
        float(payload["z_min"]),
        float(payload["z_max"]),
        np.asarray(payload["z_nodes"], dtype=np.float64),
        np.asarray(payload["radius_x"], dtype=np.float64),
        np.asarray(payload["radius_y"], dtype=np.float64),
        float(payload["contact_margin_m"]),
    )


def load_material_profiles(root: Path) -> tuple[tuple[str, ...], dict[str, dict]]:
    base = root / "build/material_calibration_cp4/profiles"
    profiles = {}
    for directory in sorted(item for item in base.iterdir() if item.is_dir()):
        warp = read_json(directory / "warp_material_profile.json")
        sizing = read_json(directory / "material_sizing_profile.json")
        profiles[directory.name] = {
            **warp,
            "areal_density_kg_m2": sizing["areal_density_kg_m2"],
            "thickness_m": sizing["thickness_m"],
            "recommended_meshing_edge_m": sizing["recommended_meshing_edge_m"],
        }
    ids = tuple(sorted(profiles))
    if len(ids) != 3:
        raise ValueError(f"expected three materials, got {ids}")
    return ids, profiles


def scenario_paths(root: Path, material_id: str, pose_id: str) -> tuple[Path, Path, Path]:
    archive = root / SOURCE_REL / "scenarios" / material_id / f"{pose_id}.npz"
    package = archive.with_suffix(".json")
    receipt = root / SOURCE_REL / "pose_receipts" / material_id / f"{pose_id}.json"
    return archive, package, receipt


def load_scenario(root: Path, material_id: str, pose_id: str) -> dict:
    archive, _, receipt_path = scenario_paths(root, material_id, pose_id)
    with np.load(archive, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
        maps = {name: np.asarray(data[name], dtype=np.float64) for name in MAP_KEYS}
    return {"positions": positions, "maps": maps, "receipt": read_json(receipt_path)}


def passed_hashes(root: Path, material_ids: tuple[str, ...], failed: set[tuple[str, str]], suite) -> dict:
    output = {}
    for material_id in material_ids:
        for pose in suite.poses:
            identity = (material_id, pose.pose_id)
            if identity in failed:
                continue
            archive, package, receipt = scenario_paths(root, *identity)
            output[f"{material_id}::{pose.pose_id}"] = {
                "archive_sha256": sha256_file(archive),
                "package_sha256": sha256_file(package),
                "receipt_sha256": sha256_file(receipt),
            }
    return output


def archive_repair(root: Path, result, fit_maps) -> dict:
    relative = BUILD_REL / "scenarios" / result.material_id / f"{result.pose_id}.npz"
    path = root / relative
    arrays = {
        "positions": result.positions.astype(np.float32),
        "target_positions": result.target_positions.astype(np.float32),
        "drive_weights": result.drive_weights.astype(np.float32),
        "frame_max_displacement_m": result.frame_max_displacement_m.astype(np.float32),
        **{name: values.astype(np.float32) for name, values in fit_maps.arrays().items()},
    }
    write_canonical_npz(path, arrays)
    package = {
        "contract": "MotionFitMapPackage/1",
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP5_R1",
        "pose_id": result.pose_id,
        "material_id": result.material_id,
        "archive_path": relative.as_posix(),
        "array_shapes": {name: list(value.shape) for name, value in sorted(arrays.items())},
        "archive_sha256": sha256_file(path),
        "canonical_archive": True,
        "source_cp5_scenario": f"{result.material_id}::{result.pose_id}",
        "schedule": {
            "transition_frames": result.schedule.transition_frames,
            "settle_frames": result.schedule.settle_frames,
            "substeps": result.schedule.substeps,
            "iterations": result.schedule.iterations,
        },
    }
    package["package_sha256"] = canonical_sha256(package)
    write_json(path.with_suffix(".json"), package)
    return package


def execute_repairs(root: Path, suite, mesh, envelope, material_ids, profiles, failed):
    pose_by_id = {pose.pose_id: pose for pose in suite.poses}
    repaired = {}
    receipts = []
    packages = []
    completed = 0
    for material_id, pose_id in sorted(failed):
        pose = pose_by_id[pose_id]
        result = solve_failed_pose(mesh, envelope, pose, material_id, profiles[material_id])
        fit_maps = compute_repair_maps(mesh, envelope, profiles[material_id], result)
        receipt = qualify_pose(pose, material_id, repair_metrics(fit_maps, result), len(mesh.seam_pairs))
        package = archive_repair(root, result, fit_maps)
        receipt["checkpoint"] = "GARMENT_CAD_PRO_R1A_CP5_R1"
        receipt["map_package_sha256"] = package["package_sha256"]
        receipt["runtime"] = result.runtime
        receipt["repair_schedule"] = package["schedule"]
        receipt["thresholds_changed"] = False
        receipt["receipt_sha256"] = canonical_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
        write_json(root / BUILD_REL / "pose_receipts" / material_id / f"{pose_id}.json", receipt)
        repaired[(material_id, pose_id)] = {
            "positions": result.positions,
            "maps": fit_maps.arrays(),
            "receipt": receipt,
        }
        receipts.append(receipt)
        packages.append(package)
        completed += 1
        print(json.dumps({"scenario": f"{material_id}/{pose_id}", "pass": receipt["pose_pass"], "completed": completed, "total": len(failed)}, sort_keys=True))
    return repaired, tuple(receipts), tuple(packages)


def merge_records(root: Path, suite, material_ids, failed, repaired):
    original = {}
    merged = {}
    receipts = []
    for material_id in material_ids:
        for pose in suite.poses:
            identity = (material_id, pose.pose_id)
            original[identity] = load_scenario(root, *identity)
            record = repaired[identity] if identity in failed else original[identity]
            merged[identity] = record
            receipts.append(record["receipt"])
    return original, merged, tuple(receipts)


def write_report(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1A / CP5-R1 실행 보고서

## 판정

```text
terminal decision        {receipt['terminal_decision']}
failed scenarios rerun   {receipt['rerun_scenario_count']}
preserved scenarios      {receipt['preserved_scenario_count']}
final pose pass           {receipt['pose_pass_count']}/{receipt['scenario_count']}
motion-fit pass           {receipt['motion_fit_pass']}
threshold changes         false
```

CP5에서 통과한 12개 canonical scenario는 byte-identical하게 보존했다. 실패한 18개만
segmented torso/pelvis pose field, pose-consistent body inverse, pelvis/thigh lower-body contact,
leg-aware stride transport와 transition 이후 settle schedule로 다시 실행했다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1A_CP5_R1_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    next_task = (
        "`GARMENT-CAD-PRO-R1A / CP6 — MANUFACTURING 2D / GAME 3D EXPORT / TROUSERS PRODUCT CLOSEOUT`"
        if receipt["motion_fit_pass"]
        else "`GARMENT-CAD-PRO-R1A / CP5-R2 — REMAINING FAILED-POSE BOUNDED REPAIR`"
    )
    (root / "NEXT_TASK.md").write_text(f"# Next task\n\n{next_task}\n", encoding="utf-8")


def main() -> int:
    root = parse_args().root.resolve()
    cp5_tree_before = hash_tree(root / SOURCE_REL)
    suite = professional_tunic_suite()
    mesh = load_motion_mesh(root)
    envelope = load_envelope(root)
    material_ids, profiles = load_material_profiles(root)
    cp5_receipt = read_json(root / SOURCE_REL / "cp5_receipt.json")
    failed = {tuple(value.split("::", 1)) for value in cp5_receipt["failed_scenarios"]}
    expected_failed = {(material, pose) for material in material_ids for pose in FAILED_POSES}
    if failed != expected_failed or len(failed) != 18:
        raise ValueError(f"unexpected CP5 failed set: {sorted(failed)}")
    pass_hash_before = passed_hashes(root, material_ids, failed, suite)
    repaired, repair_receipts, packages = execute_repairs(root, suite, mesh, envelope, material_ids, profiles, failed)
    original, merged, merged_receipts = merge_records(root, suite, material_ids, failed, repaired)
    authority = read_json(root / SOURCE_REL / "source_authority.json")
    fit_receipt = qualify_suite(suite, merged_receipts, material_ids, authority)
    write_json(root / BUILD_REL / "fit_qualification_receipt.json", fit_receipt)
    pass_hash_after = passed_hashes(root, material_ids, failed, suite)
    cp5_tree_after = hash_tree(root / SOURCE_REL)
    preservation = {
        "contract": "CP5ScenarioPreservationReceipt/1",
        "preserved_scenario_count": len(pass_hash_before),
        "passed_hashes_before": pass_hash_before,
        "passed_hashes_after": pass_hash_after,
        "passed_scenarios_byte_identical": pass_hash_before == pass_hash_after,
        "cp5_build_tree_before": cp5_tree_before,
        "cp5_build_tree_after": cp5_tree_after,
        "cp5_build_tree_unchanged": cp5_tree_before == cp5_tree_after,
        "reexecuted_passed_scenarios": 0,
    }
    preservation["receipt_sha256"] = canonical_sha256(preservation)
    write_json(root / BUILD_REL / "preservation_receipt.json", preservation)
    reference = "WOOL_TWILL_MEDIUM_REFERENCE"
    render_overview(root / BUILD_REL / "cp5_r1_motion_fit_overview.png", suite, reference, merged)
    render_before_after(root / BUILD_REL / "cp5_r1_before_after.png", reference, original, repaired)
    worst_pose = render_maps(root / BUILD_REL / "cp5_r1_fit_maps_evidence.png", reference, repaired)
    receipt = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP5_R1",
        "terminal_decision": "CP5_R1_ACCEPTED" if fit_receipt["motion_fit_pass"] else "HOLD_CP5_R1_MOTION_FIT",
        "scenario_count": len(merged_receipts),
        "rerun_scenario_count": len(repair_receipts),
        "preserved_scenario_count": len(pass_hash_before),
        "pose_pass_count": fit_receipt["pose_pass_count"],
        "failed_scenarios": fit_receipt["failed_scenarios"],
        "motion_fit_pass": fit_receipt["motion_fit_pass"],
        "product_acceptance": False,
        "repair_package_count": len(packages),
        "worst_repaired_reference_pose": worst_pose,
        "thresholds_changed": False,
        "passed_scenarios_reexecuted": 0,
        "cp5_predecessor_mutated": not preservation["cp5_build_tree_unchanged"],
        "pattern_or_construction_mutated": False,
        "feature_complete_cp3_topology_triangulated": False,
        "mesh_scaling": "FORBIDDEN",
        "runtime": repair_receipts[0]["runtime"],
        "next_checkpoint": "GARMENT_CAD_PRO_R1A_CP6" if fit_receipt["motion_fit_pass"] else "GARMENT_CAD_PRO_R1A_CP5_R2",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(root / BUILD_REL / "cp5_r1_receipt.json", receipt)
    write_json(root / "PROFESSIONAL_STATUS.json", receipt)
    write_report(root, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
