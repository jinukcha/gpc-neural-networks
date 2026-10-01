#!/usr/bin/env python3
"""Close CP5-R1 by rerunning only heavy-canvas forward bend."""
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


CP5_REL = Path("build/motion_fit_cp5")
CP5_R1_REL = Path("build/motion_fit_cp5_r1")
BUILD_REL = Path("build/motion_fit_cp5_r1_closeout")
TARGET = ("COTTON_CANVAS_HEAVY_REFERENCE", "FORWARD_BEND")
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
    payload = read_json(root / CP5_REL / "body_envelope.json")
    return BodyEnvelope(
        np.asarray(payload["center_xy"], dtype=np.float64),
        float(payload["z_min"]),
        float(payload["z_max"]),
        np.asarray(payload["z_nodes"], dtype=np.float64),
        np.asarray(payload["radius_x"], dtype=np.float64),
        np.asarray(payload["radius_y"], dtype=np.float64),
        float(payload["contact_margin_m"]),
    )


def load_material_profile(root: Path, material_id: str) -> dict:
    directory = root / "build/material_calibration_cp4/profiles" / material_id
    warp = read_json(directory / "warp_material_profile.json")
    sizing = read_json(directory / "material_sizing_profile.json")
    return {
        **warp,
        "areal_density_kg_m2": sizing["areal_density_kg_m2"],
        "thickness_m": sizing["thickness_m"],
        "recommended_meshing_edge_m": sizing["recommended_meshing_edge_m"],
    }


def scenario_base(root: Path, identity: tuple[str, str], repaired: bool) -> Path:
    material_id, pose_id = identity
    owner = CP5_R1_REL if repaired else CP5_REL
    return root / owner / "scenarios" / material_id / pose_id


def receipt_path(root: Path, identity: tuple[str, str], repaired: bool) -> Path:
    material_id, pose_id = identity
    owner = CP5_R1_REL if repaired else CP5_REL
    return root / owner / "pose_receipts" / material_id / f"{pose_id}.json"


def load_record(root: Path, identity: tuple[str, str], repaired: bool) -> dict:
    base = scenario_base(root, identity, repaired)
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
        maps = {name: np.asarray(data[name], dtype=np.float64) for name in MAP_KEYS}
    return {
        "positions": positions,
        "maps": maps,
        "receipt": read_json(receipt_path(root, identity, repaired)),
    }


def cp5_original_pass_ids(root: Path) -> set[tuple[str, str]]:
    receipt = read_json(root / CP5_REL / "cp5_receipt.json")
    failed = {tuple(item.split("::", 1)) for item in receipt["failed_scenarios"]}
    suite = professional_tunic_suite()
    materials = tuple(sorted((root / "build/material_calibration_cp4/profiles").iterdir()))
    return {
        (directory.name, pose.pose_id)
        for directory in materials
        for pose in suite.poses
        if (directory.name, pose.pose_id) not in failed
    }


def preserved_hashes(root: Path, all_ids: set[tuple[str, str]]) -> dict:
    original_pass = cp5_original_pass_ids(root)
    output = {}
    for identity in sorted(all_ids - {TARGET}):
        repaired = identity not in original_pass
        base = scenario_base(root, identity, repaired)
        output[f"{identity[0]}::{identity[1]}"] = {
            "archive_sha256": sha256_file(base.with_suffix(".npz")),
            "package_sha256": sha256_file(base.with_suffix(".json")),
            "receipt_sha256": sha256_file(receipt_path(root, identity, repaired)),
        }
    return output


def archive_closeout(root: Path, result, fit_maps) -> dict:
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
    payload = {
        "contract": "MotionFitMapPackage/1",
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP5_R1_CLOSEOUT",
        "pose_id": result.pose_id,
        "material_id": result.material_id,
        "archive_path": relative.as_posix(),
        "archive_sha256": sha256_file(path),
        "array_shapes": {name: list(value.shape) for name, value in sorted(arrays.items())},
        "canonical_archive": True,
        "source_cp5_r1_scenario": f"{result.material_id}::{result.pose_id}",
        "schedule": {
            "transition_frames": result.schedule.transition_frames,
            "settle_frames": result.schedule.settle_frames,
            "substeps": result.schedule.substeps,
            "iterations": result.schedule.iterations,
        },
    }
    payload["package_sha256"] = canonical_sha256(payload)
    write_json(path.with_suffix(".json"), payload)
    return payload


def execute_target(root: Path, mesh, envelope, pose) -> dict:
    material_id, pose_id = TARGET
    profile = load_material_profile(root, material_id)
    result = solve_failed_pose(mesh, envelope, pose, material_id, profile)
    fit_maps = compute_repair_maps(mesh, envelope, profile, result)
    metrics = repair_metrics(fit_maps, result)
    receipt = qualify_pose(pose, material_id, metrics, len(mesh.seam_pairs))
    package = archive_closeout(root, result, fit_maps)
    receipt.update(
        {
            "checkpoint": "GARMENT_CAD_PRO_R1A_CP5_R1_CLOSEOUT",
            "map_package_sha256": package["package_sha256"],
            "runtime": result.runtime,
            "repair_schedule": package["schedule"],
            "thresholds_changed": False,
        }
    )
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    write_json(root / BUILD_REL / "pose_receipts" / material_id / f"{pose_id}.json", receipt)
    return {"positions": result.positions, "maps": fit_maps.arrays(), "receipt": receipt}


def merge_records(root: Path, suite, target_record: dict):
    original_pass = cp5_original_pass_ids(root)
    materials = tuple(sorted(item.name for item in (root / "build/material_calibration_cp4/profiles").iterdir()))
    merged = {}
    repair_six = {}
    original_six = {}
    receipts = []
    all_ids = {(material_id, pose.pose_id) for material_id in materials for pose in suite.poses}
    for identity in sorted(all_ids):
        if identity == TARGET:
            record = target_record
        else:
            record = load_record(root, identity, identity not in original_pass)
        merged[identity] = record
        receipts.append(record["receipt"])
        if identity[1] in FAILED_POSES:
            original_six[identity] = load_record(root, identity, False)
            repair_six[identity] = record
    return materials, all_ids, merged, original_six, repair_six, tuple(receipts)


def write_report(root: Path, receipt: dict, target_receipt: dict) -> None:
    metrics = target_receipt["metrics"]
    report = f"""# GARMENT-CAD-PRO-R1A / CP5-R1 CLOSEOUT

```text
terminal decision       {receipt['terminal_decision']}
final pose pass         {receipt['pose_pass_count']}/{receipt['scenario_count']}
closeout rerun          1
preserved scenarios     {receipt['preserved_scenario_count']}
threshold changes       false
heavy forward strain    {metrics['strain_p99_ratio']}
heavy forward pressure  {metrics['pressure_p99_kpa']} kPa
heavy forward clearance {metrics['clearance_min_m']} m
heavy forward tail      {metrics['tail_peak_displacement_m']} m
```

첫 CP5-R1 통합본의 29개 통과 scenario는 byte-identical하게 보존하고 heavy-canvas
FORWARD_BEND 한 건만 phase-consistent body contact와 bounded settle schedule로 재실행했다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1A_CP5_R1_CLOSEOUT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    next_task = (
        "`GARMENT-CAD-PRO-R1A / CP6 — MANUFACTURING 2D / GAME 3D EXPORT / TROUSERS PRODUCT CLOSEOUT`"
        if receipt["motion_fit_pass"]
        else "`GARMENT-CAD-PRO-R1A / CP5-R2 — HEAVY FORWARD-BEND BOUNDED REPAIR`"
    )
    (root / "NEXT_TASK.md").write_text(f"# Next task\n\n{next_task}\n", encoding="utf-8")


def main() -> int:
    root = parse_args().root.resolve()
    suite = professional_tunic_suite()
    pose = next(item for item in suite.poses if item.pose_id == TARGET[1])
    mesh = load_motion_mesh(root)
    envelope = load_envelope(root)
    cp5_tree_before = hash_tree(root / CP5_REL)
    cp5_r1_tree_before = hash_tree(root / CP5_R1_REL)
    target_record = execute_target(root, mesh, envelope, pose)
    materials, all_ids, merged, original_six, repair_six, receipts = merge_records(
        root, suite, target_record
    )
    before_hashes = preserved_hashes(root, all_ids)
    authority = read_json(root / CP5_REL / "source_authority.json")
    fit_receipt = qualify_suite(suite, receipts, materials, authority)
    write_json(root / BUILD_REL / "fit_qualification_receipt.json", fit_receipt)
    after_hashes = preserved_hashes(root, all_ids)
    preservation = {
        "contract": "CP5R1CloseoutPreservationReceipt/1",
        "preserved_scenario_count": len(before_hashes),
        "preserved_hashes_before": before_hashes,
        "preserved_hashes_after": after_hashes,
        "preserved_scenarios_byte_identical": before_hashes == after_hashes,
        "cp5_tree_unchanged": cp5_tree_before == hash_tree(root / CP5_REL),
        "cp5_r1_tree_unchanged": cp5_r1_tree_before == hash_tree(root / CP5_R1_REL),
        "reexecuted_scenario_ids": [f"{TARGET[0]}::{TARGET[1]}"],
        "thresholds_changed": False,
    }
    preservation["receipt_sha256"] = canonical_sha256(preservation)
    write_json(root / BUILD_REL / "preservation_receipt.json", preservation)
    reference = "WOOL_TWILL_MEDIUM_REFERENCE"
    render_overview(root / BUILD_REL / "cp5_r1_final_overview.png", suite, reference, merged)
    render_before_after(
        root / BUILD_REL / "cp5_r1_final_before_after.png",
        reference,
        original_six,
        repair_six,
    )
    worst_pose = render_maps(
        root / BUILD_REL / "cp5_r1_final_fit_maps.png",
        reference,
        repair_six,
    )
    receipt = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP5_R1_CLOSEOUT",
        "terminal_decision": "CP5_R1_ACCEPTED" if fit_receipt["motion_fit_pass"] else "HOLD_CP5_R1_CLOSEOUT",
        "scenario_count": len(receipts),
        "pose_pass_count": fit_receipt["pose_pass_count"],
        "failed_scenarios": fit_receipt["failed_scenarios"],
        "motion_fit_pass": fit_receipt["motion_fit_pass"],
        "product_acceptance": False,
        "closeout_rerun_scenario_count": 1,
        "preserved_scenario_count": len(before_hashes),
        "preserved_scenarios_byte_identical": preservation["preserved_scenarios_byte_identical"],
        "thresholds_changed": False,
        "cp5_predecessor_mutated": not preservation["cp5_tree_unchanged"],
        "cp5_r1_predecessor_mutated": not preservation["cp5_r1_tree_unchanged"],
        "pattern_or_construction_mutated": False,
        "feature_complete_cp3_topology_triangulated": False,
        "mesh_scaling": "FORBIDDEN",
        "runtime": target_record["receipt"]["runtime"],
        "worst_reference_pose": worst_pose,
        "next_checkpoint": "GARMENT_CAD_PRO_R1A_CP6" if fit_receipt["motion_fit_pass"] else "GARMENT_CAD_PRO_R1A_CP5_R2",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(root / BUILD_REL / "cp5_r1_final_receipt.json", receipt)
    write_json(root / "PROFESSIONAL_STATUS.json", receipt)
    write_report(root, receipt, target_record["receipt"])
    print(json.dumps({"receipt": receipt, "target": target_record["receipt"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
