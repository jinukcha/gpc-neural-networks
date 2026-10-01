#!/usr/bin/env python3
"""Run CP5 motion-fit scenarios, canonical maps, qualification, and evidence boards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.pattern_cad.document.model import canonical_sha256
from wuxia_garment_oss.qualification.motion_fit.archive import write_canonical_npz
from wuxia_garment_oss.qualification.motion_fit.body import derive_body_envelope
from wuxia_garment_oss.qualification.motion_fit.contracts import motion_fit_schemas
from wuxia_garment_oss.qualification.motion_fit.maps import compute_fit_maps, map_metrics
from wuxia_garment_oss.qualification.motion_fit.mesh import load_motion_mesh
from wuxia_garment_oss.qualification.motion_fit.poses import professional_tunic_suite
from wuxia_garment_oss.qualification.motion_fit.qualification import qualify_pose, qualify_suite
from wuxia_garment_oss.qualification.motion_fit.render import (
    render_fit_maps,
    render_material_comparison,
    render_motion_overview,
)
from wuxia_garment_oss.qualification.motion_fit.solver import solve_pose


BUILD_REL = Path("build/motion_fit_cp5")


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
    target = root / "contracts/qualification"
    for filename, schema in motion_fit_schemas().items():
        write_json(target / filename, schema)


def load_material_profiles(root: Path) -> tuple[tuple[str, ...], dict[str, dict]]:
    base = root / "build/material_calibration_cp4/profiles"
    profiles = {}
    for directory in sorted(item for item in base.iterdir() if item.is_dir()):
        warp = load_json(directory / "warp_material_profile.json")
        sizing = load_json(directory / "material_sizing_profile.json")
        profiles[directory.name] = {
            **warp,
            "areal_density_kg_m2": sizing["areal_density_kg_m2"],
            "thickness_m": sizing["thickness_m"],
            "recommended_meshing_edge_m": sizing["recommended_meshing_edge_m"],
            "sizing_profile_sha256": sizing["profile_sha256"],
        }
    ids = tuple(sorted(profiles))
    if len(ids) != 3:
        raise ValueError(f"CP5 requires three calibrated material profiles, got {ids}")
    return ids, profiles


def source_authority(root: Path, mesh, material_ids: tuple[str, ...]) -> dict:
    paths = {
        "pattern_document": root / "build/pattern_cad_cp2/features/composite_pattern_document.json",
        "construction_package": root / "build/construction_cp3/construction_package.json",
        "cp4_receipt": root / "build/material_calibration_cp4/cp4_receipt.json",
        "final_mesh": root / mesh.source_paths["final_mesh"],
        "static_model": root / mesh.source_paths["static_model"],
    }
    result = {name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for name, path in paths.items()}
    result["material_profiles"] = {
        material_id: {
            "warp_sha256": sha256_file(root / f"build/material_calibration_cp4/profiles/{material_id}/warp_material_profile.json"),
            "sizing_sha256": sha256_file(root / f"build/material_calibration_cp4/profiles/{material_id}/material_sizing_profile.json"),
        }
        for material_id in material_ids
    }
    result["mesh_source_keys"] = mesh.source_keys
    result["immutable_predecessor"] = True
    return result


def archive_scenario(root: Path, material_id: str, pose_id: str, result, fit_maps) -> dict:
    relative = BUILD_REL / "scenarios" / material_id / f"{pose_id}.npz"
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
        "pose_id": pose_id,
        "material_id": material_id,
        "archive_path": relative.as_posix(),
        "array_shapes": {name: list(value.shape) for name, value in sorted(arrays.items())},
        "archive_sha256": sha256_file(path),
        "canonical_archive": True,
    }
    package["package_sha256"] = canonical_sha256(package)
    write_json(path.with_suffix(".json"), package)
    return package


def run_scenarios(root: Path, suite, mesh, envelope, material_ids, profiles):
    scenario_data = {}
    receipts = []
    packages = []
    total = len(material_ids) * len(suite.poses)
    completed = 0
    for material_id in material_ids:
        profile = profiles[material_id]
        for pose in suite.poses:
            result = solve_pose(mesh, envelope, pose, suite, material_id, profile)
            fit_maps = compute_fit_maps(mesh, envelope, pose, suite, profile, result)
            metrics = map_metrics(fit_maps, result)
            receipt = qualify_pose(pose, material_id, metrics, len(mesh.seam_pairs))
            package = archive_scenario(root, material_id, pose.pose_id, result, fit_maps)
            receipt["map_package_sha256"] = package["package_sha256"]
            receipt["runtime"] = result.runtime
            receipt["receipt_sha256"] = canonical_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
            receipt_path = root / BUILD_REL / "pose_receipts" / material_id / f"{pose.pose_id}.json"
            write_json(receipt_path, receipt)
            scenario_data[(material_id, pose.pose_id)] = {
                "positions": result.positions,
                "maps": fit_maps.arrays(),
                "receipt": receipt,
            }
            receipts.append(receipt)
            packages.append(package)
            completed += 1
            print(json.dumps({"scenario": f"{material_id}/{pose.pose_id}", "pass": receipt["pose_pass"], "completed": completed, "total": total}, sort_keys=True))
    return scenario_data, tuple(receipts), tuple(packages)


def write_report(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1A / CP5 실행 보고서

## 판정

```text
terminal decision        {receipt['terminal_decision']}
scenario count           {receipt['scenario_count']}
pose pass                 {receipt['pose_pass_count']}/{receipt['scenario_count']}
motion-fit pass           {receipt['motion_fit_pass']}
product acceptance        false
```

CP5는 CP4 calibrated material 3종과 10개 pose를 조합한 30개 quasi-static motion-fit
시나리오를 exact Warp 1.17.0 CPU drive kernel + project-owned structural/seam/contact projection으로
실행한다. Stress, strain, pressure, clearance, seam tension, contact persistence, mobility restriction을
canonical per-vertex map으로 저장한다. 사용한 3D mesh는 immutable CP2B fitted tunic reference이며,
CP3 dart·pleat·gather·gusset construction topology는 아직 triangulation되지 않았으므로 제품 수락은 아니다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1A_CP5_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    next_text = (
        "# Next task\n\n"
        "`GARMENT-CAD-PRO-R1A / CP6 — MANUFACTURING 2D / GAME 3D EXPORT / TROUSERS PRODUCT CLOSEOUT`\n\n"
        "Use CP5 motion-fit qualification and immutable CP1–CP4 authorities. Implement manufacturing pattern export, game garment product export, "
        "fresh reopen parity, and the trousers pilot closeout.\n"
        if receipt["motion_fit_pass"]
        else "# Next task\n\n`GARMENT-CAD-PRO-R1A / CP5-R1 — FAILED-POSE LOCAL REPAIR / MOTION-FIT REQUALIFICATION`\n"
    )
    (root / "NEXT_TASK.md").write_text(next_text, encoding="utf-8")


def main() -> int:
    root = parse_args().root.resolve()
    publish_schemas(root)
    suite = professional_tunic_suite()
    suite_payload = suite.to_dict()
    write_json(root / BUILD_REL / "motion_fit_suite.json", suite_payload)
    mesh = load_motion_mesh(root)
    envelope = derive_body_envelope(mesh.positions)
    write_json(root / BUILD_REL / "body_envelope.json", envelope.to_dict())
    material_ids, profiles = load_material_profiles(root)
    authority = source_authority(root, mesh, material_ids)
    write_json(root / BUILD_REL / "source_authority.json", authority)
    scenario_data, pose_receipts, packages = run_scenarios(root, suite, mesh, envelope, material_ids, profiles)
    fit_receipt = qualify_suite(suite, pose_receipts, material_ids, authority)
    write_json(root / BUILD_REL / "fit_qualification_receipt.json", fit_receipt)
    reference_material = "WOOL_TWILL_MEDIUM_REFERENCE" if "WOOL_TWILL_MEDIUM_REFERENCE" in material_ids else material_ids[1]
    render_motion_overview(root / BUILD_REL / "cp5_motion_fit_overview.png", suite, reference_material, scenario_data)
    worst_pose = render_fit_maps(root / BUILD_REL / "cp5_fit_maps_evidence.png", suite, reference_material, scenario_data)
    comparison_pose = render_material_comparison(root / BUILD_REL / "cp5_material_comparison.png", suite, material_ids, scenario_data)
    receipt = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP5",
        "terminal_decision": "CP5_COMPLETE_MOTION_FIT" if fit_receipt["motion_fit_pass"] else "HOLD_CP5_MOTION_FIT",
        "material_count": len(material_ids),
        "pose_count": len(suite.poses),
        "scenario_count": len(pose_receipts),
        "pose_pass_count": fit_receipt["pose_pass_count"],
        "failed_scenarios": fit_receipt["failed_scenarios"],
        "motion_fit_pass": fit_receipt["motion_fit_pass"],
        "product_acceptance": False,
        "scenario_package_count": len(packages),
        "worst_reference_pose": worst_pose,
        "material_comparison_pose": comparison_pose,
        "vertex_count": mesh.vertex_count,
        "triangle_count": int(len(mesh.triangles)),
        "edge_count": int(len(mesh.edges)),
        "seam_pair_count": int(len(mesh.seam_pairs)),
        "runtime": pose_receipts[0]["runtime"],
        "cp4_predecessor_mutated": False,
        "pattern_or_construction_mutated": False,
        "feature_complete_cp3_topology_triangulated": False,
        "mesh_scaling": "FORBIDDEN",
        "next_checkpoint": "GARMENT_CAD_PRO_R1A_CP6" if fit_receipt["motion_fit_pass"] else "GARMENT_CAD_PRO_R1A_CP5_R1",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(root / BUILD_REL / "cp5_receipt.json", receipt)
    write_json(root / "PROFESSIONAL_STATUS.json", receipt)
    write_report(root, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
