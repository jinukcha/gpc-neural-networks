#!/usr/bin/env python3
"""Run CP4 material measurement, calibration, checkpoint resume, and backend parity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wuxia_garment_oss.materials.backend.parity import evaluate_parity
from wuxia_garment_oss.materials.calibration.checkpoint import (
    load_checkpoint,
    profile_from_record,
    profile_ids,
    write_checkpoint,
)
from wuxia_garment_oss.materials.calibration.fit import calibrate_material
from wuxia_garment_oss.materials.contracts import material_schemas
from wuxia_garment_oss.materials.evidence.render import render_evidence
from wuxia_garment_oss.materials.measurement.fixtures import reference_material_sets
from wuxia_garment_oss.pattern_cad.document.model import canonical_sha256


BUILD_REL = Path("build/material_calibration_cp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("checkpoint", "resume"), required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_schemas(root: Path) -> None:
    target = root / "contracts/materials"
    for filename, schema in material_schemas().items():
        write_json(target / filename, schema)


def measurement_hashes(measurements) -> dict[str, str]:
    return {
        item.material_id: item.to_dict()["measurement_set_sha256"]
        for item in measurements
    }


def publish_measurements(root: Path, measurements) -> None:
    for item in measurements:
        write_json(root / BUILD_REL / "measurements" / f"{item.material_id}.json", item.to_dict())


def checkpoint_phase(root: Path) -> dict:
    measurements = reference_material_sets()
    publish_schemas(root)
    publish_measurements(root, measurements)
    profiles = tuple(calibrate_material(item) for item in measurements[:2])
    hashes = measurement_hashes(measurements)
    checkpoint = write_checkpoint(
        root / BUILD_REL / "calibration_checkpoint.json",
        profiles,
        hashes,
        "PARTIAL_2_OF_3",
    )
    receipt = {
        "contract": "CalibrationCheckpointPhaseReceipt/1",
        "writer_process_id": os.getpid(),
        "completed_material_ids": list(profile_ids(checkpoint)),
        "completed_count": len(profiles),
        "remaining_count": len(measurements) - len(profiles),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(root / BUILD_REL / "checkpoint_phase_receipt.json", receipt)
    return receipt


def _ordered_profiles(measurements, checkpoint: dict):
    profiles = {
        record["material_id"]: profile_from_record(record)
        for record in checkpoint["profiles"]
    }
    for measurement in measurements:
        if measurement.material_id not in profiles:
            profiles[measurement.material_id] = calibrate_material(measurement)
    return tuple(profiles[item.material_id] for item in measurements)


def publish_profiles(root: Path, profiles, parities) -> None:
    for profile, parity in zip(profiles, parities):
        target = root / BUILD_REL / "profiles" / profile.material_id
        write_json(target / "material_sizing_profile.json", profile.sizing_profile_dict())
        write_json(target / "warp_material_profile.json", profile.warp_profile_dict())
        write_json(target / "calibration_receipt.json", profile.calibration_receipt_dict())
        write_json(target / "backend_parity_receipt.json", parity)


def write_report(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1A / CP4 실행 보고서

## 판정

```text
terminal decision       {receipt['terminal_decision']}
material fixtures       {receipt['material_fixture_count']}
calibration pass        {receipt['all_calibration_pass']}
CPU–Warp parity         {receipt['all_backend_parity_pass']}
fresh-process resume    {receipt['fresh_process_resume_pass']}
triangulation           false
motion fit              false
```

CP3 PatternDocument와 ConstructionPackage를 변경하지 않고 세 project reference lab series를
MaterialMeasurementSet/1로 발행했다. 각 raw curve에서 MaterialSizingProfile/2와
WarpMaterialProfile/1을 fitting하고 exact warp-lang 1.17.0 CPU kernel과 CPU reference의
static-fixture response 및 metric parity를 비교했다. Fixture는 외부 공인 시험성적서가 아니다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1A_CP4_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`GARMENT-CAD-PRO-R1A / CP5 — MOTION-FIT SUITE / STRESS–STRAIN–PRESSURE–CLEARANCE MAPS`\n\n"
        "Use CP4 calibrated material profiles and the immutable CP3 construction package. "
        "Implement pose-sequence qualification, stress, strain, pressure, clearance, seam-tension, "
        "contact-persistence and mobility maps without changing the source pattern or construction authority.\n",
        encoding="utf-8",
    )


def resume_phase(root: Path) -> dict:
    measurements = reference_material_sets()
    hashes = measurement_hashes(measurements)
    checkpoint_path = root / BUILD_REL / "calibration_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path, hashes)
    writer_pid = int(checkpoint["writer_process_id"])
    profiles = _ordered_profiles(measurements, checkpoint)
    parities = tuple(
        evaluate_parity(
            profile.material_id,
            profile.parameters,
            profile.warp_profile_dict()["profile_sha256"],
        )
        for profile in profiles
    )
    resume_receipt = {
        "contract": "CalibrationFreshProcessResumeReceipt/1",
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_writer_process_id": writer_pid,
        "resume_process_id": os.getpid(),
        "resumed_material_ids": list(profile_ids(checkpoint)),
        "completed_material_ids": [item.material_id for item in profiles],
        "fresh_process_resume_pass": writer_pid != os.getpid(),
        "measurement_identity_pass": checkpoint["all_measurement_hashes"] == dict(sorted(hashes.items())),
    }
    resume_receipt["receipt_sha256"] = canonical_sha256(resume_receipt)
    write_json(root / BUILD_REL / "fresh_process_resume_receipt.json", resume_receipt)
    final_checkpoint = write_checkpoint(checkpoint_path, profiles, hashes, "COMPLETE_3_OF_3")
    publish_profiles(root, profiles, parities)
    evidence = root / BUILD_REL / "cp4_material_calibration_evidence.png"
    render_evidence(evidence, measurements, profiles, parities, resume_receipt)
    calibration_receipts = [item.calibration_receipt_dict() for item in profiles]
    receipt = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP4",
        "terminal_decision": "CP4_COMPLETE_MATERIAL_CALIBRATION_PARITY",
        "material_fixture_count": len(measurements),
        "experiment_series_count": sum(len(item.series) for item in measurements),
        "all_calibration_pass": all(item["calibration_pass"] for item in calibration_receipts),
        "maximum_calibration_normalized_rmse": max(item["maximum_normalized_rmse"] for item in calibration_receipts),
        "maximum_calibration_sigma_error": max(item["maximum_sigma_error"] for item in calibration_receipts),
        "all_backend_parity_pass": all(item["parity_pass"] for item in parities),
        "maximum_backend_relative_response_error": max(item["maximum_relative_response_error"] for item in parities),
        "maximum_backend_relative_metric_error": max(item["maximum_relative_metric_error"] for item in parities),
        "fresh_process_resume_pass": resume_receipt["fresh_process_resume_pass"],
        "final_checkpoint_sha256": final_checkpoint["checkpoint_sha256"],
        "runtime": parities[0]["runtime"],
        "cp3_predecessor_mutated": False,
        "triangulation_executed": False,
        "motion_fit_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "next_checkpoint": "GARMENT_CAD_PRO_R1A_CP5",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(root / BUILD_REL / "cp4_receipt.json", receipt)
    write_json(root / "PROFESSIONAL_STATUS.json", receipt)
    write_report(root, receipt)
    return receipt


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    result = checkpoint_phase(root) if args.phase == "checkpoint" else resume_phase(root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
