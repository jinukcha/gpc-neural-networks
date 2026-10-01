from __future__ import annotations

import json
from pathlib import Path

from wuxia_garment_oss.materials.calibration.checkpoint import (
    load_checkpoint,
    profile_from_record,
    write_checkpoint,
)
from wuxia_garment_oss.materials.calibration.fit import calibrate_material
from wuxia_garment_oss.materials.measurement.fixtures import reference_material_sets


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/material_calibration_cp4"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_three_measurement_sets_cover_required_experiments() -> None:
    measurements = reference_material_sets()
    assert len(measurements) == 3
    assert all(len(item.series) == 10 for item in measurements)
    assert all(item.source_kind == "PROJECT_REFERENCE_LAB_SERIES" for item in measurements)
    assert all(item.certification_status == "NOT_EXTERNAL_LAB_CERTIFIED" for item in measurements)


def test_calibration_fits_each_reference_fixture() -> None:
    profiles = tuple(calibrate_material(item) for item in reference_material_sets())
    receipts = [item.calibration_receipt_dict() for item in profiles]
    assert all(item["calibration_pass"] for item in receipts)
    assert all(item["experiment_count"] == 10 for item in receipts)
    assert all(item["parameter_count"] >= 12 for item in receipts)


def test_checkpoint_round_trip_preserves_measurement_identity(tmp_path: Path) -> None:
    measurements = reference_material_sets()
    hashes = {item.material_id: item.to_dict()["measurement_set_sha256"] for item in measurements}
    profiles = tuple(calibrate_material(item) for item in measurements[:2])
    path = tmp_path / "checkpoint.json"
    written = write_checkpoint(path, profiles, hashes, "PARTIAL_2_OF_3")
    reopened = load_checkpoint(path, hashes)
    assert reopened["checkpoint_sha256"] == written["checkpoint_sha256"]
    assert [profile_from_record(item).material_id for item in reopened["profiles"]] == [
        item.material_id for item in profiles
    ]


def test_published_profiles_and_backend_parity_pass() -> None:
    directories = sorted(item for item in (BUILD / "profiles").iterdir() if item.is_dir())
    assert len(directories) == 3
    for directory in directories:
        calibration = _load(directory / "calibration_receipt.json")
        parity = _load(directory / "backend_parity_receipt.json")
        sizing = _load(directory / "material_sizing_profile.json")
        assert calibration["calibration_pass"] is True
        assert parity["parity_pass"] is True
        assert parity["runtime"]["version"] == "1.17.0"
        assert parity["runtime"]["device"] == "cpu"
        assert sizing["recommended_meshing_edge_m"] > 0.0


def test_fresh_process_resume_and_final_checkpoint() -> None:
    receipt = _load(BUILD / "fresh_process_resume_receipt.json")
    checkpoint = _load(BUILD / "calibration_checkpoint.json")
    assert receipt["fresh_process_resume_pass"] is True
    assert receipt["measurement_identity_pass"] is True
    assert checkpoint["phase"] == "COMPLETE_3_OF_3"
    assert len(checkpoint["completed_material_ids"]) == 3


def test_cp4_does_not_execute_pattern_or_motion_products() -> None:
    status = _load(ROOT / "PROFESSIONAL_STATUS.json")
    assert status["triangulation_executed"] is False
    assert status["motion_fit_executed"] is False
    assert status["mesh_scaling"] == "FORBIDDEN"
    assert status["cp3_predecessor_mutated"] is False
