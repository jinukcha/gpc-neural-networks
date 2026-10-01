from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.qualification.motion_fit.mesh import load_motion_mesh
from wuxia_garment_oss.qualification.motion_fit.poses import professional_tunic_suite


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/motion_fit_cp5"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_professional_motion_suite_has_required_ten_poses() -> None:
    suite = professional_tunic_suite()
    suite.validate()
    ids = {item.pose_id for item in suite.poses}
    assert len(ids) == 10
    assert {
        "NEUTRAL_A", "ARMS_FORWARD", "ARMS_OVERHEAD", "CROSS_BODY_REACH",
        "DEEP_ELBOW_BEND", "TORSO_TWIST", "FORWARD_BEND", "SEATED",
        "SQUAT", "WALK_STRIDE",
    } == ids


def test_immutable_reference_mesh_has_motion_authority() -> None:
    mesh = load_motion_mesh(ROOT)
    assert mesh.vertex_count > 1000
    assert len(mesh.triangles) > 1000
    assert len(mesh.edges) > 1000
    assert len(mesh.seam_pairs) > 0
    assert np.isfinite(mesh.positions).all()


def test_thirty_motion_fit_scenarios_are_published() -> None:
    archives = sorted((BUILD / "scenarios").glob("*/*.npz"))
    receipts = sorted((BUILD / "pose_receipts").glob("*/*.json"))
    assert len(archives) == 30
    assert len(receipts) == 30
    assert all(_load(path)["contract"] == "PoseFitQualificationReceipt/1" for path in receipts)


def test_canonical_maps_cover_professional_metrics() -> None:
    archive = next((BUILD / "scenarios/WOOL_TWILL_MEDIUM_REFERENCE").glob("*.npz"))
    with np.load(archive, allow_pickle=False) as data:
        expected = {
            "stress_n_m", "strain_ratio", "pressure_pa", "clearance_m",
            "seam_tension_n_m", "contact_persistence", "mobility_restriction",
        }
        assert expected.issubset(data.files)
        assert all(np.isfinite(data[name]).all() for name in expected)
        assert np.max(data["strain_ratio"]) > 0.0


def test_fit_qualification_is_complete_and_non_product() -> None:
    receipt = _load(BUILD / "fit_qualification_receipt.json")
    status = _load(ROOT / "PROFESSIONAL_STATUS.json")
    assert receipt["contract"] == "FitQualificationReceipt/2"
    assert receipt["scenario_count"] == 30
    assert receipt["scenario_identity_complete"] is True
    assert receipt["product_acceptance"] is False
    assert status["product_acceptance"] is False
    assert status["feature_complete_cp3_topology_triangulated"] is False


def test_cp5_preserves_all_predecessor_authorities() -> None:
    status = _load(ROOT / "PROFESSIONAL_STATUS.json")
    assert status["cp4_predecessor_mutated"] is False
    assert status["pattern_or_construction_mutated"] is False
    assert status["mesh_scaling"] == "FORBIDDEN"
