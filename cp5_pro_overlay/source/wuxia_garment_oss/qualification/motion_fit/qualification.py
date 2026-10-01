"""Pose-level and suite-level motion-fit acceptance."""
from __future__ import annotations

from ...pattern_cad.document.model import canonical_sha256
from .poses import MotionFitSuite, MotionPose


def qualify_pose(
    pose: MotionPose,
    material_id: str,
    metrics: dict[str, float],
    seam_pair_count: int,
) -> dict:
    gates = {
        "finite_maps": metrics["non_finite_map_count"] == 0.0,
        "body_clearance": metrics["clearance_min_m"] >= -1.0e-5,
        "strain_p99": metrics["strain_p99_ratio"] <= pose.strain_p99_limit,
        "pressure_p99": metrics["pressure_p99_kpa"] <= pose.pressure_p99_kpa_limit,
        "seam_tension_p95": metrics["seam_tension_p95_n_m"] <= pose.seam_tension_p95_n_m_limit,
        "contact_persistence": metrics["contact_persistence_p99"] <= 0.90,
        "mobility_restriction": metrics["mobility_restriction_p95"] <= pose.mobility_restriction_p95_limit,
        "convergence": metrics["tail_peak_displacement_m"] <= 0.006,
        "seam_authority_present": seam_pair_count > 0,
    }
    payload = {
        "contract": "PoseFitQualificationReceipt/1",
        "pose_id": pose.pose_id,
        "material_id": material_id,
        "metrics": metrics,
        "limits": {
            "strain_p99_ratio": pose.strain_p99_limit,
            "pressure_p99_kpa": pose.pressure_p99_kpa_limit,
            "seam_tension_p95_n_m": pose.seam_tension_p95_n_m_limit,
            "mobility_restriction_p95": pose.mobility_restriction_p95_limit,
            "tail_peak_displacement_m": 0.006,
        },
        "gates": gates,
        "pose_pass": all(gates.values()),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def qualify_suite(
    suite: MotionFitSuite,
    pose_receipts: tuple[dict, ...],
    material_ids: tuple[str, ...],
    source_authority: dict,
) -> dict:
    expected = {(material_id, pose.pose_id) for material_id in material_ids for pose in suite.poses}
    actual = {(item["material_id"], item["pose_id"]) for item in pose_receipts}
    failed = sorted(
        f"{item['material_id']}::{item['pose_id']}"
        for item in pose_receipts
        if not item["pose_pass"]
    )
    payload = {
        "contract": "FitQualificationReceipt/2",
        "suite": suite.to_dict(),
        "material_ids": list(material_ids),
        "scenario_count": len(pose_receipts),
        "expected_scenario_count": len(expected),
        "scenario_identity_complete": actual == expected,
        "pose_pass_count": sum(bool(item["pose_pass"]) for item in pose_receipts),
        "failed_scenarios": failed,
        "source_authority": source_authority,
        "reference_mesh_scope": "IMMUTABLE_CP2B_FITTED_TUNIC_REFERENCE",
        "construction_semantics_scope": "CP3_READ_ONLY_NOT_FEATURE_TRIANGULATED",
        "motion_fit_pass": actual == expected and not failed,
        "product_acceptance": False,
        "product_acceptance_reason": "CP5 qualifies the immutable reference mesh; CP3 feature-complete construction topology is not triangulated in this checkpoint.",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
