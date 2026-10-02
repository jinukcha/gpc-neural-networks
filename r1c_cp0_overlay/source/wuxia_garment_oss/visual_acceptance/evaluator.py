"""Evaluate visual observations without mutating pattern or product authority."""
from __future__ import annotations

from wuxia_garment_oss.pattern_components.model import canonical_sha256

from .model import VisualAcceptanceProfile


def _passes(comparator: str, actual, threshold) -> bool:
    if comparator == "EQ":
        return actual == threshold
    if comparator == "LE":
        return float(actual) <= float(threshold)
    if comparator == "GE":
        return float(actual) >= float(threshold)
    if comparator == "TRUE":
        return bool(actual) is True
    raise ValueError(comparator)


def evaluate_visual_review(
    profile: VisualAcceptanceProfile,
    garment_family: str,
    technical_pass: bool,
    observations: dict[str, float | bool],
    evidence_refs: dict[str, list[str]],
) -> dict:
    profile.validate()
    gate_results = []
    missing_observations = []
    for gate in profile.gates:
        if gate.gate_id not in observations:
            missing_observations.append(gate.gate_id)
            continue
        actual = observations[gate.gate_id]
        passed = _passes(gate.comparator, actual, gate.threshold)
        gate_results.append({
            "gate_id": gate.gate_id,
            "severity": gate.severity,
            "actual": actual,
            "threshold": gate.threshold,
            "comparator": gate.comparator,
            "passed": passed,
            "repair_disposition": gate.repair_disposition,
            "evidence_refs": evidence_refs.get(gate.gate_id, []),
        })
    failed = [item for item in gate_results if not item["passed"] and item["severity"] != "INFORMATIONAL"]
    required_views = [
        item.view_id
        for item in profile.required_views
        if "ALL" in item.applicable_families or garment_family in item.applicable_families
    ]
    supplied_views = sorted({ref for refs in evidence_refs.values() for ref in refs})
    missing_views = sorted(set(required_views) - set(supplied_views))
    visual_pass = not failed and not missing_observations and not missing_views
    product_acceptance = technical_pass and visual_pass
    payload = {
        "contract": "VisualReviewReceipt/1",
        "profile_id": profile.profile_id,
        "garment_family": garment_family,
        "technical_pass": technical_pass,
        "visual_review": "PASS" if visual_pass else "FAIL",
        "product_acceptance": product_acceptance,
        "required_view_count": len(required_views),
        "supplied_view_count": len(set(supplied_views)),
        "missing_views": missing_views,
        "missing_observations": sorted(missing_observations),
        "gate_results": gate_results,
        "failed_gate_ids": [item["gate_id"] for item in failed],
        "repair_dispositions": sorted({item["repair_disposition"] for item in failed}),
        "pattern_mutated": False,
        "geometry_mutated": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
