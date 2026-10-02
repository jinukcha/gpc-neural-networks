"""Diagnose incomplete or drifted pattern assemblies against immutable CP2 authority."""
from __future__ import annotations

from .geometry import changed_segments, seam_lengths
from .model import canonical_sha256, make_issue, strongest_disposition


SAFE_GEOMETRY_BUDGET_M = 0.003
GUIDED_GEOMETRY_BUDGET_M = 0.012
SAFE_SEAM_RATIO_DELTA = 0.015
GUIDED_SEAM_RATIO_DELTA = 0.04
_REQUIRED_SEAM_FIELDS = ("seam_allowance_a_m", "seam_allowance_b_m", "turn_of_cloth_m")


def _package_instances(snapshot: dict) -> dict[str, dict]:
    return {
        item["instance_id"]: item
        for item in snapshot["assembled_package"]["component_instances"]
    }


def _package_seams(snapshot: dict) -> dict[str, dict]:
    return {
        item["interface_id"]: item
        for item in snapshot["assembled_package"]["seams"]
    }


def _geometry_disposition(delta: float) -> str:
    if delta <= SAFE_GEOMETRY_BUDGET_M:
        return "SAFE_AUTO"
    if delta <= GUIDED_GEOMETRY_BUDGET_M:
        return "GUIDED"
    return "HOLD"


def _missing_component_issues(canonical: dict, candidate: dict) -> tuple[list[dict], dict[str, str]]:
    expected = _package_instances(canonical)
    observed = _package_instances(candidate)
    issues = []
    dispositions = {}
    for instance_id, owner in expected.items():
        if instance_id in observed:
            continue
        mirror_source = owner.get("mirror_source_instance_id")
        disposition = "SAFE_AUTO" if mirror_source else "HOLD"
        dispositions[instance_id] = disposition
        issues.append(make_issue(
            "MISSING_COMPONENT",
            f"component/{instance_id}",
            disposition,
            "required component instance is absent",
            None,
            owner,
            (f"component/{instance_id}",),
        ))
    return issues, dispositions


def _missing_seam_issues(canonical: dict, candidate: dict, missing_components: dict[str, str]) -> list[dict]:
    expected = _package_seams(canonical)
    observed = _package_seams(candidate)
    issues = []
    for interface_id, seam in expected.items():
        if interface_id in observed:
            continue
        endpoint_instances = {
            seam["endpoint_a"]["component_instance_id"],
            seam["endpoint_b"]["component_instance_id"],
        }
        endpoint_dispositions = [
            missing_components[item] for item in endpoint_instances if item in missing_components
        ]
        disposition = strongest_disposition(endpoint_dispositions) if endpoint_dispositions else "SAFE_AUTO"
        issues.append(make_issue(
            "MISSING_INTERFACE",
            f"seam/{interface_id}",
            disposition,
            "assembly interface is absent",
            None,
            seam,
            (f"seam/{interface_id}",),
        ))
    return issues


def _missing_parameter_issues(canonical: dict, candidate: dict) -> list[dict]:
    expected = _package_seams(canonical)
    observed = _package_seams(candidate)
    issues = []
    for interface_id, seam in expected.items():
        current = observed.get(interface_id)
        if current is None:
            continue
        for field in _REQUIRED_SEAM_FIELDS:
            if field in current:
                continue
            issues.append(make_issue(
                "MISSING_PARAMETER",
                f"seam/{interface_id}/field/{field}",
                "SAFE_AUTO",
                "required seam parameter is absent",
                None,
                seam[field],
                (f"seam/{interface_id}/field/{field}",),
            ))
    return issues


def _missing_notch_issues(canonical: dict, candidate: dict) -> list[dict]:
    issues = []
    for instance_id, expected_geometry in canonical["geometry_by_instance"].items():
        current_geometry = candidate["geometry_by_instance"].get(instance_id)
        if current_geometry is None:
            continue
        expected = {item["notch_id"]: item for item in expected_geometry.get("notches", [])}
        observed = {item["notch_id"]: item for item in current_geometry.get("notches", [])}
        for notch_id, notch in expected.items():
            if notch_id in observed:
                continue
            target = f"geometry/{instance_id}/notch/{notch_id}"
            issues.append(make_issue(
                "MISSING_NOTCH",
                target,
                "SAFE_AUTO",
                "semantic notch is absent",
                None,
                notch,
                (target,),
            ))
    return issues


def _geometry_drift_issues(canonical: dict, candidate: dict) -> list[dict]:
    issues = []
    for instance_id, expected_geometry in canonical["geometry_by_instance"].items():
        current_geometry = candidate["geometry_by_instance"].get(instance_id)
        if current_geometry is None:
            continue
        for change in changed_segments(expected_geometry, current_geometry):
            category = "ENDPOINT_DRIFT" if change["endpoint_changed"] else "TANGENT_DRIFT"
            target = f"geometry/{instance_id}/segment/{change['segment_id']}"
            disposition = _geometry_disposition(change["delta_m"])
            issues.append(make_issue(
                category,
                target,
                disposition,
                "source pattern curve differs from immutable authority",
                change["delta_m"],
                0.0,
                (target,),
                change["delta_m"],
                SAFE_GEOMETRY_BUDGET_M if disposition == "SAFE_AUTO" else GUIDED_GEOMETRY_BUDGET_M,
            ))
    return issues


def _changed_targets_for_seam(canonical: dict, candidate: dict, seam: dict) -> tuple[str, ...]:
    targets = []
    for endpoint in (seam["endpoint_a"], seam["endpoint_b"]):
        instance_id = endpoint["component_instance_id"]
        expected = canonical["geometry_by_instance"].get(instance_id)
        observed = candidate["geometry_by_instance"].get(instance_id)
        if expected is None or observed is None:
            continue
        for change in changed_segments(expected, observed):
            targets.append(f"geometry/{instance_id}/segment/{change['segment_id']}")
    return tuple(sorted(set(targets)))


def _seam_length_issues(canonical: dict, candidate: dict) -> list[dict]:
    issues = []
    candidate_seams = _package_seams(candidate)
    for seam in canonical["assembled_package"]["seams"]:
        if seam["interface_id"] not in candidate_seams:
            continue
        measured = seam_lengths(candidate, seam)
        if measured is None:
            continue
        first_delta = abs(measured[0] - seam["length_a_m"]) / max(seam["length_a_m"], 1.0e-12)
        second_delta = abs(measured[1] - seam["length_b_m"]) / max(seam["length_b_m"], 1.0e-12)
        delta = max(first_delta, second_delta)
        if delta <= 1.0e-8:
            continue
        if delta <= SAFE_SEAM_RATIO_DELTA:
            disposition = "SAFE_AUTO"
            budget = SAFE_SEAM_RATIO_DELTA
        elif delta <= GUIDED_SEAM_RATIO_DELTA:
            disposition = "GUIDED"
            budget = GUIDED_SEAM_RATIO_DELTA
        else:
            disposition = "HOLD"
            budget = GUIDED_SEAM_RATIO_DELTA
        issues.append(make_issue(
            "SEAM_LENGTH_MISMATCH",
            f"seam/{seam['interface_id']}/length",
            disposition,
            "physical boundary length differs from accepted source pattern",
            {"length_a_m": measured[0], "length_b_m": measured[1]},
            {"length_a_m": seam["length_a_m"], "length_b_m": seam["length_b_m"]},
            _changed_targets_for_seam(canonical, candidate, seam),
            delta,
            budget,
        ))
    return issues


def diagnose_completion(canonical: dict, candidate: dict, diagnosis_id: str) -> dict:
    issues, missing_components = _missing_component_issues(canonical, candidate)
    issues.extend(_missing_seam_issues(canonical, candidate, missing_components))
    issues.extend(_missing_parameter_issues(canonical, candidate))
    issues.extend(_missing_notch_issues(canonical, candidate))
    issues.extend(_geometry_drift_issues(canonical, candidate))
    issues.extend(_seam_length_issues(canonical, candidate))
    issues.sort(key=lambda item: (item["owner_path"], item["category"]))
    dispositions = [item["disposition"] for item in issues]
    payload = {
        "contract": "CompletionDiagnosisReport/1",
        "diagnosis_id": diagnosis_id,
        "canonical_state_sha256": canonical["state_sha256"],
        "candidate_state_sha256": candidate["state_sha256"],
        "issue_count": len(issues),
        "issues": issues,
        "disposition_counts": {
            name: sum(value == name for value in dispositions)
            for name in ("SAFE_AUTO", "GUIDED", "HOLD")
        },
        "strongest_disposition": strongest_disposition(dispositions),
        "triangulation_executed": False,
        "simulation_executed": False,
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload
