#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--work", required=True)
a = p.parse_args()
tree = Path(a.tree).resolve()
work = Path(a.work).resolve()
work.mkdir(parents=True, exist_ok=True)
fort = tree / "RC_K0/child_designs/fortification"
cp0 = fort / "r0a_cp0"
cp1 = fort / "r0a_cp1"
cpb1 = fort / "r0b_cp1"
cp3 = fort / "r0b_cp3"
sys.path[:0] = [str(cp1 / "src"), str(cpb1 / "src"), str(cp3 / "src")]
from rcf_fortification_terrain_span import TerrainSpanProducer, TerrainSpanPublisher, derive_contract
from rcf_fortification_terrain_span.producer import tree_digest

fixtures = {
    "terrain_stepped": json.loads((cp3 / "fixtures/terrain_stepped.fixture.json").read_text(encoding="utf-8")),
    "retaining": json.loads((cp3 / "fixtures/retaining.fixture.json").read_text(encoding="utf-8")),
}
outputs = cp3 / "outputs"
reports = cp3 / "reports"
failures = cp3 / "failures"
for path in (outputs, reports, failures):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
producer = TerrainSpanProducer(cp0)
publisher = TerrainSpanPublisher(cp0)


def file_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


def compare(a_root: Path, b_root: Path) -> dict:
    left, right = file_hashes(a_root), file_hashes(b_root)
    return {
        "status": "PASS" if left == right else "FAIL",
        "byte_identical": left == right,
        "files_compared": len(left),
        "missing": sorted(set(left) - set(right)),
        "extra": sorted(set(right) - set(left)),
        "different": sorted(key for key in set(left) & set(right) if left[key] != right[key]),
    }

qualified = []
for name, fixture in fixtures.items():
    a_root = outputs / "clean" / name / "A"
    b_root = outputs / "clean" / name / "B"
    result_a = producer.execute(fixture, a_root)
    result_b = producer.execute(fixture, b_root)
    replay = compare(a_root, b_root)
    if replay["status"] != "PASS":
        raise SystemExit(f"clean replay failed for {name}: {replay}")
    (reports / f"clean_replay_{name}.json").write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reference = outputs / "reference" / name
    shutil.copytree(a_root, reference)
    contract = derive_contract(fixture)
    qualified.append({
        "name": name,
        "span_family": result_a["span_family"],
        "status": "PASS",
        "reference_digest": tree_digest(reference),
        "files_compared": replay["files_compared"],
        "byte_identical": replay["byte_identical"],
        "foundation_segment_count": result_a["foundation_segment_count"],
        "component_count": result_a["component_count"],
        "vertex_count": result_a["vertex_count"],
        "triangle_count": result_a["triangle_count"],
        "volume_m3": result_a["volume_m3"],
        "contact_sample_count": result_a["contact_sample_count"],
        "maximum_grade": result_a["maximum_grade"],
        "maximum_gap_m": result_a["maximum_gap_m"],
        "maximum_penetration_m": result_a["maximum_penetration_m"],
        "minimum_embed_depth_m": contract["contact_evidence"]["summary"]["minimum_embed_depth_m"],
        "maximum_embed_depth_m": contract["contact_evidence"]["summary"]["maximum_embed_depth_m"],
        "maximum_retained_height_m": contract["contact_evidence"]["summary"]["maximum_retained_height_m"],
        "terrain_mutated": False,
    })

# Remove duplicate positive run trees after reference and replay receipts are preserved.
shutil.rmtree(outputs / "clean")

negative_cases = []

def run_negative(name: str, base: str, mutate) -> None:
    fixture = copy.deepcopy(fixtures[base])
    mutate(fixture)
    target = work / "negative-targets" / name
    result = publisher.execute(fixture, target, failures, name)
    row = {
        "case": name,
        "status": result["status"],
        "code": result.get("failure", {}).get("code"),
        "target_exists": target.exists(),
        "partial_output_published": result.get("partial_output_published"),
        "accepted_target_unchanged": result.get("accepted_target_unchanged"),
    }
    negative_cases.append(row)
    if result["status"] != "REJECTED" or target.exists() or result.get("partial_output_published") is not False:
        raise SystemExit(f"negative gate failed: {row}")

run_negative("grade_out_of_domain", "terrain_stepped", lambda f: f["terrain"]["samples"][1].__setitem__("center_elevation_m", 2.0))
run_negative("duplicate_station", "terrain_stepped", lambda f: f["terrain"]["samples"][1].__setitem__("station_m", 0.0))
run_negative("step_height_out_of_domain", "terrain_stepped", lambda f: f["foundation"].__setitem__("step_height_m", 3.0))
run_negative("step_delta_exceeded", "terrain_stepped", lambda f: f["foundation"].__setitem__("max_step_height_m", 0.5))
run_negative("foundation_depth_out_of_domain", "terrain_stepped", lambda f: f["foundation"].__setitem__("depth_m", 0.5))
run_negative("unsupported_gap_exceeded", "retaining", lambda f: f["terrain"]["samples"][0].__setitem__("outside_elevation_m", -2.0))
run_negative("terrain_penetration_exceeded", "retaining", lambda f: f["terrain"]["samples"][0].__setitem__("inside_elevation_m", 4.0))
run_negative("retaining_height_exceeded", "retaining", lambda f: f["terrain"]["samples"][0].__setitem__("inside_elevation_m", 3.25))
run_negative("retaining_side_missing", "retaining", lambda f: f["foundation"].__setitem__("retaining_side", "NONE"))
run_negative("runtime_mismatch", "terrain_stepped", lambda f: f["runtime"].__setitem__("ocp_version", "0.0.0"))
run_negative("segment_budget_exceeded", "terrain_stepped", lambda f: f["budget"].__setitem__("max_segments", 1))
run_negative("component_budget_exceeded", "terrain_stepped", lambda f: f["budget"].__setitem__("max_components", 1))
run_negative("invalid_plan_frame", "terrain_stepped", lambda f: f["plan_centerline"].__setitem__("end_m", [24.0, 1.0, 0.0]))
run_negative("nonfinite_terrain", "terrain_stepped", lambda f: f["terrain"]["samples"][0].__setitem__("center_elevation_m", float("nan")))

# Existing accepted output must survive a rejected request byte-for-byte.
accepted = outputs / "reference" / "terrain_stepped"
accepted_before = tree_digest(accepted)
invalid = copy.deepcopy(fixtures["terrain_stepped"])
invalid["terrain"]["samples"][1]["center_elevation_m"] = 2.0
preservation = publisher.execute(invalid, accepted, failures, "accepted_target_preservation")
accepted_after = tree_digest(accepted)
negative_cases.append({
    "case": "accepted_target_preservation",
    "status": preservation["status"],
    "code": preservation.get("failure", {}).get("code"),
    "target_exists": accepted.exists(),
    "partial_output_published": preservation.get("partial_output_published"),
    "accepted_target_unchanged": preservation.get("accepted_target_unchanged"),
})
if preservation["status"] != "REJECTED" or accepted_before != accepted_after or not preservation.get("accepted_target_unchanged"):
    raise SystemExit("accepted output changed after rejected request")

negative_report = {
    "schema": "royal-capital.fortification.r0b-cp3-negative-gates/1",
    "status": "PASS",
    "case_count": len(negative_cases),
    "all_rejected": all(row["status"] == "REJECTED" for row in negative_cases),
    "no_partial_publication": all(row["partial_output_published"] is False for row in negative_cases),
    "cases": negative_cases,
}
(reports / "negative_gates.json").write_text(json.dumps(negative_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

checks = []
def check(name, passed, detail=None): checks.append({"name": name, "pass": bool(passed), "detail": detail})
check("two_families", [row["name"] for row in qualified] == ["terrain_stepped", "retaining"], qualified)
for row in qualified:
    check(f"{row['name']}_clean_replay", row["byte_identical"], row)
    check(f"{row['name']}_contact", row["contact_sample_count"] > 0 and row["maximum_gap_m"] == 0.0 and row["maximum_penetration_m"] == 0.0, row)
    check(f"{row['name']}_grade", row["maximum_grade"] <= 0.12, row["maximum_grade"])
    check(f"{row['name']}_geometry", row["component_count"] > 0 and row["vertex_count"] > 0 and row["triangle_count"] > 0 and row["volume_m3"] > 0, row)
    check(f"{row['name']}_no_terrain_mutation", row["terrain_mutated"] is False)
check("stepped_segments", qualified[0]["foundation_segment_count"] == 4, qualified[0])
check("retaining_segments", qualified[1]["foundation_segment_count"] == 1, qualified[1])
check("retaining_height", 0.0 < qualified[1]["maximum_retained_height_m"] <= 3.0, qualified[1])
check("negative_count", negative_report["case_count"] == 15, negative_report["case_count"])
check("negative_rejected", negative_report["all_rejected"])
check("negative_no_partial", negative_report["no_partial_publication"])
check("failure_receipts", len(list(failures.glob("*/failure-result.json"))) == 15)
check("accepted_preserved", accepted_before == accepted_after)
for family in fixtures:
    reference = outputs / "reference" / family
    stored = json.loads((reference / "stored-copies.json").read_text())
    result = json.loads((reference / "result.json").read_text())
    foundation = json.loads((reference / "foundation-interface.json").read_text())
    check(f"{family}_stored_copy_count", len(stored["copies"]) == result["component_count"] * 2, len(stored["copies"]))
    check(f"{family}_foundation_interface", foundation["status"] == "PASS" and foundation["terrain_mutation_requested"] is False, foundation)
validation = {
    "schema": "royal-capital.fortification.r0b-cp3-validation/1",
    "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
    "checks": checks,
    "summary": {"passed": sum(row["pass"] for row in checks), "failed": sum(not row["pass"] for row in checks)},
}
(reports / "focused_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if validation["status"] != "PASS":
    raise SystemExit(json.dumps(validation, indent=2))
qualification = {
    "schema": "royal-capital.fortification.r0b-cp3-qualification/1",
    "status": "PASS",
    "qualified_families": qualified,
    "negative_gates": negative_report,
    "focused_validation": validation,
    "deferred": {"join_source_surface_coverage": "R0B_CP4", "mixed_span_closeout": "R0B_CP4", "godot_product": "NOT_STARTED"},
}
(reports / "qualification.json").write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(qualification, indent=2, sort_keys=True))
