#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--work", required=True)
a = p.parse_args()

tree = Path(a.tree).resolve()
work = Path(a.work).resolve()
fort = tree / "RC_K0/child_designs/fortification"
cp0 = fort / "r0a_cp0"
cp1 = fort / "r0a_cp1"
cp1_tower = fort / "r0c_cp1"
sys.path[:0] = [str(cp1 / "src"), str(cp1_tower / "src")]

from rcf_fortification_tower import TowerFamilyProducer, TowerFailureCode


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


outputs = cp1_tower / "outputs"
failures = cp1_tower / "failures"
reports = cp1_tower / "reports"
for path in (outputs, failures, reports):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

producer = TowerFamilyProducer(cp0)
fixtures: dict[str, dict[str, Any]] = {}
for fixture_path in sorted((cp1_tower / "fixtures").glob("tower_*.fixture.json")):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixtures[fixture["family"]] = fixture
if set(fixtures) != {"ROUND", "SQUARE", "POLYGONAL"}:
    raise SystemExit(f"fixture families mismatch: {sorted(fixtures)}")

clean_rows = []
positive_summary = []
for family in ("ROUND", "SQUARE", "POLYGONAL"):
    fixture = fixtures[family]
    run_a = outputs / "runs" / family.lower() / "A"
    run_b = outputs / "runs" / family.lower() / "B"
    result_a = producer.execute(fixture, run_a, failures, failure_case=f"positive_{family}_A")
    result_b = producer.execute(fixture, run_b, failures, failure_case=f"positive_{family}_B")
    if result_a.get("status") != "SUCCEEDED" or result_b.get("status") != "SUCCEEDED":
        raise SystemExit(f"positive {family} failed: {result_a} {result_b}")
    hashes_a = file_hashes(run_a)
    hashes_b = file_hashes(run_b)
    row = {
        "family": family,
        "status": "PASS" if hashes_a == hashes_b else "FAIL",
        "files_compared": len(hashes_a),
        "byte_identical": hashes_a == hashes_b,
        "missing_in_b": sorted(set(hashes_a) - set(hashes_b)),
        "extra_in_b": sorted(set(hashes_b) - set(hashes_a)),
        "different": sorted(key for key in set(hashes_a) & set(hashes_b) if hashes_a[key] != hashes_b[key]),
    }
    clean_rows.append(row)
    if row["status"] != "PASS":
        raise SystemExit(f"clean replay mismatch: {row}")
    reference = outputs / "reference" / family.lower()
    shutil.copytree(run_a, reference)
    positive_summary.append({
        "family": family,
        "tower_id": result_a["tower_id"],
        "part_count": result_a["part_count"],
        "socket_count": result_a["socket_count"],
        "stored_copy_count": result_a["stored_copy_count"],
        "vertex_count": result_a["vertex_count"],
        "triangle_count": result_a["triangle_count"],
        "volume_m3": result_a["volume_m3"],
        "bounds_m": result_a["bounds_m"],
        "body_outside_projection_m": result_a["body_outside_projection_m"],
        "reference_tree_digest": "sha256:" + hashlib.sha256("".join(f"{k}\0{v}" for k, v in sorted(hashes_a.items())).encode("utf-8")).hexdigest(),
    })

(reports / "clean_replay.json").write_text(pretty({
    "schema": "royal-capital.fortification.tower-clean-replay/1",
    "status": "PASS",
    "families": clean_rows,
    "family_count": len(clean_rows),
    "all_byte_identical": all(row["byte_identical"] for row in clean_rows),
}), encoding="utf-8")

# Positive run directories are redundant after byte-identical reference publication.
shutil.rmtree(outputs / "runs")

negative_cases: list[tuple[str, str, Any]] = []

def mutate(base_family: str, function) -> dict[str, Any]:
    value = copy.deepcopy(fixtures[base_family])
    function(value)
    return value

negative_cases.extend([
    ("unsupported_family", str(TowerFailureCode.FAMILY_UNSUPPORTED), mutate("ROUND", lambda f: f.__setitem__("family", "HEX_MAGIC"))),
    ("width_below_domain", str(TowerFailureCode.DIMENSION_OUT_OF_DOMAIN), mutate("SQUARE", lambda f: f.__setitem__("outer_width_m", 7.0))),
    ("width_above_domain", str(TowerFailureCode.DIMENSION_OUT_OF_DOMAIN), mutate("SQUARE", lambda f: f.__setitem__("outer_width_m", 29.0))),
    ("height_above_domain", str(TowerFailureCode.DIMENSION_OUT_OF_DOMAIN), mutate("POLYGONAL", lambda f: f.__setitem__("height_m", 19.0))),
    ("foundation_depth_above_domain", str(TowerFailureCode.DIMENSION_OUT_OF_DOMAIN), mutate("ROUND", lambda f: f.__setitem__("foundation_depth_m", 9.0))),
    ("round_side_count", str(TowerFailureCode.SIDE_COUNT_OUT_OF_DOMAIN), mutate("ROUND", lambda f: f.__setitem__("side_count", 16))),
    ("square_side_count", str(TowerFailureCode.SIDE_COUNT_OUT_OF_DOMAIN), mutate("SQUARE", lambda f: f.__setitem__("side_count", 5))),
    ("polygon_side_count", str(TowerFailureCode.SIDE_COUNT_OUT_OF_DOMAIN), mutate("POLYGONAL", lambda f: f.__setitem__("side_count", 5))),
    ("body_projection_exceeded", str(TowerFailureCode.BODY_PROJECTION_OUT_OF_DOMAIN), mutate("ROUND", lambda f: f.__setitem__("center_m", [0.0, 0.0, -8.0]))),
    ("wall_attachment_missing", str(TowerFailureCode.ATTACHMENT_INTERSECTION_MISSING), mutate("SQUARE", lambda f: f.__setitem__("center_m", [0.0, 0.0, 10.0]))),
    ("runtime_mismatch", str(TowerFailureCode.RUNTIME_MISMATCH), mutate("ROUND", lambda f: f["runtime"].__setitem__("ocp_version", "0.0.0"))),
    ("vertex_budget", str(TowerFailureCode.GEOMETRY_BUDGET_EXCEEDED), mutate("SQUARE", lambda f: f["budget"].__setitem__("max_vertices", 1))),
    ("part_budget", str(TowerFailureCode.GEOMETRY_BUDGET_EXCEEDED), mutate("POLYGONAL", lambda f: f["budget"].__setitem__("max_parts", 2))),
])

negative_rows = []
negative_targets = work / "negative-targets"
negative_targets.mkdir(parents=True, exist_ok=True)
for case_name, expected_code, fixture in negative_cases:
    target = negative_targets / case_name
    result = producer.execute(fixture, target, failures, failure_case=case_name)
    actual_code = result.get("failure", {}).get("code")
    row = {
        "case": case_name,
        "status": result.get("status"),
        "expected_code": expected_code,
        "actual_code": actual_code,
        "target_exists": target.exists(),
        "partial_output_published": result.get("partial_output_published"),
        "failure_receipt_exists": (failures / case_name / "failure-result.json").is_file(),
    }
    row["pass"] = (
        row["status"] == "REJECTED"
        and row["actual_code"] == expected_code
        and not row["target_exists"]
        and row["partial_output_published"] is False
        and row["failure_receipt_exists"]
    )
    negative_rows.append(row)
    if not row["pass"]:
        raise SystemExit(f"negative gate failed: {row}")

(reports / "negative_gates.json").write_text(pretty({
    "schema": "royal-capital.fortification.tower-negative-gates/1",
    "status": "PASS",
    "case_count": len(negative_rows),
    "passed": sum(1 for row in negative_rows if row["pass"]),
    "cases": negative_rows,
    "partial_output_published": False,
}), encoding="utf-8")

qualification = {
    "schema": "royal-capital.fortification.r0c-cp1-qualification/1",
    "status": "PASS",
    "positive_families": positive_summary,
    "clean_replay": {"status": "PASS", "families": len(clean_rows), "all_byte_identical": True},
    "negative_gates": {"status": "PASS", "passed": len(negative_rows), "total": len(negative_rows)},
    "surface_source_coverage": "DEFERRED_TO_R0C_CP4",
    "tower_joins": "DEFERRED_TO_R0C_CP2",
    "battlements": "DEFERRED_TO_R0C_CP3",
    "godot_product": "NOT_STARTED",
}
(reports / "qualification.json").write_text(pretty(qualification), encoding="utf-8")
print(pretty(qualification), end="")
