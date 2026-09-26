#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--tree", required=True)
parser.add_argument("--report", required=True)
args = parser.parse_args()

tree = Path(args.tree).resolve()
fort = tree / "RC_K0/child_designs/fortification"
cp3 = fort / "r0b_cp3"
checks: list[dict] = []


def check(name: str, value, detail=None):
    checks.append({"name": name, "pass": bool(value), "detail": detail})


def read(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))


def verify_ref(root: Path, ref: dict) -> bool:
    path = root / ref["path"]
    return path.is_file() and path.stat().st_size == ref["bytes"] and "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]

qualification = read(cp3 / "reports", "qualification.json")
check("qualification", qualification["status"] == "PASS", qualification)
check("families", set(qualification["families"]) == {"stepped", "retaining"}, sorted(qualification["families"]))
check("clean_replay", all(row["byte_identical"] for row in qualification["families"].values()), qualification["families"])
check("negative_gates", qualification["negative_count"] == 10, qualification["negative"])
check("negative_typed", all(row["status"] == "PASS_EXPECTED_REJECTION" and row["code"] for row in qualification["negative"]))

expected = {
    "stepped": {
        "family": "TERRAIN_STEPPED",
        "segments": 3,
        "units": 15,
        "sockets": 14,
        "stored": 30,
        "interface_count": 3,
        "transition_count": 2,
        "contact_area": 240.0,
        "sample_count": 15,
        "max_embedment": 2.0,
        "retained_height": 0.0,
        "overall_grade": 2.0 / 30.0,
        "max_step": 1.0,
    },
    "retaining": {
        "family": "RETAINING",
        "segments": 1,
        "units": 5,
        "sockets": 8,
        "stored": 10,
        "interface_count": 3,
        "transition_count": 0,
        "contact_area": 264.0,
        "sample_count": 27,
        "max_embedment": 5.0,
        "retained_height": 3.0,
        "overall_grade": 0.0,
        "max_step": 0.0,
    },
}

for name, contract in expected.items():
    root = cp3 / "outputs" / name / "reference"
    result = read(root, "result.json")
    plan = read(root, "terrain-span-plan.json")
    profile = read(root, "canonical-terrain-profile.json")
    interface = read(root, "foundation-interface.json")
    contact = read(root, "contact-evidence.json")
    grade = read(root, "grade-evidence.json")
    sockets = read(root, "interface-sockets.json")
    parts = read(root, "semantic-parts.json")
    units = read(root, "construction-units.json")
    mesh = read(root, "neutral-mesh.json")
    stored = read(root, "stored-copies.json")
    receipt = read(root, "cad-provider-receipt.json")

    check(f"{name}_status", result["status"] == "SUCCEEDED", result)
    check(f"{name}_family", result["family"] == contract["family"], result["family"])
    check(f"{name}_segments", result["segment_count"] == contract["segments"], result["segment_count"])
    check(f"{name}_units", result["unit_count"] == contract["units"] == units["unit_count"] == len(units["units"]), result["unit_count"])
    check(f"{name}_solid_count", result["solid_count"] == contract["units"], result["solid_count"])
    check(f"{name}_parts", result["part_count"] == 5 and [row["part_id"] for row in parts["parts"]] == ["foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet"])
    check(f"{name}_sockets", result["socket_count"] == contract["sockets"] == len(sockets["sockets"]), result["socket_count"])
    check(f"{name}_socket_unique", len({row["socket_id"] for row in sockets["sockets"]}) == contract["sockets"])
    check(f"{name}_stored", len(stored["copies"]) == contract["stored"], len(stored["copies"]))
    check(f"{name}_stored_formats", all({row["format"] for row in stored["copies"] if row["unit_id"] == unit["unit_id"]} == {"STEP", "BREP"} for unit in units["units"]))
    check(f"{name}_interface", interface["summary"]["status"] == "PASS" and interface["summary"]["interface_count"] == contract["interface_count"] and interface["summary"]["transition_count"] == contract["transition_count"], interface["summary"])
    check(f"{name}_contact_area", abs(interface["summary"]["total_contact_area_m2"] - contract["contact_area"]) <= 1e-9, interface["summary"]["total_contact_area_m2"])
    check(f"{name}_gap", contact["summary"]["status"] == "PASS" and contact["summary"]["sample_count"] == contract["sample_count"] and contact["summary"]["contact_ratio"] == 1.0 and contact["summary"]["maximum_gap_m"] == 0.0, contact["summary"])
    check(f"{name}_embedment", abs(interface["summary"]["maximum_embedment_depth_m"] - contract["max_embedment"]) <= 1e-9, interface["summary"]["maximum_embedment_depth_m"])
    check(f"{name}_grade", grade["status"] == "PASS" and abs(grade["overall_centerline_grade"] - contract["overall_grade"]) <= 1e-9 and abs(grade["maximum_step_height_m"] - contract["max_step"]) <= 1e-9, grade)
    check(f"{name}_retained_height", abs(grade["retained_height_m"] - contract["retained_height"]) <= 1e-9, grade["retained_height_m"])
    check(f"{name}_terrain_digest", profile["profile_digest"] == plan["terrain_profile_digest"] == receipt["terrain_profile_digest"], [profile["profile_digest"], plan["terrain_profile_digest"], receipt["terrain_profile_digest"]])
    check(f"{name}_terrain_not_mutated", not result["terrain_mutation"] and not plan["terrain_mutation"] and not receipt["terrain_mutation"] and not interface["summary"]["terrain_mutation"])
    check(f"{name}_mesh", result["vertex_count"] > 0 and result["triangle_count"] > 0 and result["volume_m3"] > 0 and len(mesh["unit_ranges"]) == contract["units"])
    check(f"{name}_indices", all(0 <= index < len(mesh["vertices_m"]) for triangle in mesh["triangles"] for index in triangle))
    check(f"{name}_unit_volumes", all(row["relative_volume_error"] <= 1e-8 and row["volume_m3"] > 0 for row in units["units"]))
    check(f"{name}_surface_deferred", all(row["surface_coverage"] == "DEFERRED_TO_R0B_CP4" for row in parts["parts"]))
    check(f"{name}_artifact_refs", all(verify_ref(root, ref) for ref in result["artifacts"].values()))

check("stepped_walk_transition_sockets", sum(row["role"] == "STAIR_RAMP_SOCKET" for row in read(cp3 / "outputs/stepped/reference", "interface-sockets.json")["sockets"]) == 4)
retaining_grade = read(cp3 / "outputs/retaining/reference", "grade-evidence.json")
check("retaining_cross_slope", abs(retaining_grade["equivalent_cross_slope_ratio"] - 1.0 / 3.0) <= 1e-9, retaining_grade["equivalent_cross_slope_ratio"])

status_rows = list(csv.DictReader((fort / "data/CP_STATUS.csv").open(newline="")))
check("parent_cp2_complete", any(row["stage"] == "R0B" and row["checkpoint"] == "CP2" and row["stage_completion"] == "R0B_CP2_COMPLETE" for row in status_rows))

bad = [
    path.as_posix() for path in tree.rglob("*")
    if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts or ".godot" in path.parts)
]
check("no_cache", not bad, bad)

parse_errors = []
for path in cp3.rglob("*.json"):
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        parse_errors.append([path.as_posix(), str(exc)])
for path in cp3.rglob("*.csv"):
    try:
        list(csv.reader(path.open(newline="")))
    except Exception as exc:
        parse_errors.append([path.as_posix(), str(exc)])
check("machine_parse", not parse_errors, parse_errors)

report = {
    "schema": "royal-capital.fortification.r0b-cp3-validation/1",
    "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
    "checks": checks,
    "summary": {
        "passed": sum(row["pass"] for row in checks),
        "failed": sum(not row["pass"] for row in checks),
    },
}
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["status"] == "PASS" else 1)
