#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--tree", required=True)
parser.add_argument("--python", required=True)
parser.add_argument("--work", required=True)
args = parser.parse_args()

tree = Path(args.tree).resolve()
work = Path(args.work).resolve()
work.mkdir(parents=True, exist_ok=True)
fort = tree / "RC_K0/child_designs/fortification"
cp0 = fort / "r0a_cp0"
adapter = fort / "r0a_cp1"
path_span = fort / "r0b_cp1"
cp3 = fort / "r0b_cp3"
outputs = cp3 / "outputs"
reports = cp3 / "reports"
reports.mkdir(parents=True, exist_ok=True)
if outputs.exists():
    shutil.rmtree(outputs)
outputs.mkdir(parents=True)

sys.path.insert(0, str(adapter / "src"))
sys.path.insert(0, str(path_span / "src"))
sys.path.insert(0, str(cp3 / "src"))
from rcf_fortification_terrain_span import TerrainContractError, terrain_profile_digest, validate_fixture

fixtures = {
    "stepped": cp3 / "fixtures/terrain_stepped_span.fixture.json",
    "retaining": cp3 / "fixtures/retaining_span.fixture.json",
}

def file_map(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }

summary = {
    "schema": "royal-capital.fortification.r0b-cp3-qualification/1",
    "status": "OPEN",
    "families": {},
    "negative": [],
}
runner = cp3 / "tools/run_r0b_cp3.py"
environment = dict(os.environ)
environment["PYTHONDONTWRITEBYTECODE"] = "1"

for name, fixture_path in fixtures.items():
    family_root = outputs / name
    a = family_root / "A"
    b = family_root / "B"
    reference = family_root / "reference"
    for label, destination in (("A", a), ("B", b)):
        command = [
            args.python,
            str(runner),
            "--cp0-root", str(cp0),
            "--adapter-root", str(adapter),
            "--path-span-root", str(path_span),
            "--cp3-root", str(cp3),
            "--fixture", str(fixture_path),
            "--output", str(destination),
        ]
        with (reports / f"{name}_{label}.log").open("wb") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if process.returncode:
            raise SystemExit(f"{name} {label} failed with {process.returncode}")
    files_a = file_map(a)
    files_b = file_map(b)
    if files_a != files_b:
        different = sorted(path for path in set(files_a) | set(files_b) if files_a.get(path) != files_b.get(path))
        raise SystemExit(f"clean replay mismatch {name}: {different}")
    shutil.copytree(a, reference)
    result = json.loads((reference / "result.json").read_text(encoding="utf-8"))
    interface = json.loads((reference / "foundation-interface.json").read_text(encoding="utf-8"))
    contact = json.loads((reference / "contact-evidence.json").read_text(encoding="utf-8"))
    grade = json.loads((reference / "grade-evidence.json").read_text(encoding="utf-8"))
    stored = json.loads((reference / "stored-copies.json").read_text(encoding="utf-8"))
    summary["families"][name] = {
        "status": "PASS",
        "family": result["family"],
        "segment_count": result["segment_count"],
        "unit_count": result["unit_count"],
        "solid_count": result["solid_count"],
        "socket_count": result["socket_count"],
        "vertex_count": result["vertex_count"],
        "triangle_count": result["triangle_count"],
        "volume_m3": result["volume_m3"],
        "bounds_m": result["bounds_m"],
        "foundation_interface_status": interface["summary"]["status"],
        "maximum_gap_m": contact["summary"]["maximum_gap_m"],
        "contact_ratio": contact["summary"]["contact_ratio"],
        "overall_centerline_grade": grade["overall_centerline_grade"],
        "maximum_step_height_m": grade["maximum_step_height_m"],
        "retained_height_m": grade["retained_height_m"],
        "stored_copy_count": len(stored["copies"]),
        "files_compared": len(files_a),
        "byte_identical": True,
        "reference_digest": "sha256:" + hashlib.sha256(
            b"".join(path.encode("utf-8") + b"\0" + bytes.fromhex(files_a[path]) for path in sorted(files_a))
        ).hexdigest(),
    }
    shutil.rmtree(a)
    shutil.rmtree(b)

stepped = json.loads(fixtures["stepped"].read_text(encoding="utf-8"))
retaining = json.loads(fixtures["retaining"].read_text(encoding="utf-8"))

def refresh_digest(value: dict) -> dict:
    value["terrain_source"]["profile_digest"] = terrain_profile_digest(value)
    return value

negative_cases: list[tuple[str, dict, str]] = []
case = copy.deepcopy(stepped)
case["terrain"]["terraces"][1]["elevation_m"] = 2.0
case["terrain"]["terraces"][2]["elevation_m"] = 4.0
negative_cases.append(("overall_grade_exceeded", refresh_digest(case), "GRADE_OUT_OF_DOMAIN"))
case = copy.deepcopy(stepped)
case["terrain"]["terraces"][1]["elevation_m"] = 3.0
case["terrain"]["terraces"][2]["elevation_m"] = 3.5
negative_cases.append(("step_height_exceeded", refresh_digest(case), "GRADE_OUT_OF_DOMAIN"))
case = copy.deepcopy(stepped)
case["foundation"]["terrain_contact_offset_m"] = 0.05
negative_cases.append(("foundation_gap_exceeded", case, "FOUNDATION_GAP_OUT_OF_DOMAIN"))
case = copy.deepcopy(stepped)
case["terrain"]["terraces"][1]["start_station_m"] = 11.0
negative_cases.append(("terrace_coverage_gap", refresh_digest(case), "INVALID_TERRAIN_PROFILE"))
case = copy.deepcopy(stepped)
case["foundation"]["embedment_depth_m"] = 9.0
case["foundation"]["max_embedment_depth_m"] = 9.0
negative_cases.append(("embedment_out_of_domain", case, "FOUNDATION_EMBEDMENT_OUT_OF_DOMAIN"))
case = copy.deepcopy(retaining)
case["terrain"]["inside_elevation_m"] = 6.0
case["foundation"]["max_embedment_depth_m"] = 7.0
negative_cases.append(("retaining_upper_embedment_exceeded", refresh_digest(case), "FOUNDATION_EMBEDMENT_OUT_OF_DOMAIN"))
case = copy.deepcopy(retaining)
case["terrain"]["outside_toe_width_m"] = 5.1
negative_cases.append(("stale_terrain_digest", case, "STALE_INPUT_REFERENCE"))
case = copy.deepcopy(stepped)
case["runtime"]["ocp_version"] = "0.0.0"
negative_cases.append(("runtime_mismatch", case, "CAD_RUNTIME_VERSION_MISMATCH"))
case = copy.deepcopy(stepped)
case["budget"]["max_units"] = 14
negative_cases.append(("unit_budget_exceeded", case, "GEOMETRY_BUDGET_EXCEEDED"))
case = copy.deepcopy(retaining)
case["terrain"]["inside_elevation_m"] = 7.0
negative_cases.append(("retaining_height_exceeded", refresh_digest(case), "RETAINING_HEIGHT_OUT_OF_DOMAIN"))

for name, value, expected_code in negative_cases:
    try:
        validate_fixture(value)
    except TerrainContractError as exc:
        if exc.code != expected_code:
            raise SystemExit(f"negative case {name} returned {exc.code}, expected {expected_code}: {exc}")
        summary["negative"].append({
            "case": name,
            "status": "PASS_EXPECTED_REJECTION",
            "code": exc.code,
            "message": exc.message,
        })
    except Exception as exc:
        raise SystemExit(f"negative case {name} returned unexpected {type(exc).__name__}: {exc}")
    else:
        raise SystemExit(f"negative fixture accepted: {name}")

summary["negative_count"] = len(summary["negative"])
summary["status"] = "PASS"
(reports / "qualification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

unit_environment = dict(environment)
unit_environment["PYTHONPATH"] = os.pathsep.join([
    str(adapter / "src"),
    str(path_span / "src"),
    str(cp3 / "src"),
])
unit_environment["RCF_R0B_CP3_OUTPUT_ROOT"] = str(outputs)
with (reports / "unittest.log").open("wb") as log:
    process = subprocess.run(
        [args.python, "-m", "unittest", "discover", "-s", str(cp3 / "tests"), "-p", "test_*.py", "-v"],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=unit_environment,
    )
(reports / "unittest.json").write_text(
    json.dumps({"status": "PASS" if process.returncode == 0 else "FAIL", "exit_code": process.returncode}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if process.returncode:
    raise SystemExit(process.returncode)
print(json.dumps(summary, indent=2, sort_keys=True))
