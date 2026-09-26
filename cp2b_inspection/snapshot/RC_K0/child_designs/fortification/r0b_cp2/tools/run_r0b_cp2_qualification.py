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
cp1 = fort / "r0a_cp1"
cp2 = fort / "r0b_cp2"
reports = cp2 / "reports"
outputs = cp2 / "outputs"
reports.mkdir(parents=True, exist_ok=True)
if outputs.exists():
    shutil.rmtree(outputs)
outputs.mkdir(parents=True)

sys.path[:0] = [str(cp1 / "src"), str(cp2 / "src")]
from rcf_fortification_join import WallJoinProducer
from rcf_fortification_join.model import validate_fixture

producer = WallJoinProducer(cp0)
fixtures = {
    "miter": cp2 / "fixtures/miter_join.fixture.json",
    "bevel": cp2 / "fixtures/bevel_join.fixture.json",
    "transition": cp2 / "fixtures/profile_transition_join.fixture.json",
}


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): digest_file(path) for path in root.rglob("*") if path.is_file()}


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        h.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()

source_span_root = fort / "r0b_cp1/outputs"
source_span_digest_before = tree_digest(source_span_root)
summary = {
    "schema": "royal-capital.fortification.r0b-cp2-qualification/1",
    "status": "OPEN",
    "families": {},
    "negative": [],
    "source_span_digest_before": source_span_digest_before,
}

for family, fixture_path in fixtures.items():
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    family_root = work / family
    a = family_root / "A"
    b = family_root / "B"
    if family_root.exists():
        shutil.rmtree(family_root)
    family_root.mkdir(parents=True)
    result_a = producer.execute(fixture, a)
    result_b = producer.execute(fixture, b)
    files_a, files_b = files(a), files(b)
    if files_a != files_b:
        raise SystemExit(f"clean replay mismatch for {family}: {[key for key in set(files_a)|set(files_b) if files_a.get(key)!=files_b.get(key)]}")
    reference = outputs / family / "reference"
    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(a, reference)
    alignment = json.loads((reference / "socket-alignment.json").read_text(encoding="utf-8"))
    overlap = json.loads((reference / "bounded-overlap.json").read_text(encoding="utf-8"))
    summary["families"][family] = {
        "status": "PASS",
        "family": result_a["family"],
        "turn_angle_deg": result_a["turn_angle_deg"],
        "section_count": result_a["section_count"],
        "solid_count": result_a["solid_count"],
        "socket_count": result_a["socket_count"],
        "socket_alignment_count": result_a["socket_alignment_count"],
        "socket_alignment_status": alignment["status"],
        "bounded_overlap_status": overlap["status"],
        "volume_m3": result_a["volume_m3"],
        "bounds_m": result_a["bounds_m"],
        "vertex_count": result_a["vertex_count"],
        "triangle_count": result_a["triangle_count"],
        "files_compared": len(files_a),
        "byte_identical": True,
        "reference_digest": tree_digest(reference),
    }

base = json.loads(fixtures["miter"].read_text(encoding="utf-8"))
bevel = json.loads(fixtures["bevel"].read_text(encoding="utf-8"))
transition = json.loads(fixtures["transition"].read_text(encoding="utf-8"))
negative_cases = []

case = copy.deepcopy(base)
angle = 30.0
r = __import__("math").radians(angle)
case["outgoing"]["frame"]["tangent"] = [__import__("math").cos(r), 0.0, __import__("math").sin(r)]
case["outgoing"]["frame"]["inside"] = [-__import__("math").sin(r), 0.0, __import__("math").cos(r)]
case["outgoing"]["frame"]["outside"] = [__import__("math").sin(r), 0.0, -__import__("math").cos(r)]
for row in case["outgoing"]["sockets"].values(): row["frame"] = copy.deepcopy(case["outgoing"]["frame"])
negative_cases.append(("miter_turn_out_of_domain", case))

case = copy.deepcopy(bevel)
angle = 10.0
r = __import__("math").radians(angle)
case["outgoing"]["frame"]["tangent"] = [__import__("math").cos(r), 0.0, __import__("math").sin(r)]
case["outgoing"]["frame"]["inside"] = [-__import__("math").sin(r), 0.0, __import__("math").cos(r)]
case["outgoing"]["frame"]["outside"] = [__import__("math").sin(r), 0.0, -__import__("math").cos(r)]
for row in case["outgoing"]["sockets"].values(): row["frame"] = copy.deepcopy(case["outgoing"]["frame"])
negative_cases.append(("bevel_turn_out_of_domain", case))

case = copy.deepcopy(transition); case["outgoing_profile"] = copy.deepcopy(case["incoming_profile"])
negative_cases.append(("transition_requires_distinct_profile", case))
case = copy.deepcopy(base); case["outgoing"]["sockets"]["span"]["position_m"][0] += 0.01
negative_cases.append(("socket_position_mismatch", case))
case = copy.deepcopy(base); case["outgoing"]["sockets"]["span"]["frame"]["inside"] = [0.0, 0.0, -1.0]
negative_cases.append(("socket_frame_mismatch", case))
case = copy.deepcopy(base); case["overlap"]["incoming_m"] = 11.0; case["overlap"]["max_m"] = 10.0
negative_cases.append(("overlap_exceeds_max", case))
case = copy.deepcopy(base); case["budget"]["max_sections"] = 2
negative_cases.append(("section_budget", case))
case = copy.deepcopy(base); case["runtime"]["ocp_version"] = "0.0.0"
negative_cases.append(("runtime_mismatch", case))
case = copy.deepcopy(base); case["outgoing"]["frame"]["tangent"] = [-1.0, 0.0, 0.0]; case["outgoing"]["frame"]["inside"] = [0.0, 0.0, -1.0]; case["outgoing"]["frame"]["outside"] = [0.0, 0.0, 1.0]
for row in case["outgoing"]["sockets"].values(): row["frame"] = copy.deepcopy(case["outgoing"]["frame"])
negative_cases.append(("opposed_tangents", case))

for name, value in negative_cases:
    try:
        validate_fixture(value)
    except Exception as exc:
        summary["negative"].append({"case": name, "status": "PASS_EXPECTED_REJECTION", "error": type(exc).__name__, "message": str(exc)})
    else:
        raise SystemExit(f"negative fixture accepted: {name}")

summary["negative_count"] = len(summary["negative"])
summary["source_span_digest_after"] = tree_digest(source_span_root)
summary["source_spans_unchanged"] = summary["source_span_digest_before"] == summary["source_span_digest_after"]
if not summary["source_spans_unchanged"]:
    raise SystemExit("R0B-CP1 source spans changed during join qualification")
summary["status"] = "PASS"
(reports / "qualification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

# Unit tests cover pure contracts and accepted output documents.
environment = dict(os.environ)
environment["PYTHONPATH"] = str(cp1 / "src") + os.pathsep + str(cp2 / "src")
environment["RCF_R0B_CP2_OUTPUT_ROOT"] = str(outputs)
environment["PYTHONDONTWRITEBYTECODE"] = "1"
with (reports / "unittest.log").open("wb") as log:
    process = subprocess.run(
        [args.python, "-m", "unittest", "discover", "-s", str(cp2 / "tests"), "-p", "test_*.py", "-v"],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
    )
(reports / "unittest.json").write_text(json.dumps({"status": "PASS" if process.returncode == 0 else "FAIL", "exit_code": process.returncode}, indent=2, sort_keys=True) + "\n")
if process.returncode:
    raise SystemExit(process.returncode)
print(json.dumps(summary, indent=2, sort_keys=True))
