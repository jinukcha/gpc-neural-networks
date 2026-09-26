#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--tree", required=True)
parser.add_argument("--report", required=True)
args = parser.parse_args()

tree = Path(args.tree).resolve()
fort = tree / "RC_K0/child_designs/fortification"
cp2 = fort / "r0b_cp2"
checks = []
TURN_COMPARISON_TOLERANCE_DEG = 1e-6


def check(name, value, detail=None):
    checks.append({"name": name, "pass": bool(value), "detail": detail})

qualification = json.loads((cp2 / "reports/qualification.json").read_text())
check("qualification", qualification["status"] == "PASS", qualification)
check("families", set(qualification["families"]) == {"miter", "bevel", "transition"}, sorted(qualification["families"]))
check("clean_replay", all(row["byte_identical"] for row in qualification["families"].values()))
check("negative_gates", qualification["negative_count"] == 9, qualification["negative_count"])
check("source_spans_unchanged", qualification["source_spans_unchanged"], [qualification["source_span_digest_before"], qualification["source_span_digest_after"]])
expected = {
    "miter": {"family": "MITER", "turn": 10.0, "sections": 3, "overlap": 8.0},
    "bevel": {"family": "BEVEL", "turn": 30.0, "sections": 4, "overlap": 10.0},
    "transition": {"family": "PROFILE_TRANSITION", "turn": 0.0, "sections": 3, "overlap": 6.0},
}
for name, contract in expected.items():
    root = cp2 / "outputs" / name / "reference"
    result = json.loads((root / "result.json").read_text())
    plan = json.loads((root / "join-plan.json").read_text())
    sections = json.loads((root / "section-plan.json").read_text())
    alignment = json.loads((root / "socket-alignment.json").read_text())
    overlap = json.loads((root / "bounded-overlap.json").read_text())
    sockets = json.loads((root / "join-sockets.json").read_text())
    parts = json.loads((root / "semantic-parts.json").read_text())
    stored = json.loads((root / "stored-copies.json").read_text())
    mesh = json.loads((root / "neutral-mesh.json").read_text())
    check(f"{name}_status", result["status"] == "SUCCEEDED", result)
    check(f"{name}_family", result["family"] == contract["family"], result["family"])
    check(f"{name}_turn", abs(result["turn_angle_deg"] - contract["turn"]) <= TURN_COMPARISON_TOLERANCE_DEG, result["turn_angle_deg"])
    check(f"{name}_sections", result["section_count"] == contract["sections"] == len(sections["sections"]), result["section_count"])
    check(f"{name}_parts", result["part_count"] == 5 and len(parts["parts"]) == 5)
    check(f"{name}_sockets", result["socket_count"] == 6 and len(sockets["sockets"]) == 6)
    check(f"{name}_alignment", alignment["status"] == "PASS" and alignment["alignment_count"] == 6 and all(row["status"] == "PASS" for row in alignment["alignments"]), alignment)
    check(f"{name}_alignment_exact", all(row["exact_numeric_equal"] for row in alignment["alignments"]))
    check(f"{name}_overlap", overlap["status"] == "PASS" and overlap["incoming"]["overlap_m"] == contract["overlap"] and overlap["outgoing"]["overlap_m"] == contract["overlap"], overlap)
    check(f"{name}_bounded_ratio", overlap["incoming"]["ratio"] <= 0.5 and overlap["outgoing"]["ratio"] <= 0.5)
    check(f"{name}_stored", len(stored["copies"]) == 10)
    check(f"{name}_mesh", result["vertex_count"] > 0 and result["triangle_count"] > 0 and result["volume_m3"] > 0 and len(mesh["part_ranges"]) == 5)
    check(f"{name}_ownership", plan["join_ownership"] == "JOIN_MODULE_ONLY_SPAN_GEOMETRY_UNCHANGED")

status_rows = list(csv.DictReader((fort / "data/CP_STATUS.csv").open(newline="")))
check("parent_cp1_complete", any(row["stage"] == "R0B" and row["checkpoint"] == "CP1" and row["stage_completion"] == "R0B_CP1_COMPLETE" for row in status_rows))
bad = [path.as_posix() for path in tree.rglob("*") if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts or ".godot" in path.parts)]
check("no_cache", not bad, bad)
parse_errors = []
for path in cp2.rglob("*.json"):
    try: json.loads(path.read_text())
    except Exception as exc: parse_errors.append([path.as_posix(), str(exc)])
for path in cp2.rglob("*.csv"):
    try: list(csv.reader(path.open(newline="")))
    except Exception as exc: parse_errors.append([path.as_posix(), str(exc)])
check("machine_parse", not parse_errors, parse_errors)
report = {
    "schema": "royal-capital.fortification.r0b-cp2-validation/1",
    "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
    "checks": checks,
    "summary": {"passed": sum(row["pass"] for row in checks), "failed": sum(not row["pass"] for row in checks)},
}
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["status"] == "PASS" else 1)
