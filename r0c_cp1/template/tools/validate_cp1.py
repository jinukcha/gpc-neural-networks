#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--tree", required=True)
p.add_argument("--report", required=True)
a = p.parse_args()

tree = Path(a.tree).resolve()
fort = tree / "RC_K0/child_designs/fortification"
cp1 = fort / "r0c_cp1"
reference = cp1 / "outputs/reference"
checks: list[dict] = []


def check(name: str, passed: bool, detail=None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


families = {name: reference / name for name in ("round", "square", "polygonal")}
check("reference_families_present", all(path.is_dir() for path in families.values()), sorted(str(path) for path in families.values() if not path.is_dir()))
expected_family = {"round": "ROUND", "square": "SQUARE", "polygonal": "POLYGONAL"}
expected_sides = {"round": 32, "square": 4, "polygonal": 8}
expected_parts = ["foundation", "tower_body", "tower_crown"]
expected_sockets = [
    "span_in", "span_out", "wall_walk_in", "wall_walk_out",
    "foundation_in", "foundation_out", "foundation_center",
    "foundation_outside", "foundation_inside", "roof_socket",
]
required_root = {
    "tower-plan.json", "semantic-parts.json", "sockets.json",
    "bounds-attachment.json", "foundation-interface.json",
    "fixed-tessellation.json", "neutral-mesh.json", "stored-copies.json",
    "cad-provider-receipt.json", "result.json",
}

for name, root in families.items():
    check(f"{name}_required_files", required_root <= {path.name for path in root.iterdir()}, sorted(required_root - {path.name for path in root.iterdir()}))
    result = json.loads((root / "result.json").read_text())
    plan = json.loads((root / "tower-plan.json").read_text())
    parts = json.loads((root / "semantic-parts.json").read_text())
    sockets = json.loads((root / "sockets.json").read_text())
    bounds = json.loads((root / "bounds-attachment.json").read_text())
    foundation = json.loads((root / "foundation-interface.json").read_text())
    tess = json.loads((root / "fixed-tessellation.json").read_text())
    mesh = json.loads((root / "neutral-mesh.json").read_text())
    stored = json.loads((root / "stored-copies.json").read_text())
    receipt = json.loads((root / "cad-provider-receipt.json").read_text())

    check(f"{name}_status_family", result["status"] == "SUCCEEDED" and result["family"] == expected_family[name], result)
    check(f"{name}_part_contract", result["part_count"] == 3 and [row["part_id"] for row in parts["parts"]] == expected_parts, parts)
    check(f"{name}_socket_contract", result["socket_count"] == 10 and [row["socket_id"] for row in sockets["sockets"]] == expected_sockets, sockets)
    check(f"{name}_solid_positive", result["solid_count"] == 3 and result["volume_m3"] > 0.0, result)
    check(f"{name}_mesh_finite", result["vertex_count"] == len(mesh["vertices_m"]) and result["triangle_count"] == len(mesh["triangles"]) and all(math.isfinite(float(v)) for point in mesh["vertices_m"] for v in point), [result["vertex_count"], result["triangle_count"]])
    check(f"{name}_mesh_indices", all(0 <= int(index) < len(mesh["vertices_m"]) for triangle in mesh["triangles"] for index in triangle))
    check(f"{name}_stored_copy_contract", result["stored_copy_count"] == 6 and len(stored["copies"]) == 6 and {row["format"] for row in stored["copies"]} == {"STEP", "BREP"}, stored)
    check(f"{name}_foundation_evidence", foundation["status"] == "PASS" and foundation["contact_ratio"] == 1.0 and foundation["maximum_gap_m"] == 0.0 and foundation["terrain_mutation"] is False, foundation)
    check(f"{name}_attachment_evidence", bounds["bounds_status"] == "PASS" and bounds["attachment_intersection_count"] == 2 and bounds["attachment_width_m"] > 0.0 and bounds["body_outside_projection_m"] <= bounds["maximum_body_projection_m"], bounds)
    check(f"{name}_frame_contract", all(row["frame"]["determinant"] == 1.0 for row in sockets["sockets"]))
    check(f"{name}_side_count", plan["side_count"] == expected_sides[name], plan)
    check(f"{name}_deferred_scope", plan["tower_join_realization"] == "DEFERRED_TO_R0C_CP2" and plan["battlement_realization"] == "DEFERRED_TO_R0C_CP3" and plan["surface_source_coverage"] == "DEFERRED_TO_R0C_CP4" and receipt["surface_source_coverage"] == "NOT_AUTHORED_R0C_CP4")
    check(f"{name}_tessellation", tess["linear_deflection_m"] == 0.05 and tess["angular_deflection_rad"] == 0.1 and tess["mesh_round_digits"] == 9, tess)

round_tess = json.loads((reference / "round/fixed-tessellation.json").read_text())
check("round_approximation", round_tess["round_route"] == "POLYGONAL_APPROXIMATION_32" and round_tess["round_radial_chord_error_m"] <= round_tess["linear_deflection_m"], round_tess)

clean = json.loads((cp1 / "reports/clean_replay.json").read_text())
negative = json.loads((cp1 / "reports/negative_gates.json").read_text())
qualification = json.loads((cp1 / "reports/qualification.json").read_text())
unittest = json.loads((cp1 / "reports/unittest.json").read_text())
check("clean_replay", clean["status"] == "PASS" and clean["family_count"] == 3 and clean["all_byte_identical"] is True, clean)
check("negative_gates", negative["status"] == "PASS" and negative["case_count"] == 13 and negative["passed"] == 13 and negative["partial_output_published"] is False, negative)
check("qualification", qualification["status"] == "PASS" and len(qualification["positive_families"]) == 3, qualification)
check("unittest", unittest["status"] == "PASS" and unittest["tests_run"] == 8 and unittest["failures"] == 0 and unittest["errors"] == 0, unittest)
check("failure_receipts", len(list((cp1 / "failures").glob("*/failure-result.json"))) == 13)
check("runtime_wheels_retained", len(list((fort / "r0a_cp0/runtime/wheelhouse").glob("*.whl"))) == 58)

parse_errors = []
for path in fort.rglob("*.json"):
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        parse_errors.append({"path": str(path), "error": str(exc)})
for path in fort.rglob("*.csv"):
    try:
        list(csv.reader(path.open(encoding="utf-8", newline="")))
    except Exception as exc:
        parse_errors.append({"path": str(path), "error": str(exc)})
check("machine_files_parse", not parse_errors, parse_errors)

cache = [path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts or ".godot" in path.parts)]
check("generated_cache_absent", not cache, cache)

report = {
    "schema": "royal-capital.fortification.r0c-cp1-validation/1",
    "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
    "checks": checks,
    "summary": {"passed": sum(1 for row in checks if row["pass"]), "failed": sum(1 for row in checks if not row["pass"]), "total": len(checks)},
}
out = Path(a.report)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
raise SystemExit(0 if report["status"] == "PASS" else 1)
