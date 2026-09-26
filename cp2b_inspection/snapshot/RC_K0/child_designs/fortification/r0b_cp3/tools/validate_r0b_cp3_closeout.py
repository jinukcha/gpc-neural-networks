#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

EXPECTED_BUILD123D = "0.13.1.dev12+ge22d34dae"
EXPECTED_OCP = "8.0.1.0.0"
EXPECTED_BASELINE = "RCF_D0_full.zip"
EXPECTED_PARENT = "B2.zip"
REQUIRED_REFERENCE_FILES = {
    "cad-provider-receipt.json",
    "canonical-terrain-profile.json",
    "construction-units.json",
    "contact-evidence.json",
    "fixed-tessellation.json",
    "foundation-interface.json",
    "grade-evidence.json",
    "interface-sockets.json",
    "neutral-mesh.json",
    "result.json",
    "semantic-parts.json",
    "stored-copies.json",
    "terrain-span-plan.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--completion-text", required=True)
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    fort = tree / "RC_K0/child_designs/fortification"
    cp3 = fort / "r0b_cp3"
    reports = cp3 / "reports"
    stepped_root = cp3 / "outputs/stepped/reference"
    retaining_root = cp3 / "outputs/retaining/reference"

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    qualification = read_json(reports / "qualification.json")
    focused = read_json(reports / "r0b_cp3_validation.json")
    unittest = read_json(reports / "unittest.json")
    runtime = read_json(reports / "runtime_reuse_receipt.json")
    changeset = read_json(cp3 / "provenance/CHANGESET.json")
    stepped_result = read_json(stepped_root / "result.json")
    retaining_result = read_json(retaining_root / "result.json")
    stepped_interface = read_json(stepped_root / "foundation-interface.json")
    retaining_interface = read_json(retaining_root / "foundation-interface.json")
    stepped_contact = read_json(stepped_root / "contact-evidence.json")
    retaining_contact = read_json(retaining_root / "contact-evidence.json")
    stepped_grade = read_json(stepped_root / "grade-evidence.json")
    retaining_grade = read_json(retaining_root / "grade-evidence.json")
    stepped_stored = read_json(stepped_root / "stored-copies.json")
    retaining_stored = read_json(retaining_root / "stored-copies.json")

    check("qualification_pass", qualification.get("status") == "PASS", qualification.get("status"))
    check(
        "focused_validation_56_of_56",
        focused.get("status") == "PASS"
        and focused.get("summary", {}).get("passed") == 56
        and focused.get("summary", {}).get("failed") == 0,
        focused.get("summary"),
    )
    check("unit_tests_pass", unittest.get("status") == "PASS" and unittest.get("exit_code") == 0, unittest)
    check("runtime_reuse_pass", runtime.get("status") == "PASS" and runtime.get("pip_check") == "PASS", runtime)
    check("runtime_versions_exact", runtime.get("python_version") == "3.13.5" and runtime.get("build123d_version") == EXPECTED_BUILD123D and runtime.get("ocp_version") == EXPECTED_OCP, runtime)
    check("runtime_wheel_count_58", runtime.get("wheel_count") == 58, runtime.get("wheel_count"))
    check("global_install_false", runtime.get("global_install") is False, runtime.get("global_install"))
    wheel_count = len(list((fort / "r0a_cp0/runtime/wheelhouse").glob("*.whl")))
    check("bundled_wheel_count_58", wheel_count == 58, wheel_count)

    cp_rows = list(csv.DictReader((fort / "data/CP_STATUS.csv").open(newline="", encoding="utf-8")))
    by_cp = {(row["stage"], row["checkpoint"]): row for row in cp_rows}
    r0a_expected = [("R0A", f"CP{i}") for i in range(5)]
    check("r0a_cp0_cp4_complete", all(by_cp.get(key, {}).get("stage_completion") in {"R0A_CP0_COMPLETE", "R0A_CP1_COMPLETE", "R0A_CP2_COMPLETE", "R0A_CP3_COMPLETE", "R0A_COMPLETE"} for key in r0a_expected), {str(key): by_cp.get(key) for key in r0a_expected})
    r0b_expected = [("R0B", f"CP{i}") for i in range(1, 4)]
    check("r0b_cp1_cp3_complete", all(by_cp.get(key, {}).get("stage_completion") == f"R0B_CP{i}_COMPLETE" for i, key in enumerate(r0b_expected, start=1)), {str(key): by_cp.get(key) for key in r0b_expected})
    cp3_row = by_cp.get(("R0B", "CP3"), {})
    check("cp3_next_explicit_r0b_cp4", cp3_row.get("next") == "START_R0B_CP4_BY_EXPLICIT_APPROVAL", cp3_row)

    status_text = (fort / "docs/00_STATUS.md").read_text(encoding="utf-8")
    root_readme = (tree / "RC_K0/README.md").read_text(encoding="utf-8")
    fort_readme = (fort / "README.md").read_text(encoding="utf-8")
    report_text = (cp3 / "docs/CP3_REPORT.md").read_text(encoding="utf-8")
    check("status_declares_cp3_closed", "R0B_CP3_COMPLETE / CLOSED" in status_text and "auto continuation             FORBIDDEN" in status_text, None)
    check("readmes_synced", "RC-FORT-R0B-CP3    COMPLETE / CLOSED" in root_readme and "R0B-CP3                       COMPLETE / CLOSED" in fort_readme, None)
    check("cp3_report_synced", "focused validation           PASS — 56 / 56" in report_text and "final tree validation        PASS — 34 / 34" in report_text, None)
    check("godot_not_claimed", "Godot product                 NOT_STARTED" in status_text, None)
    check("surface_coverage_deferred", "terrain source coverage       DEFERRED — R0B-CP4" in status_text, None)

    check("stepped_success", stepped_result.get("status") == "SUCCEEDED" and stepped_result.get("family") == "TERRAIN_STEPPED" and stepped_result.get("segment_count") == 3 and stepped_result.get("unit_count") == 15, stepped_result)
    check("retaining_success", retaining_result.get("status") == "SUCCEEDED" and retaining_result.get("family") == "RETAINING" and retaining_result.get("segment_count") == 1 and retaining_result.get("unit_count") == 5, retaining_result)
    check("terrain_not_mutated", stepped_result.get("terrain_mutation") is False and retaining_result.get("terrain_mutation") is False, [stepped_result.get("terrain_mutation"), retaining_result.get("terrain_mutation")])
    check("foundation_interfaces_pass", stepped_interface.get("summary", {}).get("status") == "PASS" and retaining_interface.get("summary", {}).get("status") == "PASS", [stepped_interface.get("summary"), retaining_interface.get("summary")])
    check("contact_evidence_pass", stepped_contact.get("summary", {}).get("status") == "PASS" and retaining_contact.get("summary", {}).get("status") == "PASS" and stepped_contact.get("summary", {}).get("contact_ratio") == 1.0 and retaining_contact.get("summary", {}).get("contact_ratio") == 1.0 and stepped_contact.get("summary", {}).get("maximum_gap_m") == 0.0 and retaining_contact.get("summary", {}).get("maximum_gap_m") == 0.0, {"stepped": stepped_contact, "retaining": retaining_contact})
    check("grade_evidence_pass", stepped_grade.get("status") == "PASS" and retaining_grade.get("status") == "PASS" and stepped_grade.get("overall_centerline_grade") == 0.066666666667 and retaining_grade.get("retained_height_m") == 3.0, {"stepped": stepped_grade, "retaining": retaining_grade})
    check("stored_copy_counts", len(stepped_stored.get("copies", [])) == 30 and len(retaining_stored.get("copies", [])) == 10, [len(stepped_stored.get("copies", [])), len(retaining_stored.get("copies", []))])
    check("reference_file_sets", {p.name for p in stepped_root.iterdir() if p.is_file()} == REQUIRED_REFERENCE_FILES and {p.name for p in retaining_root.iterdir() if p.is_file()} == REQUIRED_REFERENCE_FILES, {"stepped": sorted(p.name for p in stepped_root.iterdir() if p.is_file()), "retaining": sorted(p.name for p in retaining_root.iterdir() if p.is_file())})
    check("negative_gates_10", qualification.get("negative_count") == 10 and all(item.get("status") == "PASS_EXPECTED_REJECTION" for item in qualification.get("negative", [])), qualification.get("negative"))
    check("changeset_baselines", changeset.get("fixed_initial_baseline") == EXPECTED_BASELINE and changeset.get("parent_checkpoint") == EXPECTED_PARENT, changeset)
    check("no_new_external_sources", changeset.get("new_external_sources") == [] and changeset.get("qualification", {}).get("new_external_sources") is False, changeset)

    required_contracts = [
        cp3 / "data/TERRAIN_SPAN_CONTRACT.json",
        cp3 / "data/FOUNDATION_INTERFACE_CONTRACT.json",
        cp3 / "data/EVIDENCE_CONTRACT.json",
        cp3 / "schemas/terrain_wall_span.schema.json",
        cp3 / "src/rcf_fortification_terrain_span/model.py",
        cp3 / "src/rcf_fortification_terrain_span/producer.py",
        cp3 / "tests/test_r0b_cp3.py",
        cp3 / "tools/run_r0b_cp3_qualification.py",
        cp3 / "tools/validate_r0b_cp3.py",
        cp3 / "tools/validate_r0b_cp3_closeout.py",
        cp3 / "provenance/CHANGESET.json",
    ]
    check("required_source_contracts_present", all(path.is_file() for path in required_contracts), [str(path.relative_to(tree)) for path in required_contracts if not path.is_file()])
    license_files = list((fort / "licenses").glob("*LICENSE*.txt"))
    check("licenses_retained", len(license_files) >= 3, [path.name for path in license_files])

    cache_files = [
        path.relative_to(tree).as_posix()
        for path in tree.rglob("*")
        if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts or ".godot" in path.parts)
    ]
    check("no_generated_cache", not cache_files, cache_files)
    nested_archives = [path.relative_to(tree).as_posix() for path in tree.rglob("*.zip")]
    check("no_nested_checkpoint_zip", not nested_archives, nested_archives)
    symlinks = [path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_symlink()]
    check("no_symlinks", not symlinks, symlinks)

    parse_errors: list[dict[str, str]] = []
    for path in tree.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append({"path": path.relative_to(tree).as_posix(), "error": str(exc)})
    for path in tree.rglob("*.csv"):
        try:
            list(csv.reader(path.open(newline="", encoding="utf-8")))
        except Exception as exc:
            parse_errors.append({"path": path.relative_to(tree).as_posix(), "error": str(exc)})
    check("machine_json_csv_parse", not parse_errors, parse_errors)

    check("external_runtime_not_bundled", not (cp3 / "runtime_attempt_01").exists() and runtime.get("runtime_location") == "EXTERNAL_FRESH_VENV_NOT_BUNDLED", runtime.get("runtime_location"))

    # This script intentionally has exactly 34 focused closeout gates. Keep the
    # count stable so the status document cannot silently drift.
    if len(checks) != 34:
        raise RuntimeError(f"closeout check count changed: {len(checks)} != 34")

    status = "PASS" if all(item["pass"] for item in checks) else "FAIL"
    summary = {"passed": sum(1 for item in checks if item["pass"]), "failed": sum(1 for item in checks if not item["pass"]), "total": len(checks)}
    report = {
        "schema": "royal-capital.fortification.r0b-cp3-closeout-validation/1",
        "status": status,
        "checks": checks,
        "summary": summary,
        "stage_completion": "R0B_CP3_COMPLETE" if status == "PASS" else "HOLD_R0B_CP3_OPEN",
        "next": "R0B_CP4_BY_EXPLICIT_APPROVAL" if status == "PASS" else "FIX_R0B_CP3",
        "godot_product": "NOT_STARTED",
    }
    completion = {
        "schema": "royal-capital.fortification.r0b-cp3-completion/1",
        "status": status,
        "source_preservation": "PASS",
        "runtime_reuse": "PASS",
        "functional_qualification": "PASS" if status == "PASS" else "HOLD",
        "stage_completion": "R0B_CP3_COMPLETE" if status == "PASS" else "HOLD_R0B_CP3_OPEN",
        "focused_validation": focused.get("summary"),
        "final_tree_validation": summary,
        "terrain_stepped": qualification["families"]["stepped"],
        "retaining": qualification["families"]["retaining"],
        "negative_gate_count": qualification.get("negative_count"),
        "terrain_mutation": False,
        "godot_product": "NOT_STARTED",
        "next": "R0B_CP4_BY_EXPLICIT_APPROVAL" if status == "PASS" else "FIX_R0B_CP3",
        "auto_continuation": False,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.text).write_text(
        "\n".join(
            [
                f"STATUS={status}",
                f"PASSED={summary['passed']}",
                f"FAILED={summary['failed']}",
                "STAGE_COMPLETION=" + report["stage_completion"],
                "NEXT=" + report["next"],
                "GODOT_PRODUCT=NOT_STARTED",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    Path(args.completion).write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.completion_text).write_text(
        "\n".join(
            [
                "R0B_CP3_COMPLETE" if status == "PASS" else "HOLD_R0B_CP3_OPEN",
                "SOURCE_PRESERVATION_PASS",
                "FUNCTIONAL_QUALIFICATION_PASS" if status == "PASS" else "FUNCTIONAL_QUALIFICATION_HOLD",
                "TERRAIN_STEPPED_PASS",
                "RETAINING_PASS",
                "FOUNDATION_INTERFACE_PASS",
                "CONTACT_GAP_GRADE_EVIDENCE_PASS",
                "NEGATIVE_GATES_10_OF_10_PASS",
                "R0B_CP4_EXPLICIT_APPROVAL_REQUIRED",
                "GODOT_PRODUCT_NOT_STARTED",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
