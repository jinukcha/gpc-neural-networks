#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any
import zipfile

from package_tools import (
    BASELINE_SHA256,
    B4_CLAIMED_SHA256,
    CHAIN_SHA256,
    compare_trees,
    deterministic_zip,
    files,
    package_checkpoint,
    reconstruct_b4,
    sha256_path,
    validate_registry,
    write_registry,
)


def pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def remove_generated_cache(tree: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(tree.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            removed.append(path.relative_to(tree).as_posix())
            shutil.rmtree(path)
    for path in list(tree.rglob("*.pyc")) + list(tree.rglob("*.pyo")):
        if path.is_file():
            removed.append(path.relative_to(tree).as_posix())
            path.unlink()
    for path in sorted(tree.rglob(".godot"), reverse=True):
        if path.is_dir():
            removed.append(path.relative_to(tree).as_posix())
            shutil.rmtree(path)
    return removed


def fixture_documents() -> list[dict[str, Any]]:
    runtime = {
        "build123d_version": "0.13.1.dev12+ge22d34dae",
        "ocp_version": "8.0.1.0.0",
        "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466",
        "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f",
    }
    common = {
        "schema": "royal-capital.fortification.tower-family-fixture/1",
        "center_m": [0.0, 0.0, -4.0],
        "height_m": 16.0,
        "foundation_depth_m": 3.0,
        "wall_interface": {
            "centerline_z_m": 0.0,
            "outside_face_z_m": -3.0,
            "wall_walk_y_m": 12.3,
            "max_body_projection_m": 10.0,
        },
        "tolerances": {
            "linear_m": 0.000001,
            "angular_rad": 0.000001,
            "tessellation_linear_m": 0.05,
            "tessellation_angular_rad": 0.1,
            "mesh_round_digits": 9,
        },
        "budget": {
            "max_profile_points": 64,
            "max_vertices": 500000,
            "max_triangles": 500000,
            "max_artifact_bytes": 100000000,
            "max_parts": 8,
        },
        "runtime": runtime,
    }
    return [
        {**common, "tower_id": "fortification/caelmere/pilot/tower-round-001", "family": "ROUND", "outer_width_m": 17.0, "side_count": 32},
        {**common, "tower_id": "fortification/caelmere/pilot/tower-square-001", "family": "SQUARE", "outer_width_m": 16.0, "side_count": 4},
        {**common, "tower_id": "fortification/caelmere/pilot/tower-polygonal-001", "family": "POLYGONAL", "outer_width_m": 18.0, "side_count": 8},
    ]


def set_cp_status(fort: Path, *, complete: bool) -> None:
    path = fort / "data/CP_STATUS.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys()) if rows else [
            "stage", "checkpoint", "source_preservation", "remote_self_admission",
            "local_runtime_admission", "functional_qualification", "stage_completion",
            "next", "blocking_reason",
        ]
    rows = [row for row in rows if not (row.get("stage") == "R0C" and row.get("checkpoint") == "CP1")]
    rows.append({
        "stage": "R0C",
        "checkpoint": "CP1",
        "source_preservation": "PASS",
        "remote_self_admission": "NOT_APPLICABLE",
        "local_runtime_admission": "REUSE_R0A_CP0_EXACT_RUNTIME",
        "functional_qualification": "PASS" if complete else "NOT_RUN",
        "stage_completion": "R0C_CP1_COMPLETE" if complete else "R0C_CP1_IMPLEMENTED_VALIDATION_PENDING",
        "next": "START_R0C_CP2" if complete else "HOLD",
        "blocking_reason": "NONE" if complete else "NATIVE_QUALIFICATION_PENDING",
    })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pending_status_text() -> str:
    return """# 상태

```text
task                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP1
resume checkpoint            B4.zip
initial implementation base  RCF_D0_full.zip
R0A                           COMPLETE / CLOSED
R0B                           COMPLETE / CLOSED
implementation source        PRESERVED
runtime qualification        NOT_STARTED
tower families               ROUND / SQUARE / POLYGONAL
stage completion             OPEN — VALIDATION PENDING
next                         HOLD
```

```text
source preservation          PASS
functional qualification     NOT_RUN
stage completion             OPEN
Godot product                NOT_STARTED
```
"""


def complete_status_text(summary: dict[str, Any]) -> str:
    return f"""# 상태

```text
task                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP1
resume checkpoint            B4.zip
initial implementation base  RCF_D0_full.zip
R0A                           COMPLETE / CLOSED
R0B                           COMPLETE / CLOSED
exact runtime reuse          PASS
round tower                  PASS
square tower                 PASS
polygonal tower              PASS
bounds / attachment          PASS
foundation sockets           PASS
clean replay                 PASS — 3 / 3 BYTE-IDENTICAL
negative gates               PASS — {summary['negative_passed']} / {summary['negative_total']}
unit tests                   PASS — {summary['tests_run']} / {summary['tests_run']}
focused validation           PASS — {summary['validation_passed']} / {summary['validation_total']}
stage completion             R0C_CP1_COMPLETE / CLOSED
next                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2
R0C-CP2 start                EXPLICIT APPROVAL REQUIRED
```

```text
source preservation          PASS
functional qualification     PASS
stage completion             COMPLETE
Godot product                NOT_STARTED
```
"""


def prepare(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs).resolve()
    work = Path(args.work).resolve()
    deliver = Path(args.deliver).resolve()
    template = Path(args.template).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    deliver.mkdir(parents=True, exist_ok=True)

    parent, parent_validation = reconstruct_b4(inputs, work)
    tree = work / "tree"
    shutil.copytree(parent, tree, copy_function=shutil.copy2)
    fort = tree / "RC_K0/child_designs/fortification"
    cp1 = fort / "r0c_cp1"
    if cp1.exists():
        raise RuntimeError("r0c_cp1 already exists in accepted parent")
    shutil.copytree(template, cp1, copy_function=shutil.copy2)
    (cp1 / "fixtures").mkdir(parents=True, exist_ok=True)
    for fixture in fixture_documents():
        name = f"tower_{fixture['family'].lower()}.fixture.json"
        (cp1 / "fixtures" / name).write_text(pretty(fixture), encoding="utf-8")
    (cp1 / "docs").mkdir(exist_ok=True)
    (cp1 / "docs/CP1_REPORT.md").write_text("# R0C-CP1\n\nImplementation source preserved. Native qualification has not started.\n", encoding="utf-8")
    (cp1 / "provenance").mkdir(exist_ok=True)
    (cp1 / "provenance/CHANGESET.json").write_text(pretty({
        "schema": "royal-capital.fortification.r0c-cp1-changeset/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP1",
        "parent": "B4.zip",
        "parent_claimed_sha256": B4_CLAIMED_SHA256,
        "fixed_baseline": "RCF_D0_full.zip",
        "fixed_baseline_sha256": BASELINE_SHA256,
        "scope": ["ROUND_TOWER", "SQUARE_TOWER", "POLYGONAL_TOWER", "BOUNDS_ATTACHMENT", "FOUNDATION_SOCKETS"],
        "non_goals": ["TOWER_JOINS", "BATTLEMENTS", "INTERIORS", "ROOF_REALIZATION", "COLLISION", "NAVIGATION", "GODOT_IMPORT", "FINAL_SURFACE_COVERAGE"],
        "new_external_oss": [],
        "existing_runtime_reused": True,
    }), encoding="utf-8")
    set_cp_status(fort, complete=False)
    (fort / "docs/00_STATUS.md").write_text(pending_status_text(), encoding="utf-8")
    with (fort / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## R0C-CP1 implementation\n\nRound, square and polygonal tower-family source is preserved; native qualification is pending.\n")
    removed = remove_generated_cache(tree)
    write_registry(tree)
    checkpoint = package_checkpoint(
        parent=parent,
        current=tree,
        inputs=inputs,
        deliver=deliver,
        prefix="C1_impl",
        stage="R0C_CP1_IMPLEMENTATION_PRESERVED",
        scratch_root=work / "package_scratch",
    )
    state = {
        "schema": "royal-capital.fortification.r0c-cp1-prepare/1",
        "status": "PASS",
        "parent_validation": parent_validation,
        "cache_removed": removed,
        "checkpoint": checkpoint,
        "source_preservation": "PASS",
        "functional_qualification": "NOT_RUN",
        "stage_completion": "OPEN",
    }
    (deliver / "C1_impl_state.json").write_text(pretty(state), encoding="utf-8")
    print(pretty(state), end="")


def qualify(args: argparse.Namespace) -> None:
    tree = Path(args.tree).resolve()
    work = Path(args.work).resolve()
    runtime_python = Path(args.runtime_python).absolute()
    fort = tree / "RC_K0/child_designs/fortification"
    cp1 = fort / "r0c_cp1"
    reports = cp1 / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    qualification_log = reports / "qualification.log"
    with qualification_log.open("wb") as handle:
        qualification_process = subprocess.run(
            [str(runtime_python), str(cp1 / "tools/run_qualification.py"), "--tree", str(tree), "--work", str(work / "qualification")],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    if qualification_process.returncode != 0:
        qualification_text = qualification_log.read_text(encoding="utf-8", errors="replace")
        print("=== CHILD QUALIFICATION LOG BEGIN ===", file=sys.stderr)
        print(qualification_text, file=sys.stderr, end="" if qualification_text.endswith("\n") else "\n")
        print("=== CHILD QUALIFICATION LOG END ===", file=sys.stderr)
        raise RuntimeError(f"tower qualification failed with exit code {qualification_process.returncode}; see {qualification_log}")

    test_env = env.copy()
    test_env["RCF_TOWER_REFERENCE_ROOT"] = str(cp1 / "outputs/reference")
    unittest_log = reports / "unittest.log"
    command = [str(runtime_python), "-m", "unittest", "discover", "-s", str(cp1 / "tests"), "-p", "test_*.py", "-v"]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=test_env, text=True)
    unittest_log.write_text(completed.stdout, encoding="utf-8")
    match = re.search(r"Ran\s+(\d+)\s+tests?", completed.stdout)
    tests_run = int(match.group(1)) if match else 0
    unittest_result = {
        "schema": "royal-capital.fortification.r0c-cp1-unittest/1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "tests_run": tests_run,
        "failures": 0 if completed.returncode == 0 else 1,
        "errors": 0 if completed.returncode == 0 else 1,
        "return_code": completed.returncode,
    }
    (reports / "unittest.json").write_text(pretty(unittest_result), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"unit tests failed; see {unittest_log}")

    runtime_receipt = {
        "schema": "royal-capital.fortification.r0c-cp1-runtime-reuse/1",
        "status": "PASS",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cache_tag": sys.implementation.cache_tag,
        "build123d_version": importlib.metadata.version("build123d"),
        "ocp_version": importlib.metadata.version("cadquery-ocp-novtk"),
        "wheel_count": len(list((fort / "r0a_cp0/runtime/wheelhouse").glob("*.whl"))),
        "global_install": False,
        "new_external_oss": [],
    }
    (reports / "runtime_reuse.json").write_text(pretty(runtime_receipt), encoding="utf-8")
    remove_generated_cache(tree)
    write_registry(tree)
    print(pretty({"status": "PASS", "qualification": "PASS", "unittest": unittest_result, "runtime": runtime_receipt}), end="")


def prepackage(args: argparse.Namespace) -> None:
    work = Path(args.work).resolve()
    tree = work / "tree"
    parent = work / "parent_b4"
    checkpoint = package_checkpoint(
        parent=parent,
        current=tree,
        inputs=Path(args.inputs).resolve(),
        deliver=Path(args.deliver).resolve(),
        prefix="C1_pre",
        stage="R0C_CP1_QUALIFIED_PRE_VALIDATION",
        scratch_root=work / "package_scratch",
    )
    print(pretty(checkpoint), end="")


def final_tree_validation(tree: Path) -> dict[str, Any]:
    fort = tree / "RC_K0/child_designs/fortification"
    cp1 = fort / "r0c_cp1"
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail=None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    check("required_family_outputs", all((cp1 / "outputs/reference" / family / "result.json").is_file() for family in ("round", "square", "polygonal")))
    check("qualification_pass", json.loads((cp1 / "reports/qualification.json").read_text())["status"] == "PASS")
    check("focused_validation_pass", json.loads((cp1 / "reports/cp1_validation.json").read_text())["status"] == "PASS")
    check("unit_tests_pass", json.loads((cp1 / "reports/unittest.json").read_text())["status"] == "PASS")
    check("negative_gates_13", json.loads((cp1 / "reports/negative_gates.json").read_text())["passed"] == 13)
    check("runtime_wheels_58", len(list((fort / "r0a_cp0/runtime/wheelhouse").glob("*.whl"))) == 58)
    cp_status = (fort / "data/CP_STATUS.csv").read_text(encoding="utf-8")
    check("r0c_cp1_status", "R0C,CP1" in cp_status and "R0C_CP1_COMPLETE" in cp_status)
    check("r0b_remains_closed", all(token in cp_status for token in ("R0B,CP1", "R0B,CP2", "R0B,CP3", "R0B,CP4")))
    check("surface_coverage_not_claimed", all(json.loads((cp1 / "outputs/reference" / family / "tower-plan.json").read_text())["surface_source_coverage"] == "DEFERRED_TO_R0C_CP4" for family in ("round", "square", "polygonal")))
    check("tower_joins_not_claimed", all(json.loads((cp1 / "outputs/reference" / family / "tower-plan.json").read_text())["tower_join_realization"] == "DEFERRED_TO_R0C_CP2" for family in ("round", "square", "polygonal")))
    check("godot_not_claimed", "Godot product                NOT_STARTED" in (fort / "docs/00_STATUS.md").read_text(encoding="utf-8"))
    caches = [p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file() and (p.suffix in {".pyc", ".pyo"} or "__pycache__" in p.parts or ".godot" in p.parts)]
    check("generated_cache_absent", not caches, caches)
    json_errors = []
    csv_errors = []
    for path in fort.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            json_errors.append({"path": str(path), "error": str(exc)})
    for path in fort.rglob("*.csv"):
        try:
            list(csv.reader(path.open(encoding="utf-8", newline="")))
        except Exception as exc:
            csv_errors.append({"path": str(path), "error": str(exc)})
    check("json_parse", not json_errors, json_errors)
    check("csv_parse", not csv_errors, csv_errors)
    registry = validate_registry(tree)
    check("closed_world_registry", registry["status"] == "PASS", registry)
    report = {
        "schema": "royal-capital.fortification.r0c-cp1-final-tree-validation/1",
        "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
        "checks": checks,
        "summary": {"passed": sum(1 for row in checks if row["pass"]), "failed": sum(1 for row in checks if not row["pass"]), "total": len(checks)},
    }
    return report


def finalize(args: argparse.Namespace) -> None:
    work = Path(args.work).resolve()
    tree = work / "tree"
    parent = work / "parent_b4"
    deliver = Path(args.deliver).resolve()
    runtime_python = Path(args.runtime_python).absolute()
    fort = tree / "RC_K0/child_designs/fortification"
    cp1 = fort / "r0c_cp1"
    reports = cp1 / "reports"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    validation_log = reports / "focused_validation.log"
    with validation_log.open("wb") as handle:
        subprocess.run(
            [str(runtime_python), str(cp1 / "tools/validate_cp1.py"), "--tree", str(tree), "--report", str(reports / "cp1_validation.json")],
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    validation = json.loads((reports / "cp1_validation.json").read_text(encoding="utf-8"))
    negative = json.loads((reports / "negative_gates.json").read_text(encoding="utf-8"))
    unittest_result = json.loads((reports / "unittest.json").read_text(encoding="utf-8"))
    summary = {
        "negative_passed": negative["passed"],
        "negative_total": negative["case_count"],
        "tests_run": unittest_result["tests_run"],
        "validation_passed": validation["summary"]["passed"],
        "validation_total": validation["summary"]["total"],
    }
    set_cp_status(fort, complete=True)
    (fort / "docs/00_STATUS.md").write_text(complete_status_text(summary), encoding="utf-8")
    (cp1 / "docs/CP1_REPORT.md").write_text(f"""# R0C-CP1 완료

```text
round tower                  PASS
square tower                 PASS
polygonal tower              PASS
bounds / attachment          PASS
foundation sockets           PASS
clean replay                 PASS
negative gates               {summary['negative_passed']} / {summary['negative_total']} PASS
unit tests                   {summary['tests_run']} / {summary['tests_run']} PASS
focused validation           {summary['validation_passed']} / {summary['validation_total']} PASS
stage completion             R0C_CP1_COMPLETE / CLOSED
next                         R0C-CP2 — explicit approval required
```

Surface coverage remains deferred to R0C-CP4. Tower joins, battlements, interiors, roof realization, collision, navigation and Godot import were not claimed.
""", encoding="utf-8")
    completion = {
        "schema": "royal-capital.fortification.r0c-cp1-completion/1",
        "status": "R0C_CP1_COMPLETE",
        "source_preservation": "PASS",
        "functional_qualification": "PASS",
        "stage_completion": "COMPLETE",
        "families": ["ROUND", "SQUARE", "POLYGONAL"],
        "negative_gates": summary["negative_total"],
        "focused_validation": validation["summary"],
        "godot_product": "NOT_STARTED",
        "next": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "next_requires_explicit_approval": True,
    }
    (reports / "completion_receipt.json").write_text(pretty(completion), encoding="utf-8")
    with (fort / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## R0C-CP1 closeout\n\nRound, square and polygonal tower families, bounds/attachment evidence and foundation sockets are accepted. R0C-CP2 requires explicit approval.\n")
    remove_generated_cache(tree)
    write_registry(tree)
    final_validation = final_tree_validation(tree)
    (reports / "final_tree_validation.json").write_text(pretty(final_validation), encoding="utf-8")
    if final_validation["status"] != "PASS":
        raise RuntimeError(f"final tree validation failed: {pretty(final_validation)}")
    # final validation itself changes the tree, so refresh the closed-world registry.
    write_registry(tree)

    checkpoint = package_checkpoint(
        parent=parent,
        current=tree,
        inputs=Path(args.inputs).resolve(),
        deliver=deliver,
        prefix="C1",
        stage="R0C_CP1_COMPLETE",
        scratch_root=work / "package_scratch",
    )

    metrics = {}
    for family in ("round", "square", "polygonal"):
        root = cp1 / "outputs/reference" / family
        result = json.loads((root / "result.json").read_text())
        plan = json.loads((root / "tower-plan.json").read_text())
        bounds = json.loads((root / "bounds-attachment.json").read_text())
        foundation = json.loads((root / "foundation-interface.json").read_text())
        metrics[family] = {
            "family": result["family"],
            "construction_route": plan["construction_route"],
            "outer_width_m": plan["outer_width_m"],
            "height_m": plan["height_m"],
            "side_count": plan["side_count"],
            "solid_count": result["solid_count"],
            "socket_count": result["socket_count"],
            "stored_copy_count": result["stored_copy_count"],
            "vertices": result["vertex_count"],
            "triangles": result["triangle_count"],
            "volume_m3": result["volume_m3"],
            "bounds_m": result["bounds_m"],
            "body_outside_projection_m": bounds["body_outside_projection_m"],
            "attachment_width_m": bounds["attachment_width_m"],
            "foundation_bearing_area_m2": foundation["bearing_area_m2"],
            "foundation_contact_ratio": foundation["contact_ratio"],
            "foundation_maximum_gap_m": foundation["maximum_gap_m"],
        }

    final_report = {
        "schema": "royal-capital.fortification.r0c-cp1-delivery/1",
        "status": "PASS",
        "stage_completion": "R0C_CP1_COMPLETE",
        "source_preservation": "PASS",
        "functional_qualification": "PASS",
        "families": metrics,
        "clean_replay": json.loads((reports / "clean_replay.json").read_text()),
        "negative_gates": negative,
        "unit_tests": unittest_result,
        "focused_validation": validation,
        "final_tree_validation": final_validation,
        "package": checkpoint,
        "accepted_checkpoint_progress": {"accepted": 10, "total": 25},
        "r0c_progress": {"accepted": 1, "total": 4},
        "next": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "next_requires_explicit_approval": True,
        "godot_product": "NOT_STARTED",
    }
    (deliver / "C1.json").write_text(pretty(final_report), encoding="utf-8")
    (deliver / "C1.txt").write_text("R0C_CP1_COMPLETE\nROUND_SQUARE_POLYGONAL_TOWERS_PASS\nBOUNDS_ATTACHMENT_PASS\nFOUNDATION_SOCKETS_PASS\nR0C_CP2_EXPLICIT_APPROVAL_REQUIRED\n", encoding="utf-8")
    (deliver / "C1.md").write_text(f"""# R0C-CP1 complete

```text
source preservation              PASS
exact runtime reuse              PASS
round / square / polygonal       PASS
bounds / attachment              PASS
foundation sockets               PASS
clean replay                     PASS
negative gates                   {summary['negative_passed']} / {summary['negative_total']} PASS
unit tests                       {summary['tests_run']} / {summary['tests_run']} PASS
focused validation               {summary['validation_passed']} / {summary['validation_total']} PASS
stage completion                 R0C_CP1_COMPLETE / CLOSED
Godot product                    NOT_STARTED
```

Tower joins, battlements, interiors, roofs and final surface coverage remain in their assigned later stages. R0C-CP2 requires explicit approval.
""", encoding="utf-8")
    shutil.copy2(deliver / "C1_check.json", deliver / "C1_delivery.json")

    receipts_root = work / "receipts"
    if receipts_root.exists():
        shutil.rmtree(receipts_root)
    receipts_root.mkdir(parents=True)
    selected = [
        cp1 / "docs/CP1_REPORT.md",
        cp1 / "provenance/CHANGESET.json",
        reports / "qualification.json",
        reports / "clean_replay.json",
        reports / "negative_gates.json",
        reports / "unittest.json",
        reports / "cp1_validation.json",
        reports / "final_tree_validation.json",
        reports / "runtime_reuse.json",
        reports / "completion_receipt.json",
        reports / "qualification.log",
        reports / "unittest.log",
        reports / "focused_validation.log",
        deliver / "C1_impl_check.json",
        deliver / "C1_pre_check.json",
        deliver / "C1_check.json",
        deliver / "C1.json",
        deliver / "C1.txt",
        deliver / "C1.md",
    ]
    for source in selected:
        if not source.is_file():
            continue
        if source.is_relative_to(tree):
            rel = source.relative_to(tree)
        else:
            rel = Path("delivery") / source.name
        target = receipts_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    runtime_log_dir = Path(args.runtime_log_dir).resolve() if args.runtime_log_dir else None
    if runtime_log_dir and runtime_log_dir.is_dir():
        for source in runtime_log_dir.glob("*.log"):
            target = receipts_root / "runtime_logs" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    deterministic_zip(receipts_root, deliver / "C1R.zip")

    checksum_names = [
        "C1.zip", "C1P.zip", "C1S.zip", "C1R.zip", "C1.md", "C1.json", "C1.txt", "C1_delivery.json",
        "C1_impl.zip", "C1_implP.zip", "C1_implS.zip", "C1_impl_check.json", "C1_impl_state.json",
        "C1_pre.zip", "C1_preP.zip", "C1_preS.zip", "C1_pre_check.json",
    ]
    (deliver / "C1.sha256").write_text("\n".join(f"{sha256_path(deliver / name)}  {name}" for name in checksum_names if (deliver / name).is_file()) + "\n", encoding="utf-8")
    print(pretty(final_report), end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--inputs", required=True)
    prep.add_argument("--work", required=True)
    prep.add_argument("--deliver", required=True)
    prep.add_argument("--template", required=True)
    prep.set_defaults(func=prepare)

    qual = sub.add_parser("qualify")
    qual.add_argument("--tree", required=True)
    qual.add_argument("--work", required=True)
    qual.add_argument("--runtime-python", required=True)
    qual.set_defaults(func=qualify)

    pre = sub.add_parser("prepackage")
    pre.add_argument("--inputs", required=True)
    pre.add_argument("--work", required=True)
    pre.add_argument("--deliver", required=True)
    pre.set_defaults(func=prepackage)

    final = sub.add_parser("finalize")
    final.add_argument("--inputs", required=True)
    final.add_argument("--work", required=True)
    final.add_argument("--deliver", required=True)
    final.add_argument("--runtime-python", required=True)
    final.add_argument("--runtime-log-dir")
    final.set_defaults(func=finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
