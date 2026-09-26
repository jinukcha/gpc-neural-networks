from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "r0c_cp1"))
import package_tools as pt  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "cp2a_tools"))
import build_cp2a as cp2a  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
from cp2b_tools import driver as cp2b  # noqa: E402

BASELINE_SHA256 = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
C1_SHA256 = "91cf2025f311772b0a137d3509efa0b1d65b96832a78f1e4f46ba9cf3b06e878"
C1P_SHA256 = "dddb960aa467689eecf1e312724ce06c0c9175e0fdc902c534c4e12ffdca77b4"
C2B_SHA256 = "9bc60a867b53c519b7729f1b8b9181fbb2e0f1edc5ea5853f288e8eab93001d4"
C2BP_SHA256 = "d38288dfb406dc37f17283fed108db6d4b47db0d75d7aadec73fecba390d46b7"
C2B_CHECK_SHA256 = "d9433238011d34a80fdd02b362d53f1d4454e720c45d5602c9b980601ee5eeee"
STAGE = "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2_CP2-C_REQUIRED_FAMILY_MATRIX"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return pt.sha256_path(path)


def artifact(path: Path) -> dict[str, Any]:
    return cp2a.artifact(path)


def source_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def copy_overlay(source_root: Path, destination: Path) -> list[str]:
    return cp2b.copy_overlay(source_root, destination)


def remove_disposable_caches(root: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts):
            removed.append(path.relative_to(root).as_posix())
            path.unlink()
    for directory in sorted((path for path in root.rglob("__pycache__") if path.is_dir()), reverse=True):
        shutil.rmtree(directory, ignore_errors=True)
    return sorted(removed)


def validate_parent_inputs(inputs: Path) -> dict[str, Any]:
    expected = {
        "C1.zip": C1_SHA256,
        "C1P.zip": C1P_SHA256,
        "C2B.zip": C2B_SHA256,
        "C2BP.zip": C2BP_SHA256,
        "C2B_check.json": C2B_CHECK_SHA256,
    }
    observed: dict[str, Any] = {}
    for name, digest in expected.items():
        path = inputs / name
        require(path.is_file(), f"missing accepted input: {name}")
        actual = sha256(path)
        require(actual == digest, f"{name} SHA mismatch {actual} != {digest}")
        observed[name] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
            "zip": artifact(path) if path.suffix == ".zip" else None,
        }
    check = json.loads((inputs / "C2B_check.json").read_text(encoding="utf-8"))
    require(check.get("stage_completion") == "CP2_B_COMPLETE_CLOSED", "accepted C2B stage is not closed")
    require(check.get("functional_qualification") == "PASS_FX01", "accepted C2B FX01 qualification missing")
    require(check.get("r0c_cp2") == "OPEN", "accepted C2B incorrectly closes R0C-CP2")
    require(check.get("validation", {}).get("status") == "PASS", "accepted C2B validation is not PASS")
    parent_proof = check["validation"]["fixed_baseline_plus_cumulative_patch_equals_full"]
    require(str(parent_proof.get("status", "")).startswith("PASS_"), "accepted C2B fixed-baseline proof is not admitted")
    return {"files": observed, "C2B_check": check}


def build_cumulative_patch(
    parent_patch: Path,
    current_delta: Path,
    destination: Path,
    scratch: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    old_stage = cp2b.STAGE
    cp2b.STAGE = stage
    try:
        return cp2b.build_cumulative_patch(
            parent_patch,
            current_delta,
            destination,
            scratch,
            parent_full_name="C2B.zip",
            parent_full_sha256=C2B_SHA256,
            parent_patch_name="C2BP.zip",
            parent_patch_sha256=C2BP_SHA256,
        )
    finally:
        cp2b.STAGE = old_stage


def _checkpoint_names(label: str) -> dict[str, str]:
    return {
        "full": f"C2C_{label}_pre.zip",
        "patch": f"C2C_{label}_preP.zip",
        "source": f"C2C_{label}_preS.zip",
        "check": f"C2C_{label}_pre_check.json",
    }


def build_checkpoint(
    *,
    label: str,
    authority: dict[str, Any],
    c1_tree: Path,
    parent: Path,
    current: Path,
    inputs: Path,
    work: Path,
    output: Path,
    copied: list[str],
    removed_caches: list[str],
) -> dict[str, Any]:
    names = _checkpoint_names(label)
    full = output / names["full"]
    patch = output / names["patch"]
    source = output / names["source"]
    current_delta = work / f"C2C_{label}_current_delta.zip"
    stage = f"{STAGE}_{label}_PREQUALIFICATION"
    source_meta = pt.create_source_delta(
        c1_tree,
        current,
        source,
        parent_name="C1.zip",
        stage=stage,
    )
    current_delta_meta = pt.create_source_delta(
        parent,
        current,
        current_delta,
        parent_name="C2B.zip",
        stage=stage + "_CURRENT_DELTA",
    )
    patch_meta = build_cumulative_patch(
        inputs / "C2BP.zip",
        current_delta,
        patch,
        work / f"{label.lower()}_patch_build",
        stage=stage,
    )
    pt.deterministic_zip(current, full)
    full_reopen = pt.verify_zip_tree(full, current, work / f"verify_{label.lower()}_full")
    c1_plus_source = pt.verify_parent_plus_delta(c1_tree, source, current, work / f"verify_{label.lower()}_source")
    c2b_plus_delta = pt.verify_parent_plus_delta(parent, current_delta, current, work / f"verify_{label.lower()}_current")
    patch_payload = pt.verify_patch_payload(patch, current, source, work / f"verify_{label.lower()}_patch")
    for title, value in {
        "full reopen": full_reopen,
        "C1 + source": c1_plus_source,
        "C2B + current delta": c2b_plus_delta,
        "patch payload": patch_payload,
    }.items():
        require(value["status"] == "PASS", f"{label} {title} failed: {value}")
    check = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2c-pre-checkpoint/1",
        "status": "PRESERVED_BEFORE_EXACT_RUNTIME_QUALIFICATION",
        "checkpoint": f"CP2-C_{label}",
        "authority": authority,
        "source_overlay_paths": copied,
        "removed_disposable_caches": removed_caches,
        "registries": {
            "C1": pt.validate_registry(c1_tree),
            "C2B": pt.validate_registry(parent),
            "implementation": pt.validate_registry(current),
        },
        "source_delta": source_meta,
        "current_delta": current_delta_meta,
        "cumulative_patch": patch_meta,
        "artifacts": {
            names["full"]: artifact(full),
            names["patch"]: artifact(patch),
            names["source"]: artifact(source),
        },
        "equality": {
            "full_reopen": full_reopen,
            "C1_plus_source": c1_plus_source,
            "C2B_plus_current_delta": c2b_plus_delta,
            "patch_current_delta_equals_source_payload": patch_payload,
            "fixed_baseline_plus_cumulative_patch": {
                "status": "PASS_COMPOSITIONAL_WITH_ACCEPTED_C2B_REPLAY",
                "accepted_parent_proof": authority["C2B_check"]["validation"]["fixed_baseline_plus_cumulative_patch_equals_full"],
                "current_parent_replay": c2b_plus_delta,
                "conclusion": f"RCF_D0_full.zip + {names['patch']} = {names['full']}",
                "direct_fixed_baseline_replay_this_run": "NOT_REPEATED_ACCEPTED_PARENT_CHAIN_REOPENED",
            },
        },
        "functional_qualification": "NOT_RUN",
        "stage_completion": "OPEN",
        "godot_product": "NOT_STARTED",
    }
    write_json(output / names["check"], check)
    return check


def prepare_fx02(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs).resolve()
    source_root = Path(args.source_root).resolve()
    work = Path(args.work).resolve()
    output = Path(args.output).resolve()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    authority = validate_parent_inputs(inputs)
    c1_tree = work / "c1"
    parent = work / "parent_c2b"
    current = work / "current"
    pt.safe_extract(inputs / "C1.zip", c1_tree)
    pt.safe_extract(inputs / "C2B.zip", parent)
    shutil.copytree(parent, current, copy_function=shutil.copy2)
    c1_registry = pt.validate_registry(c1_tree)
    parent_registry = pt.validate_registry(parent)
    require(c1_registry["status"] == "PASS", f"C1 registry failed: {c1_registry}")
    require(c1_registry["tree_files_including_registry"] == 1418, f"C1 file count mismatch: {c1_registry}")
    require(parent_registry["status"] == "PASS", f"C2B registry failed: {parent_registry}")
    require(parent_registry["tree_files_including_registry"] == 1459, f"C2B file count mismatch: {parent_registry}")
    removed_caches = remove_disposable_caches(current)
    copied = copy_overlay(source_root, current)
    require(copied, "FX02 source overlay is empty")
    receipt = current / "RC_K0/child_designs/fortification/r0c_cp2/reports/CP2C_FX02_SOURCE_CHECKPOINT.json"
    write_json(
        receipt,
        {
            "schema": "royal-capital.fortification.r0c-cp2.cp2c-source-checkpoint/1",
            "status": "PRESERVED_BEFORE_EXACT_RUNTIME_QUALIFICATION",
            "checkpoint": "CP2-C_R0C_CP2_FX02",
            "accepted_parent": {"name": "C2B.zip", "sha256": C2B_SHA256},
            "fixed_baseline": {"name": "RCF_D0_full.zip", "sha256": BASELINE_SHA256},
            "source_overlay": source_inventory(source_root),
            "removed_disposable_caches": removed_caches,
            "functional_qualification": "NOT_RUN",
            "partial_output_published": False,
        },
    )
    pt.write_registry(current)
    registry = pt.validate_registry(current)
    require(registry["status"] == "PASS", f"FX02 implementation registry failed: {registry}")
    build_checkpoint(
        label="FX02",
        authority=authority,
        c1_tree=c1_tree,
        parent=parent,
        current=current,
        inputs=inputs,
        work=work,
        output=output,
        copied=copied,
        removed_caches=removed_caches,
    )
    write_json(
        work / "STATE.json",
        {
            "current": current.as_posix(),
            "c1": c1_tree.as_posix(),
            "parent_c2b": parent.as_posix(),
            "output": output.as_posix(),
            "last_preserved": "FX02",
        },
    )
    print(json.dumps({"status": "FX02_PREPARED", "current": current.as_posix()}, sort_keys=True))


def prepare_fx03(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs).resolve()
    source_root = Path(args.source_root).resolve()
    work = Path(args.work).resolve()
    output = Path(args.output).resolve()
    authority = validate_parent_inputs(inputs)
    c1_tree = work / "c1"
    parent = work / "parent_c2b"
    current = work / "current"
    require(current.is_dir() and c1_tree.is_dir() and parent.is_dir(), "FX02 prepared work tree is missing")
    qualification_path = current / "RC_K0/child_designs/fortification/r0c_cp2/reports/CP2C_FX02_qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    require(qualification.get("status") == "PASS", "FX02 qualification is not PASS")
    require(qualification.get("stage_completion") == "CP2_C_FX02_COMPLETE_CLOSED", "FX02 stage completion mismatch")
    removed_caches = remove_disposable_caches(current)
    copied = copy_overlay(source_root, current)
    require(copied, "FX03 source overlay is empty")
    receipt = current / "RC_K0/child_designs/fortification/r0c_cp2/reports/CP2C_FX03_SOURCE_CHECKPOINT.json"
    write_json(
        receipt,
        {
            "schema": "royal-capital.fortification.r0c-cp2.cp2c-source-checkpoint/1",
            "status": "PRESERVED_BEFORE_EXACT_RUNTIME_QUALIFICATION",
            "checkpoint": "CP2-C_R0C_CP2_FX03",
            "accepted_parent": {"name": "C2B.zip", "sha256": C2B_SHA256},
            "accepted_in_work_predecessor": "R0C_CP2_FX02",
            "fixed_baseline": {"name": "RCF_D0_full.zip", "sha256": BASELINE_SHA256},
            "source_overlay": source_inventory(source_root),
            "removed_disposable_caches": removed_caches,
            "functional_qualification": "NOT_RUN",
            "partial_output_published": False,
        },
    )
    pt.write_registry(current)
    registry = pt.validate_registry(current)
    require(registry["status"] == "PASS", f"FX03 implementation registry failed: {registry}")
    build_checkpoint(
        label="FX03",
        authority=authority,
        c1_tree=c1_tree,
        parent=parent,
        current=current,
        inputs=inputs,
        work=work,
        output=output,
        copied=copied,
        removed_caches=removed_caches,
    )
    write_json(
        work / "STATE.json",
        {
            "current": current.as_posix(),
            "c1": c1_tree.as_posix(),
            "parent_c2b": parent.as_posix(),
            "output": output.as_posix(),
            "last_preserved": "FX03",
        },
    )
    print(json.dumps({"status": "FX03_PREPARED", "current": current.as_posix()}, sort_keys=True))


def validate_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["fixture_id"]: row for row in rows}
    expected = {
        "R0C_CP2_FX01": "PASS_CP2_B",
        "R0C_CP2_FX02": "PASS_CP2_C",
        "R0C_CP2_FX03": "PASS_CP2_C",
    }
    for fixture_id, status in expected.items():
        require(by_id.get(fixture_id, {}).get("status") == status, f"matrix status mismatch for {fixture_id}")
        require(
            by_id[fixture_id].get("exact_source_binding", "").startswith("BOUND_EXACT_ACCEPTED_IDS_DIGESTS"),
            f"matrix exact binding missing for {fixture_id}",
        )
    return {"status": "PASS", "rows": len(rows), "required": {key: by_id[key] for key in expected}}


def build_receipts(
    current: Path,
    runtime_logs: Path,
    output: Path,
    scratch: Path,
    current_delta_check: dict[str, Any],
) -> dict[str, Any]:
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    reports = current / "RC_K0/child_designs/fortification/r0c_cp2/reports"
    for path in sorted(reports.glob("CP2C*.json")):
        target = scratch / "reports" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ("C2C_FX02_pre_check.json", "C2C_FX03_pre_check.json"):
        source = output / name
        if source.is_file():
            target = scratch / "checkpoints" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if runtime_logs.is_dir():
        for path in sorted(runtime_logs.rglob("*")):
            if path.is_file():
                target = scratch / "runtime_logs" / path.relative_to(runtime_logs)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    write_json(scratch / "C2C_CURRENT_DELTA_CHECK.json", current_delta_check)
    write_json(
        scratch / "FAILED_ATTEMPTS.json",
        {
            "schema": "royal-capital.fortification.r0c-cp2.cp2c-failed-attempts/1",
            "preserved": [
                {
                    "context": "conversation local execution backend",
                    "result": "ClientError before source modification",
                    "recovery": "resumed from accepted C2B through the same repository and exact GitHub Actions runtime",
                    "rollback": False,
                    "accepted_parent_deleted": False,
                },
                {
                    "context": "initial branch creation",
                    "result": "HTTP_422_WRONG_TREE_SHA_USED_AS_COMMIT_SHA",
                    "recovery": "created branch from exact accepted C2B commit 9e812701c99eb80fabe0a0fcccd38fc3332a35a4",
                    "rollback": False,
                },
            ],
        },
    )
    pt.deterministic_zip(scratch, output / "C2CR.zip")
    return artifact(output / "C2CR.zip")


def finalize(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs).resolve()
    work = Path(args.work).resolve()
    output = Path(args.output).resolve()
    runtime_logs = Path(args.runtime_logs).resolve()
    authority = validate_parent_inputs(inputs)
    current = work / "current"
    c1_tree = work / "c1"
    parent = work / "parent_c2b"
    require(current.is_dir() and c1_tree.is_dir() and parent.is_dir(), "prepared CP2-C tree is missing")
    reports = current / "RC_K0/child_designs/fortification/r0c_cp2/reports"
    fx02_q = json.loads((reports / "CP2C_FX02_qualification.json").read_text(encoding="utf-8"))
    fx03_q = json.loads((reports / "CP2C_FX03_qualification.json").read_text(encoding="utf-8"))
    family = json.loads((reports / "CP2C_FAMILY_MATRIX_qualification.json").read_text(encoding="utf-8"))
    fx02_result = json.loads((current / "RC_K0/child_designs/fortification/r0c_cp2/outputs/fx02/reference/result.json").read_text(encoding="utf-8"))
    fx03_result = json.loads((current / "RC_K0/child_designs/fortification/r0c_cp2/outputs/fx03/reference/result.json").read_text(encoding="utf-8"))
    fx02_clean = json.loads((reports / "CP2C_FX02_clean_replay.json").read_text(encoding="utf-8"))
    fx03_clean = json.loads((reports / "CP2C_FX03_clean_replay.json").read_text(encoding="utf-8"))
    require(fx02_q.get("status") == "PASS" and fx02_q.get("stage_completion") == "CP2_C_FX02_COMPLETE_CLOSED", "FX02 qualification mismatch")
    require(fx03_q.get("status") == "PASS" and fx03_q.get("stage_completion") == "CP2_C_COMPLETE_CLOSED", "FX03 qualification mismatch")
    require(family.get("status") == "PASS" and family.get("stage_completion") == "CP2_C_COMPLETE_CLOSED", "family matrix aggregate mismatch")
    for fixture_id, result, clean in (
        ("R0C_CP2_FX02", fx02_result, fx02_clean),
        ("R0C_CP2_FX03", fx03_result, fx03_clean),
    ):
        require(result.get("status") == "SUCCEEDED", f"{fixture_id} result not SUCCEEDED")
        require(result.get("accepted_sources_unchanged") is True, f"{fixture_id} mutated accepted source")
        require(result.get("stored_copies_reopened") is True, f"{fixture_id} stored copies not reopened")
        require(result.get("partial_output_published") is False, f"{fixture_id} partial output flag changed")
        require(clean.get("byte_identical") is True, f"{fixture_id} clean replay not byte-identical")
    matrix = validate_matrix(current / "RC_K0/child_designs/fortification/r0c_cp2/data/CP2_FIXTURE_MATRIX.csv")
    removed_caches = remove_disposable_caches(current)
    completion = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2c-completion/1",
        "status": "PASS",
        "checkpoint": "CP2-C",
        "completion": "CP2_C_COMPLETE_CLOSED",
        "source_preservation": "PASS",
        "functional_qualification": "PASS_FX02_FX03",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "required_family_matrix": matrix,
        "continuity": family["continuity"],
        "clean_replay": family["clean_replay"],
        "negative_gates": "DEFERRED_TO_CP2_D",
        "next_exact_checkpoint": "R0C-CP2 CP2-D",
    }
    write_json(reports / "CP2C_COMPLETION.json", completion)
    pt.write_registry(current)
    current_registry = pt.validate_registry(current)
    require(current_registry["status"] == "PASS", f"final registry failed: {current_registry}")

    full = output / "C2C.zip"
    patch = output / "C2CP.zip"
    source = output / "C2CS.zip"
    current_delta = work / "C2C_current_delta.zip"
    source_meta = pt.create_source_delta(c1_tree, current, source, parent_name="C1.zip", stage=STAGE)
    current_delta_meta = pt.create_source_delta(parent, current, current_delta, parent_name="C2B.zip", stage=STAGE + "_CURRENT_DELTA")
    patch_meta = build_cumulative_patch(
        inputs / "C2BP.zip",
        current_delta,
        patch,
        work / "final_patch_build",
        stage=STAGE,
    )
    pt.deterministic_zip(current, full)
    full_reopen = pt.verify_zip_tree(full, current, work / "verify_final_full")
    c1_plus_source = pt.verify_parent_plus_delta(c1_tree, source, current, work / "verify_final_source")
    c2b_plus_delta = pt.verify_parent_plus_delta(parent, current_delta, current, work / "verify_final_current")
    patch_payload = pt.verify_patch_payload(patch, current, source, work / "verify_final_patch")
    for title, value in {
        "full reopen": full_reopen,
        "C1 + source": c1_plus_source,
        "C2B + current delta": c2b_plus_delta,
        "patch payload": patch_payload,
    }.items():
        require(value["status"] == "PASS", f"final {title} failed: {value}")
    current_delta_check = {
        "status": "PASS",
        "source_delta": current_delta_meta,
        "C2B_plus_current_delta_equals_C2C": c2b_plus_delta,
        "accepted_C2BP_plus_current_delta_constructed_C2CP": {
            "status": "PASS",
            "C2BP_sha256": C2BP_SHA256,
            "current_delta_sha256": sha256(current_delta),
            "C2CP_sha256": sha256(patch),
        },
    }
    receipt_info = build_receipts(current, runtime_logs, output, work / "receipts", current_delta_check)
    compositional = {
        "status": "PASS_COMPOSITIONAL_WITH_ACCEPTED_C2B_REPLAY",
        "premises": {
            "RCF_D0_plus_C2BP_equals_C2B": authority["C2B_check"]["validation"]["fixed_baseline_plus_cumulative_patch_equals_full"],
            "C2B_plus_current_delta_equals_C2C": c2b_plus_delta,
            "C2CP_constructed_from_exact_C2BP_plus_current_delta": current_delta_check["accepted_C2BP_plus_current_delta_constructed_C2CP"],
        },
        "conclusion": "RCF_D0_full.zip + C2CP.zip = C2C.zip",
        "direct_fixed_baseline_replay_this_run": "NOT_REPEATED_ACCEPTED_PARENT_CHAIN_REOPENED",
    }
    artifacts = {
        "C2C.zip": artifact(full),
        "C2CP.zip": artifact(patch),
        "C2CS.zip": artifact(source),
        "C2CR.zip": receipt_info,
    }
    check = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2c-check/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "official_subtitle": "TOWER JOINS",
        "checkpoint": "CP2-C_REQUIRED_FAMILY_MATRIX",
        "source_preservation": "PASS",
        "exact_runtime": "PASS_REUSED_CPYTHON_3_13_5_BUILD123D_E22D34DAE_OCP_8_0_1_0_0",
        "functional_qualification": "PASS_FX02_FX03",
        "stage_completion": "CP2_C_COMPLETE_CLOSED",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "fixture_matrix": matrix,
        "FX02": {"qualification": fx02_q, "result": fx02_result, "clean_replay": fx02_clean},
        "FX03": {"qualification": fx03_q, "result": fx03_result, "clean_replay": fx03_clean},
        "family_matrix": family,
        "removed_disposable_caches": removed_caches,
        "registries": {
            "C1": pt.validate_registry(c1_tree),
            "C2B": pt.validate_registry(parent),
            "C2C": current_registry,
        },
        "source_delta": source_meta,
        "current_delta": current_delta_meta,
        "cumulative_patch": patch_meta,
        "artifacts": artifacts,
        "validation": {
            "status": "PASS",
            "full_reopen": full_reopen,
            "C1_plus_source_equals_full": c1_plus_source,
            "C2B_plus_current_delta_equals_full": c2b_plus_delta,
            "patch_current_delta_equals_source_payload": patch_payload,
            "fixed_baseline_plus_cumulative_patch_equals_full": compositional,
        },
        "negative_gates": "DEFERRED_TO_CP2_D",
        "roadmap_progress": {
            "r0c": "1/4",
            "accepted_checkpoints": "10/25",
            "subcheckpoint": "CP2-C_COMPLETE_CLOSED",
            "next_exact_checkpoint": "R0C-CP2 CP2-D",
        },
    }
    check_path = output / "C2C_check.json"
    write_json(check_path, check)
    report = f"""# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — CP2-C

## 판정

```text
source preservation                 PASS
exact runtime reuse                 PASS
functional qualification            PASS_FX02_FX03
stage completion                    CP2_C_COMPLETE_CLOSED
R0C-CP2                             OPEN
Godot product                       NOT_STARTED
```

## Required family matrix

```text
R0C_CP2_FX01  ROUND      TANGENT           PASS_CP2_B
R0C_CP2_FX02  SQUARE     CORNER            PASS_CP2_C
R0C_CP2_FX03  POLYGONAL  WALL_PENETRATING  PASS_CP2_C
```

## Continuity and replay

```text
wall body continuity                PASS_ALL_REQUIRED_FIXTURES
wall-walk continuity                PASS_ALL_REQUIRED_FIXTURES
foundation continuity               PASS_ALL_REQUIRED_FIXTURES
FX02 clean A/B                      {str(fx02_clean['byte_identical']).lower()} ({fx02_clean['files_compared']} files)
FX03 clean A/B                      {str(fx03_clean['byte_identical']).lower()} ({fx03_clean['files_compared']} files)
accepted sources unchanged          true
stored STEP/BREP reopened           true
negative gates                      DEFERRED_TO_CP2_D
```

CP2-C closes only the required family-matrix unit. R0C-CP2 remains open for CP2-D negative gates, no-partial-output and closeout.
"""
    report_path = output / "C2C_report.md"
    report_path.write_text(report, encoding="utf-8")
    delivery = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2c-delivery/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "checkpoint": "CP2-C_REQUIRED_FAMILY_MATRIX",
        "status": "PASS",
        "artifacts": artifacts,
        "sidecars": {
            "C2C_check.json": {"bytes": check_path.stat().st_size, "sha256": sha256(check_path)},
            "C2C_report.md": {"bytes": report_path.stat().st_size, "sha256": sha256(report_path)},
        },
        "next": "R0C-CP2 CP2-D",
    }
    delivery_path = output / "C2C_delivery.json"
    write_json(delivery_path, delivery)
    names = [
        "C2C.zip",
        "C2CP.zip",
        "C2CS.zip",
        "C2CR.zip",
        "C2C_check.json",
        "C2C_report.md",
        "C2C_delivery.json",
        "C2C_FX02_pre.zip",
        "C2C_FX02_preP.zip",
        "C2C_FX02_preS.zip",
        "C2C_FX02_pre_check.json",
        "C2C_FX03_pre.zip",
        "C2C_FX03_preP.zip",
        "C2C_FX03_preS.zip",
        "C2C_FX03_pre_check.json",
    ]
    for name in names:
        require((output / name).is_file(), f"delivery member missing: {name}")
    (output / "C2C.sha256").write_text(
        "\n".join(f"{sha256(output / name)}  {name}" for name in names) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "artifacts": artifacts}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--inputs", required=True)
    common.add_argument("--work", required=True)
    common.add_argument("--output", required=True)
    fx02 = sub.add_parser("prepare-fx02", parents=[common])
    fx02.add_argument("--source-root", required=True)
    fx02.set_defaults(function=prepare_fx02)
    fx03 = sub.add_parser("prepare-fx03", parents=[common])
    fx03.add_argument("--source-root", required=True)
    fx03.set_defaults(function=prepare_fx03)
    final = sub.add_parser("finalize", parents=[common])
    final.add_argument("--runtime-logs", required=True)
    final.set_defaults(function=finalize)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
