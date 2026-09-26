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

BASELINE_SHA256 = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
C1_SHA256 = "91cf2025f311772b0a137d3509efa0b1d65b96832a78f1e4f46ba9cf3b06e878"
C1P_SHA256 = "dddb960aa467689eecf1e312724ce06c0c9175e0fdc902c534c4e12ffdca77b4"
C2A_SHA256 = "0f369e88861705df0dc7edd0a1e3505bcaf70878a57b5c7431ad5a4349f3d66e"
C2AP_SHA256 = "db69f0d4b5000bba046704d60d8e6f01eac2e1ed0e42be639951f360e35dad94"
C2A_CHECK_SHA256 = "030e094f4c63eed7cf443c271e5e1698b71d7e85228ab798a0d8644f3dd5ac52"
STAGE = "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2_CP2-B_R0C_CP2_FX01"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return pt.sha256_path(path)


def copy_overlay(source_root: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(source_root.rglob("*")):
        rel = source.relative_to(source_root)
        target = destination / rel
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel.as_posix())
    return copied


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


def artifact(path: Path) -> dict[str, Any]:
    return cp2a.artifact(path)


def build_cumulative_patch(
    parent_patch: Path,
    current_delta: Path,
    destination: Path,
    scratch: Path,
    *,
    parent_full_name: str,
    parent_full_sha256: str,
    parent_patch_name: str,
    parent_patch_sha256: str,
) -> dict[str, Any]:
    shutil.rmtree(scratch, ignore_errors=True)
    pt.safe_extract(parent_patch, scratch)
    deleted_paths: set[str] = set()
    d_file = scratch / "D.txt"
    if d_file.is_file():
        deleted_paths.update(
            line.strip()
            for line in d_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    pt._overlay_source_delta_into_patch(scratch, current_delta, deleted_paths)
    d_file.write_text(
        "\n".join(sorted(deleted_paths)) + ("\n" if deleted_paths else ""),
        encoding="utf-8",
    )
    metadata = {
        "schema": "royal-capital.cumulative-patch/1",
        "stage": STAGE,
        "baseline_file": "RCF_D0_full.zip",
        "baseline_sha256": BASELINE_SHA256,
        "accepted_parent_full": parent_full_name,
        "accepted_parent_sha256": parent_full_sha256,
        "accepted_parent_cumulative_patch": parent_patch_name,
        "accepted_parent_cumulative_patch_sha256": parent_patch_sha256,
        "current_delta": current_delta.name,
        "deleted_count": len(deleted_paths),
        "proof": (
            "accepted fixed-baseline replay reaches the accepted parent; "
            "the exact current delta reaches the CP2-B checkpoint; overlaying the same delta "
            "on the accepted parent cumulative patch reconstructs the CP2-B checkpoint"
        ),
    }
    write_json(scratch / "PATCH_METADATA.json", metadata)
    pt.deterministic_zip(scratch, destination)
    return metadata


def validate_parent_inputs(inputs: Path) -> dict[str, Any]:
    expected = {
        "C1.zip": C1_SHA256,
        "C1P.zip": C1P_SHA256,
        "C2A.zip": C2A_SHA256,
        "C2AP.zip": C2AP_SHA256,
        "C2A_check.json": C2A_CHECK_SHA256,
    }
    observed: dict[str, Any] = {}
    for name, digest in expected.items():
        path = inputs / name
        require(path.is_file(), f"missing required accepted input: {name}")
        actual = sha256(path)
        require(actual == digest, f"{name} SHA mismatch {actual} != {digest}")
        observed[name] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
            "zip": artifact(path) if path.suffix == ".zip" else None,
        }
    check = json.loads((inputs / "C2A_check.json").read_text(encoding="utf-8"))
    require(check.get("stage_completion") == "CP2_A_COMPLETE_CLOSED", "accepted C2A stage is not closed")
    require(check.get("r0c_cp2") == "OPEN", "accepted C2A incorrectly closes R0C-CP2")
    require(check.get("validation", {}).get("status") == "PASS", "accepted C2A validation is not PASS")
    require(
        check["validation"]["fixed_baseline_plus_cumulative_patch_equals_full"]["status"]
        == "PASS_COMPOSITIONAL_WITH_ACCEPTED_DIRECT_REPLAY",
        "accepted C2A fixed-baseline proof is not admitted",
    )
    return {"files": observed, "C2A_check": check}


def prepare(args: argparse.Namespace) -> None:
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
    parent = work / "parent_c2a"
    current = work / "current"
    pt.safe_extract(inputs / "C1.zip", c1_tree)
    pt.safe_extract(inputs / "C2A.zip", parent)
    shutil.copytree(parent, current, copy_function=shutil.copy2)

    c1_registry = pt.validate_registry(c1_tree)
    parent_registry = pt.validate_registry(parent)
    require(c1_registry["status"] == "PASS", f"C1 registry failed: {c1_registry}")
    require(c1_registry["tree_files_including_registry"] == 1418, f"C1 count mismatch: {c1_registry}")
    require(parent_registry["status"] == "PASS", f"C2A registry failed: {parent_registry}")
    require(parent_registry["tree_files_including_registry"] == 1421, f"C2A count mismatch: {parent_registry}")

    copied = copy_overlay(source_root, current)
    require(copied, "CP2-B source overlay is empty")
    receipt_path = current / "RC_K0/child_designs/fortification/r0c_cp2/reports/CP2B_IMPLEMENTATION_SOURCE_CHECKPOINT.json"
    write_json(
        receipt_path,
        {
            "schema": "royal-capital.fortification.r0c-cp2.cp2b-source-checkpoint/1",
            "status": "PRESERVED_BEFORE_EXACT_RUNTIME_QUALIFICATION",
            "checkpoint": "CP2-B_R0C_CP2_FX01",
            "accepted_parent": {"name": "C2A.zip", "sha256": C2A_SHA256},
            "fixed_baseline": {"name": "RCF_D0_full.zip", "sha256": BASELINE_SHA256},
            "source_overlay": source_inventory(source_root),
            "functional_qualification": "NOT_RUN",
            "partial_output_published": False,
        },
    )
    pt.write_registry(current)
    current_registry = pt.validate_registry(current)
    require(current_registry["status"] == "PASS", f"implementation registry failed: {current_registry}")

    pre_full = output / "C2B_pre.zip"
    pre_patch = output / "C2B_preP.zip"
    pre_source = output / "C2B_preS.zip"
    source_meta = pt.create_source_delta(
        c1_tree,
        current,
        pre_source,
        parent_name="C1.zip",
        stage=STAGE + "_PREQUALIFICATION",
    )
    patch_meta = build_cumulative_patch(
        inputs / "C1P.zip",
        pre_source,
        pre_patch,
        work / "pre_patch_build",
        parent_full_name="C1.zip",
        parent_full_sha256=C1_SHA256,
        parent_patch_name="C1P.zip",
        parent_patch_sha256=C1P_SHA256,
    )
    pt.deterministic_zip(current, pre_full)

    full_reopen = pt.verify_zip_tree(pre_full, current, work / "verify_pre_full")
    c1_plus_source = pt.verify_parent_plus_delta(c1_tree, pre_source, current, work / "verify_pre_source")
    patch_payload = pt.verify_patch_payload(pre_patch, current, pre_source, work / "verify_pre_patch")
    require(full_reopen["status"] == "PASS", f"pre full reopen failed: {full_reopen}")
    require(c1_plus_source["status"] == "PASS", f"pre C1+source failed: {c1_plus_source}")
    require(patch_payload["status"] == "PASS", f"pre patch payload failed: {patch_payload}")

    pre_check = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2b-pre-checkpoint/1",
        "status": "PRESERVED_BEFORE_EXACT_RUNTIME_QUALIFICATION",
        "checkpoint": "CP2-B_R0C_CP2_FX01",
        "authority": authority,
        "registries": {"C1": c1_registry, "C2A": parent_registry, "implementation": current_registry},
        "source_overlay_paths": copied,
        "source_delta": source_meta,
        "cumulative_patch": patch_meta,
        "artifacts": {
            "full": artifact(pre_full),
            "patch": artifact(pre_patch),
            "source": artifact(pre_source),
        },
        "equality": {
            "full_reopen": full_reopen,
            "C1_plus_source": c1_plus_source,
            "patch_payload": patch_payload,
        },
        "functional_qualification": "NOT_RUN",
        "stage_completion": "OPEN",
    }
    write_json(output / "C2B_pre_check.json", pre_check)
    write_json(
        work / "STATE.json",
        {
            "current": current.as_posix(),
            "c1": c1_tree.as_posix(),
            "parent_c2a": parent.as_posix(),
            "output": output.as_posix(),
        },
    )
    print(json.dumps({"status": "PREPARED", "current": current.as_posix()}, sort_keys=True))


def validate_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["fixture_id"]: row for row in rows}
    require(by_id["R0C_CP2_FX01"]["status"] == "PASS_CP2_B", "FX01 matrix row is not PASS_CP2_B")
    require(
        by_id["R0C_CP2_FX01"]["exact_source_binding"] == "BOUND_EXACT_ACCEPTED_IDS_DIGESTS_AND_SOCKETS",
        "FX01 exact source binding is not frozen",
    )
    for fixture_id in ("R0C_CP2_FX02", "R0C_CP2_FX03"):
        require(by_id[fixture_id]["status"] == "FROZEN_PENDING_IMPLEMENTATION", f"{fixture_id} status changed prematurely")
    return {
        "status": "PASS",
        "rows": len(rows),
        "FX01": by_id["R0C_CP2_FX01"],
        "remaining": [by_id["R0C_CP2_FX02"], by_id["R0C_CP2_FX03"]],
    }


def build_receipts(current: Path, runtime_logs: Path, output_path: Path, scratch: Path, current_delta_check: dict[str, Any]) -> dict[str, Any]:
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    reports = current / "RC_K0/child_designs/fortification/r0c_cp2/reports"
    for path in sorted(reports.glob("*.json")):
        target = scratch / "reports" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if runtime_logs.is_dir():
        for path in sorted(runtime_logs.rglob("*")):
            if path.is_file():
                target = scratch / "runtime_logs" / path.relative_to(runtime_logs)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    write_json(scratch / "C2B_CURRENT_DELTA_CHECK.json", current_delta_check)
    write_json(
        scratch / "FAILED_ATTEMPTS.json",
        {
            "schema": "royal-capital.fortification.r0c-cp2.cp2b-failed-attempts/1",
            "preserved": [
                {
                    "context": "conversation local runtime",
                    "result": "ClientError before source modification",
                    "recovery": "continued from accepted C2A through exact GitHub Actions runtime",
                    "rollback": False,
                },
                {
                    "context": "source inspection workflow",
                    "run_id": 36215380245,
                    "result": "PASS_INSPECTION_ONLY",
                    "product_publication": False,
                },
            ],
        },
    )
    pt.deterministic_zip(scratch, output_path)
    return artifact(output_path)


def finalize(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs).resolve()
    work = Path(args.work).resolve()
    output = Path(args.output).resolve()
    runtime_logs = Path(args.runtime_logs).resolve()
    authority = validate_parent_inputs(inputs)
    current = work / "current"
    c1_tree = work / "c1"
    parent = work / "parent_c2a"
    require(current.is_dir() and c1_tree.is_dir() and parent.is_dir(), "prepared work tree is missing")

    qualification_path = current / "RC_K0/child_designs/fortification/r0c_cp2/reports/CP2B_FX01_qualification.json"
    result_path = current / "RC_K0/child_designs/fortification/r0c_cp2/outputs/fx01/reference/result.json"
    clean_path = current / "RC_K0/child_designs/fortification/r0c_cp2/reports/CP2B_FX01_clean_replay.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    require(qualification.get("status") == "PASS", "FX01 qualification is not PASS")
    require(qualification.get("stage_completion") == "CP2_B_COMPLETE_CLOSED", "CP2-B completion mismatch")
    require(result.get("status") == "SUCCEEDED", "FX01 result is not SUCCEEDED")
    require(clean.get("byte_identical") is True, "FX01 clean replay is not byte-identical")
    require(result.get("accepted_sources_unchanged") is True, "accepted sources were mutated")
    require(result.get("stored_copies_reopened") is True, "stored copies were not reopened")
    require(result.get("partial_output_published") is False, "partial output flag changed")
    require(result.get("foundation_source_offset_m") == 0.5, "foundation offset evidence mismatch")
    require(result.get("foundation_transition_residual_m") == 0.0, "foundation residual mismatch")
    require(result.get("socket_position_error_m") == 0.0, "socket position error mismatch")
    require(result.get("tangent_error") == 0.0, "tangent error mismatch")
    require(result.get("up_axis_error") == 0.0, "up error mismatch")
    require(result.get("inside_frame_error") == 0.0, "inside error mismatch")
    matrix = validate_matrix(current / "RC_K0/child_designs/fortification/r0c_cp2/data/CP2_FIXTURE_MATRIX.csv")

    pt.write_registry(current)
    current_registry = pt.validate_registry(current)
    require(current_registry["status"] == "PASS", f"final registry failed: {current_registry}")

    final_full = output / "C2B.zip"
    final_patch = output / "C2BP.zip"
    final_source = output / "C2BS.zip"
    current_delta = work / "C2B_current_delta.zip"
    final_receipts = output / "C2BR.zip"

    source_meta = pt.create_source_delta(
        c1_tree,
        current,
        final_source,
        parent_name="C1.zip",
        stage=STAGE,
    )
    current_delta_meta = pt.create_source_delta(
        parent,
        current,
        current_delta,
        parent_name="C2A.zip",
        stage=STAGE + "_CURRENT_DELTA",
    )
    patch_meta = build_cumulative_patch(
        inputs / "C2AP.zip",
        current_delta,
        final_patch,
        work / "final_patch_build",
        parent_full_name="C2A.zip",
        parent_full_sha256=C2A_SHA256,
        parent_patch_name="C2AP.zip",
        parent_patch_sha256=C2AP_SHA256,
    )
    pt.deterministic_zip(current, final_full)

    full_reopen = pt.verify_zip_tree(final_full, current, work / "verify_final_full")
    c1_plus_source = pt.verify_parent_plus_delta(c1_tree, final_source, current, work / "verify_final_source")
    c2a_plus_delta = pt.verify_parent_plus_delta(parent, current_delta, current, work / "verify_current_delta")
    patch_payload = pt.verify_patch_payload(final_patch, current, final_source, work / "verify_final_patch")
    require(full_reopen["status"] == "PASS", f"final full reopen failed: {full_reopen}")
    require(c1_plus_source["status"] == "PASS", f"C1+source failed: {c1_plus_source}")
    require(c2a_plus_delta["status"] == "PASS", f"C2A+delta failed: {c2a_plus_delta}")
    require(patch_payload["status"] == "PASS", f"patch payload failed: {patch_payload}")

    current_delta_check = {
        "status": "PASS",
        "source_delta": current_delta_meta,
        "C2A_plus_current_delta_equals_C2B": c2a_plus_delta,
        "accepted_C2AP_plus_current_delta_constructed_C2BP": {
            "status": "PASS",
            "C2AP_sha256": C2AP_SHA256,
            "current_delta_sha256": sha256(current_delta),
            "C2BP_sha256": sha256(final_patch),
        },
    }
    receipt_info = build_receipts(current, runtime_logs, final_receipts, work / "receipts", current_delta_check)

    compositional = {
        "status": "PASS_COMPOSITIONAL_WITH_ACCEPTED_CP2A_REPLAY",
        "premises": {
            "RCF_D0_plus_C2AP_equals_C2A": authority["C2A_check"]["validation"]["fixed_baseline_plus_cumulative_patch_equals_full"],
            "C2A_plus_current_delta_equals_C2B": c2a_plus_delta,
            "C2BP_constructed_from_exact_C2AP_plus_current_delta": current_delta_check["accepted_C2AP_plus_current_delta_constructed_C2BP"],
        },
        "conclusion": "RCF_D0_full.zip + C2BP.zip = C2B.zip",
        "direct_fixed_baseline_replay_this_run": "NOT_REPEATED_ACCEPTED_CP2A_REPLAY_REOPENED",
    }

    artifacts = {
        "C2B.zip": artifact(final_full),
        "C2BP.zip": artifact(final_patch),
        "C2BS.zip": artifact(final_source),
        "C2BR.zip": receipt_info,
    }
    check = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2b-fx01-check/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "official_subtitle": "TOWER JOINS",
        "checkpoint": "CP2-B_R0C_CP2_FX01",
        "source_preservation": "PASS",
        "exact_runtime": "PASS_REUSED_CPYTHON_3_13_5_BUILD123D_E22D34DAE_OCP_8_0_1_0_0",
        "functional_qualification": "PASS_FX01",
        "stage_completion": "CP2_B_COMPLETE_CLOSED",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "fixture_matrix": matrix,
        "qualification": qualification,
        "result": result,
        "clean_replay": clean,
        "registries": {
            "C1": pt.validate_registry(c1_tree),
            "C2A": pt.validate_registry(parent),
            "C2B": current_registry,
        },
        "source_delta": source_meta,
        "current_delta": current_delta_meta,
        "cumulative_patch": patch_meta,
        "artifacts": artifacts,
        "validation": {
            "status": "PASS",
            "full_reopen": full_reopen,
            "C1_plus_source_equals_full": c1_plus_source,
            "C2A_plus_current_delta_equals_full": c2a_plus_delta,
            "patch_current_delta_equals_source_payload": patch_payload,
            "fixed_baseline_plus_cumulative_patch_equals_full": compositional,
        },
        "roadmap_progress": {
            "r0c": "1/4",
            "accepted_checkpoints": "10/25",
            "subcheckpoint": "CP2-B_COMPLETE_CLOSED",
            "next_exact_checkpoint": "R0C-CP2 CP2-C",
        },
    }
    check_path = output / "C2B_check.json"
    write_json(check_path, check)

    report = f"""# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — CP2-B R0C_CP2_FX01

## 판정

```text
source preservation                 PASS
exact runtime reuse                 PASS
functional qualification            PASS_FX01
stage completion                    CP2_B_COMPLETE_CLOSED
R0C-CP2                             OPEN
Godot product                       NOT_STARTED
```

## 구현

```text
tower family                        ROUND
join kind                           TANGENT
span family                         STRAIGHT_SPAN
realization                         BOUNDED_TRANSITION_PIECE
span translation                    [7.45359421, 0.0, 0.0] m
socket position error               {result['socket_position_error_m']} m
tangent / up / inside error         {result['tangent_error']} / {result['up_axis_error']} / {result['inside_frame_error']}
wall-walk position error            {result['wall_walk_position_error_m']} m
foundation source offset            {result['foundation_source_offset_m']} m
foundation residual                 {result['foundation_transition_residual_m']} m
unsupported gap                     {result['unsupported_gap_m']} m
outside / inside projection         {result['outside_projection_m']} / {result['inside_projection_m']} m
stored copies reopened              {str(result['stored_copies_reopened']).lower()}
accepted sources unchanged          {str(result['accepted_sources_unchanged']).lower()}
clean replay byte-identical         {str(clean['byte_identical']).lower()}
files compared                      {clean['files_compared']}
```

CP2-B only accepts FX01. FX02 and FX03 remain pending for CP2-C. R0C-CP2 is not closed.
"""
    report_path = output / "C2B_report.md"
    report_path.write_text(report, encoding="utf-8")

    delivery = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2b-delivery/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "checkpoint": "CP2-B_R0C_CP2_FX01",
        "status": "PASS",
        "artifacts": artifacts,
        "sidecars": {
            "C2B_check.json": {"bytes": check_path.stat().st_size, "sha256": sha256(check_path)},
            "C2B_report.md": {"bytes": report_path.stat().st_size, "sha256": sha256(report_path)},
        },
        "next": "R0C-CP2 CP2-C",
    }
    delivery_path = output / "C2B_delivery.json"
    write_json(delivery_path, delivery)

    sha_names = [
        "C2B.zip",
        "C2BP.zip",
        "C2BS.zip",
        "C2BR.zip",
        "C2B_check.json",
        "C2B_report.md",
        "C2B_delivery.json",
        "C2B_pre.zip",
        "C2B_preP.zip",
        "C2B_preS.zip",
        "C2B_pre_check.json",
    ]
    rows = [f"{sha256(output / name)}  {name}" for name in sha_names]
    (output / "C2B.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for name in sha_names:
        require((output / name).is_file(), f"final delivery member missing: {name}")
    print(json.dumps({"status": "PASS", "artifacts": artifacts}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--inputs", required=True)
    common.add_argument("--work", required=True)
    common.add_argument("--output", required=True)

    prepare_parser = sub.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--source-root", required=True)
    prepare_parser.set_defaults(function=prepare)

    finalize_parser = sub.add_parser("finalize", parents=[common])
    finalize_parser.add_argument("--runtime-logs", required=True)
    finalize_parser.set_defaults(function=finalize)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
