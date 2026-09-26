from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "r0c_cp1"))
import package_tools as pt  # noqa: E402

BASELINE_SHA256 = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
C1_SHA256 = "91cf2025f311772b0a137d3509efa0b1d65b96832a78f1e4f46ba9cf3b06e878"
C1P_SHA256 = "dddb960aa467689eecf1e312724ce06c0c9175e0fdc902c534c4e12ffdca77b4"
C2C_SHA256 = "f9c40a68a37f71d8a3e24288f66559481c212ededca6842d3ef555c6f275369f"
C2CP_SHA256 = "ed81630e5a655e32235820f112b7cad1c8c3ddece532ac0bfc1963802cafffee"
C2CS_SHA256 = "fa6b1cbde6faa069637e8a1be02636c161480a8f34dab2734bfbbf9d79279616"
C2C_CHECK_SHA256 = "80f83e2b3dcf987fdfce60e81ed8415b9021e1f645f22dc970574df775bec755"
STAGE = "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2_CP2-D_CLOSEOUT"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return pt.sha256_path(path)


def remove_disposable_caches(root: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts):
            removed.append(path.relative_to(root).as_posix())
            path.unlink()
    for directory in sorted((path for path in root.rglob("__pycache__") if path.is_dir()), reverse=True):
        shutil.rmtree(directory, ignore_errors=True)
    return sorted(removed)


def copy_overlay(source_root: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        rel = source.relative_to(source_root)
        target = destination / rel
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
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def zip_artifact(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix != ".zip":
        return result
    duplicates: list[str] = []
    unsafe: list[str] = []
    symlinks: list[str] = []
    seen: set[str] = set()
    with zipfile.ZipFile(path) as zf:
        bad_crc = zf.testzip()
        infos = zf.infolist()
        for info in infos:
            if info.filename in seen:
                duplicates.append(info.filename)
            seen.add(info.filename)
            if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                unsafe.append(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                symlinks.append(info.filename)
    result.update(
        {
            "entries": len(infos),
            "bad_crc": bad_crc,
            "duplicates": duplicates,
            "unsafe": unsafe,
            "symlinks": symlinks,
            "status": "PASS" if bad_crc is None and not duplicates and not unsafe and not symlinks else "FAIL",
        }
    )
    return result


def validate_inputs(inputs: Path) -> dict[str, Any]:
    expected = {
        "RCF_D0_full.zip": BASELINE_SHA256,
        "C1.zip": C1_SHA256,
        "C1P.zip": C1P_SHA256,
        "C2C.zip": C2C_SHA256,
        "C2CP.zip": C2CP_SHA256,
        "C2CS.zip": C2CS_SHA256,
        "C2C_check.json": C2C_CHECK_SHA256,
    }
    observed: dict[str, Any] = {}
    for name, digest in expected.items():
        path = inputs / name
        require(path.is_file(), f"missing accepted input: {name}")
        actual = sha256(path)
        require(actual == digest, f"{name} SHA mismatch {actual} != {digest}")
        observed[name] = zip_artifact(path)
        if path.suffix == ".zip":
            require(observed[name]["status"] == "PASS", f"{name} ZIP admission failed: {observed[name]}")
    check = json.loads((inputs / "C2C_check.json").read_text(encoding="utf-8"))
    require(check.get("source_preservation") == "PASS", "accepted C2C source preservation missing")
    require(check.get("functional_qualification") == "PASS_FX02_FX03", "accepted C2C qualification mismatch")
    require(check.get("stage_completion") == "CP2_C_COMPLETE_CLOSED", "accepted C2C stage is not closed")
    require(check.get("r0c_cp2") == "OPEN", "accepted C2C incorrectly closes R0C-CP2")
    require(check.get("validation", {}).get("status") == "PASS", "accepted C2C validation is not PASS")
    return {"files": observed, "C2C_check": check}


def create_direct_patch(
    baseline: Path,
    current: Path,
    destination: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    diff = pt.tree_diff(baseline, current)
    root = destination.with_suffix(".tree")
    shutil.rmtree(root, ignore_errors=True)
    write_root = root / "W"
    write_root.mkdir(parents=True)
    for rel in diff["added"] + diff["modified"]:
        source = current / rel
        target = write_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (root / "D.txt").write_text(
        "\n".join(diff["deleted"]) + ("\n" if diff["deleted"] else ""),
        encoding="utf-8",
    )
    metadata = {
        "schema": "royal-capital.cumulative-patch/1",
        "stage": stage,
        "baseline_file": "RCF_D0_full.zip",
        "baseline_sha256": BASELINE_SHA256,
        "added": len(diff["added"]),
        "modified": len(diff["modified"]),
        "deleted": len(diff["deleted"]),
        "payload_entries": len(diff["added"]) + len(diff["modified"]),
        "proof": "direct fixed-baseline tree diff; replay is verified against the exact final tree",
    }
    write_json(root / "PATCH_METADATA.json", metadata)
    pt.deterministic_zip(root, destination)
    shutil.rmtree(root)
    return {**metadata, "paths": diff}


def apply_direct_patch(baseline: Path, patch: Path, destination: Path) -> dict[str, Any]:
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(baseline, destination, copy_function=shutil.copy2)
    extracted = destination.with_name(destination.name + ".patch")
    shutil.rmtree(extracted, ignore_errors=True)
    pt.safe_extract(patch, extracted)
    deleted_file = extracted / "D.txt"
    deleted = [
        line.strip()
        for line in deleted_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if deleted_file.is_file() else []
    for rel in deleted:
        target = destination / rel
        if target.is_file() or target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    write_root = extracted / "W"
    require(write_root.is_dir(), f"patch is missing W/: {patch}")
    overlay = 0
    for source in sorted(write_root.rglob("*")):
        rel = source.relative_to(write_root)
        target = destination / rel
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        overlay += 1
    shutil.rmtree(extracted)
    return {"overlay_files": overlay, "deleted_paths": deleted}


def verify_baseline_plus_patch(
    baseline: Path,
    patch: Path,
    expected: Path,
    scratch: Path,
) -> dict[str, Any]:
    replay = apply_direct_patch(baseline, patch, scratch)
    comparison = pt.compare_trees(expected, scratch)
    comparison["replay"] = replay
    comparison["baseline_sha256"] = BASELINE_SHA256
    comparison["patch_sha256"] = sha256(patch)
    shutil.rmtree(scratch)
    return comparison


def build_checkpoint(
    *,
    authority: dict[str, Any],
    baseline: Path,
    c1: Path,
    parent: Path,
    current: Path,
    output: Path,
    work: Path,
    copied: list[str],
    removed_caches: list[str],
) -> dict[str, Any]:
    full = output / "C2_pre.zip"
    patch = output / "C2_preP.zip"
    source = output / "C2_preS.zip"
    source_meta = pt.create_source_delta(c1, current, source, parent_name="C1.zip", stage=STAGE + "_PREVALIDATION")
    patch_meta = create_direct_patch(baseline, current, patch, stage=STAGE + "_PREVALIDATION")
    pt.deterministic_zip(current, full)
    full_reopen = pt.verify_zip_tree(full, current, work / "verify_pre_full")
    baseline_replay = verify_baseline_plus_patch(baseline, patch, current, work / "verify_pre_patch")
    c1_replay = pt.verify_parent_plus_delta(c1, source, current, work / "verify_pre_source")
    patch_payload = pt.verify_patch_payload(patch, current, source, work / "verify_pre_payload")
    for name, value in {
        "full reopen": full_reopen,
        "fixed baseline + patch": baseline_replay,
        "C1 + source": c1_replay,
        "patch payload": patch_payload,
    }.items():
        require(value["status"] == "PASS", f"prevalidation {name} failed: {value}")
    artifacts = {name: zip_artifact(output / name) for name in ("C2_pre.zip", "C2_preP.zip", "C2_preS.zip")}
    require(all(row["status"] == "PASS" for row in artifacts.values()), "prevalidation ZIP integrity failed")
    check = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2d-pre-checkpoint/1",
        "status": "PRESERVED_AND_REOPENED_BEFORE_LONG_VALIDATION",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "checkpoint": "CP2-D-R1",
        "authority": authority,
        "source_overlay_paths": copied,
        "removed_disposable_caches": removed_caches,
        "registries": {
            "baseline": pt.validate_registry(baseline),
            "C1": pt.validate_registry(c1),
            "C2C": pt.validate_registry(parent),
            "implementation": pt.validate_registry(current),
        },
        "source_delta": source_meta,
        "cumulative_patch": patch_meta,
        "artifacts": artifacts,
        "validation": {
            "status": "PASS",
            "full_reopen": full_reopen,
            "fixed_baseline_plus_cumulative_patch_equals_full": baseline_replay,
            "C1_plus_source_equals_full": c1_replay,
            "patch_current_delta_equals_source_payload": patch_payload,
        },
        "source_preservation": "PASS",
        "functional_qualification": "NOT_RUN",
        "stage_completion": "OPEN",
        "godot_product": "NOT_STARTED",
    }
    write_json(output / "C2_pre_check.json", check)
    return check


def prepare(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs).resolve()
    source_root = Path(args.source_root).resolve()
    work = Path(args.work).resolve()
    output = Path(args.output).resolve()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    authority = validate_inputs(inputs)
    baseline = work / "baseline"
    c1 = work / "c1"
    parent = work / "parent_c2c"
    current = work / "current"
    pt.safe_extract(inputs / "RCF_D0_full.zip", baseline)
    pt.safe_extract(inputs / "C1.zip", c1)
    pt.safe_extract(inputs / "C2C.zip", parent)
    shutil.copytree(parent, current, copy_function=shutil.copy2)
    baseline_registry = pt.validate_registry(baseline)
    c1_registry = pt.validate_registry(c1)
    parent_registry = pt.validate_registry(parent)
    require(baseline_registry["status"] == "PASS", f"baseline registry failed: {baseline_registry}")
    require(c1_registry["status"] == "PASS", f"C1 registry failed: {c1_registry}")
    require(parent_registry["status"] == "PASS", f"C2C registry failed: {parent_registry}")
    require(c1_registry["tree_files_including_registry"] == 1418, f"C1 file count mismatch: {c1_registry}")
    require(parent_registry["tree_files_including_registry"] == 1554, f"C2C file count mismatch: {parent_registry}")
    removed_caches = remove_disposable_caches(current)
    copied = copy_overlay(source_root, current)
    require(copied, "CP2-D source overlay is empty")
    receipt = current / "RC_K0/child_designs/fortification/r0c_cp2/reports/CP2D_SOURCE_CHECKPOINT.json"
    write_json(
        receipt,
        {
            "schema": "royal-capital.fortification.r0c-cp2.cp2d-source-checkpoint/1",
            "status": "PRESERVED_BEFORE_LONG_VALIDATION",
            "checkpoint": "CP2-D-R1",
            "accepted_parent": {"name": "C2C.zip", "sha256": C2C_SHA256},
            "fixed_baseline": {"name": "RCF_D0_full.zip", "sha256": BASELINE_SHA256},
            "source_overlay": source_inventory(source_root),
            "removed_disposable_caches": removed_caches,
            "functional_qualification": "NOT_RUN",
            "partial_output_published": False,
        },
    )
    pt.write_registry(current)
    current_registry = pt.validate_registry(current)
    require(current_registry["status"] == "PASS", f"implementation registry failed: {current_registry}")
    check = build_checkpoint(
        authority=authority,
        baseline=baseline,
        c1=c1,
        parent=parent,
        current=current,
        output=output,
        work=work,
        copied=copied,
        removed_caches=removed_caches,
    )
    write_json(
        work / "STATE.json",
        {
            "baseline": baseline.as_posix(),
            "c1": c1.as_posix(),
            "parent_c2c": parent.as_posix(),
            "current": current.as_posix(),
            "output": output.as_posix(),
            "last_preserved": "C2_pre",
            "prevalidation_status": check["status"],
        },
    )
    print(json.dumps({"status": "CP2D_PREPARED", "current": current.as_posix()}, sort_keys=True))


def build_receipts(current: Path, output: Path, work: Path, runtime_logs: Path) -> dict[str, Any]:
    root = work / "receipts"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    cp2 = current / "RC_K0/child_designs/fortification/r0c_cp2"
    for name in (
        "CP2D_SOURCE_CHECKPOINT.json",
        "CP2D_NEGATIVE_GATES.json",
        "CP2D_FINAL_CLEAN_REPLAY.json",
        "CP2D_COMPLETION.json",
    ):
        source = cp2 / "reports" / name
        require(source.is_file(), f"missing closeout receipt: {name}")
        target = root / "reports" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    closeout_doc = cp2 / "docs/CP2D_CLOSEOUT.md"
    require(closeout_doc.is_file(), "missing CP2-D closeout document")
    target = root / "reports/CP2D_CLOSEOUT.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(closeout_doc, target)
    negative = cp2 / "reports/negative_gates"
    for source in sorted(negative.rglob("failure-result.json")):
        target = root / "negative_gates" / source.relative_to(negative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    pre = output / "C2_pre_check.json"
    require(pre.is_file(), "missing C2_pre_check.json")
    shutil.copy2(pre, root / "C2_pre_check.json")
    if runtime_logs.is_dir():
        for source in sorted(path for path in runtime_logs.rglob("*") if path.is_file()):
            target = root / "runtime_logs" / source.relative_to(runtime_logs)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write_json(
        root / "FAILED_ATTEMPTS.json",
        {
            "schema": "royal-capital.fortification.r0c-cp2.cp2d-failed-attempts/1",
            "preserved": [
                {
                    "context": "previous conversation local execution backend",
                    "result": "ClientError before CP2-D source modification",
                    "recovery": "resumed from accepted CP2-C branch and release through GitHub Actions",
                    "rollback": False,
                    "accepted_parent_deleted": False,
                }
            ],
        },
    )
    destination = output / "C2R.zip"
    pt.deterministic_zip(root, destination)
    return zip_artifact(destination)


def finalize(args: argparse.Namespace) -> None:
    inputs = Path(args.inputs).resolve()
    work = Path(args.work).resolve()
    output = Path(args.output).resolve()
    runtime_logs = Path(args.runtime_logs).resolve()
    authority = validate_inputs(inputs)
    baseline = work / "baseline"
    c1 = work / "c1"
    parent = work / "parent_c2c"
    current = work / "current"
    require(all(path.is_dir() for path in (baseline, c1, parent, current)), "prepared CP2-D workspace is missing")
    cp2 = current / "RC_K0/child_designs/fortification/r0c_cp2"
    negative = json.loads((cp2 / "reports/CP2D_NEGATIVE_GATES.json").read_text(encoding="utf-8"))
    replay = json.loads((cp2 / "reports/CP2D_FINAL_CLEAN_REPLAY.json").read_text(encoding="utf-8"))
    completion = json.loads((cp2 / "reports/CP2D_COMPLETION.json").read_text(encoding="utf-8"))
    require(negative.get("status") == "PASS" and negative.get("passed") == 18, "18 negative gates did not pass")
    require(negative.get("atomic_no_partial_output") == "PASS_ALL_18", "atomic publication gate did not pass")
    require(replay.get("status") == "PASS" and replay.get("fixtures_passed") == 3, "final clean replay did not pass")
    require(completion.get("stage_completion") == "R0C_CP2_COMPLETE_CLOSED", "CP2-D completion receipt mismatch")
    require(completion.get("godot_product") == "NOT_STARTED", "Godot product status changed")
    removed_caches = remove_disposable_caches(current)
    pt.write_registry(current)
    final_registry = pt.validate_registry(current)
    require(final_registry["status"] == "PASS", f"final registry failed: {final_registry}")

    full = output / "C2.zip"
    patch = output / "C2P.zip"
    source = output / "C2S.zip"
    source_meta = pt.create_source_delta(c1, current, source, parent_name="C1.zip", stage=STAGE)
    patch_meta = create_direct_patch(baseline, current, patch, stage=STAGE)
    pt.deterministic_zip(current, full)
    full_reopen = pt.verify_zip_tree(full, current, work / "verify_final_full")
    baseline_replay = verify_baseline_plus_patch(baseline, patch, current, work / "verify_final_patch")
    c1_replay = pt.verify_parent_plus_delta(c1, source, current, work / "verify_final_source")
    patch_payload = pt.verify_patch_payload(patch, current, source, work / "verify_final_payload")
    parent_delta = work / "C2D_current_delta.zip"
    parent_delta_meta = pt.create_source_delta(parent, current, parent_delta, parent_name="C2C.zip", stage=STAGE + "_CURRENT_DELTA")
    parent_replay = pt.verify_parent_plus_delta(parent, parent_delta, current, work / "verify_final_parent")
    for name, value in {
        "full reopen": full_reopen,
        "fixed baseline + patch": baseline_replay,
        "C1 + source": c1_replay,
        "patch current payload": patch_payload,
        "C2C + current delta": parent_replay,
    }.items():
        require(value["status"] == "PASS", f"final {name} failed: {value}")

    receipt = build_receipts(current, output, work, runtime_logs)
    artifacts = {
        "C2.zip": zip_artifact(full),
        "C2P.zip": zip_artifact(patch),
        "C2S.zip": zip_artifact(source),
        "C2R.zip": receipt,
    }
    require(all(row.get("status") == "PASS" for row in artifacts.values()), "final ZIP integrity failed")
    check = {
        "schema": "royal-capital.fortification.r0c-cp2.closeout-check/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "official_subtitle": "TOWER JOINS",
        "checkpoint": "CP2-D-R1",
        "source_preservation": "PASS",
        "exact_runtime": "PASS_REUSED_CPYTHON_3_13_5_BUILD123D_E22D34DAE_OCP_8_0_1_0_0",
        "functional_qualification": "PASS_18_NEGATIVE_GATES_AND_FINAL_CLEAN_REPLAY",
        "stage_completion": "R0C_CP2_COMPLETE_CLOSED",
        "r0c_cp2": "COMPLETE_CLOSED",
        "godot_product": "NOT_STARTED",
        "negative_gates": negative,
        "final_clean_replay": replay,
        "removed_disposable_caches": removed_caches,
        "authority": authority,
        "registries": {
            "baseline": pt.validate_registry(baseline),
            "C1": pt.validate_registry(c1),
            "accepted_C2C": pt.validate_registry(parent),
            "C2": final_registry,
        },
        "source_delta": source_meta,
        "current_delta": parent_delta_meta,
        "cumulative_patch": patch_meta,
        "artifacts": artifacts,
        "validation": {
            "status": "PASS",
            "full_reopen": full_reopen,
            "fixed_baseline_plus_cumulative_patch_equals_full": baseline_replay,
            "C1_plus_source_equals_full": c1_replay,
            "C2C_plus_current_delta_equals_full": parent_replay,
            "patch_current_delta_equals_source_payload": patch_payload,
        },
        "roadmap_progress": {
            "r0c_before": "1/4",
            "r0c_after": "2/4",
            "accepted_checkpoints_before": "10/25",
            "accepted_checkpoints_after": "11/25",
            "next_exact_checkpoint": "R0C-CP3",
        },
    }
    write_json(output / "C2_check.json", check)
    report = f"""# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — TOWER JOINS

## Final verdict

```text
source preservation          PASS
exact runtime reuse          PASS
negative gates               18 / 18 PASS
atomic no-partial-output     PASS
final clean replay           FX01 / FX02 / FX03 PASS
functional qualification     PASS
stage completion             R0C_CP2_COMPLETE / CLOSED
Godot product                NOT_STARTED
R0C progress                 2 / 4
accepted checkpoints         11 / 25
```

## Equality

```text
RCF_D0_full.zip + C2P.zip = C2.zip   PASS_DIRECT
C1.zip          + C2S.zip = C2.zip   PASS_DIRECT
C2P current CP2 payload    = C2S     PASS
C2C.zip + CP2-D delta      = C2.zip  PASS
```

## Artifacts

```text
C2.zip   {artifacts['C2.zip']['bytes']} bytes  {artifacts['C2.zip']['sha256']}
C2P.zip  {artifacts['C2P.zip']['bytes']} bytes  {artifacts['C2P.zip']['sha256']}
C2S.zip  {artifacts['C2S.zip']['bytes']} bytes  {artifacts['C2S.zip']['sha256']}
C2R.zip  {artifacts['C2R.zip']['bytes']} bytes  {artifacts['C2R.zip']['sha256']}
```

Next exact roadmap checkpoint: `R0C-CP3 — battlement rhythm, merlon/crenel/module instancing`.
"""
    (output / "C2_report.md").write_text(report, encoding="utf-8")
    delivery = {
        "schema": "royal-capital.fortification.r0c-cp2.delivery/1",
        "status": "R0C_CP2_COMPLETE_CLOSED",
        "source_preservation": "PASS",
        "functional_qualification": "PASS_18_NEGATIVE_GATES_AND_FINAL_CLEAN_REPLAY",
        "stage_completion": "R0C_CP2_COMPLETE_CLOSED",
        "godot_product": "NOT_STARTED",
        "artifacts": artifacts,
        "validation": check["validation"],
        "roadmap_progress": check["roadmap_progress"],
    }
    write_json(output / "C2_delivery.json", delivery)
    checksum_names = [
        "C2.zip",
        "C2P.zip",
        "C2S.zip",
        "C2R.zip",
        "C2_check.json",
        "C2_report.md",
        "C2_delivery.json",
        "C2_pre.zip",
        "C2_preP.zip",
        "C2_preS.zip",
        "C2_pre_check.json",
    ]
    (output / "C2.sha256").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in checksum_names),
        encoding="utf-8",
    )
    print(json.dumps({"status": "R0C_CP2_COMPLETE_CLOSED", "artifacts": artifacts}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--inputs", required=True)
    common.add_argument("--work", required=True)
    common.add_argument("--output", required=True)
    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--source-root", required=True)
    finalize_parser = subparsers.add_parser("finalize", parents=[common])
    finalize_parser.add_argument("--runtime-logs", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
