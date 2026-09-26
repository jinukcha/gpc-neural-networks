from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "r0c_cp1"))
sys.path.insert(0, str(REPO_ROOT))

import package_tools as pt  # noqa: E402
from cp2d_tools import driver as cp2d_driver  # noqa: E402

TASK = "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP3"
CHECKPOINT = "CP3-A-R1"
OFFICIAL_SCOPE = "battlement rhythm, merlon/crenel/module instancing"
BASELINE_SHA256 = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
C2_SHA256 = "686b9857cc37d0dff2c0c26bc3630d8c94a53c878a92001d4f0e3aa4200e69b2"
C2_TREE_DIGEST = "sha256:389f7c0865ece121f370296936f02b04efa06a4c5d9bf25fe32c0a01325edcb7"
C2_FILES = 1618
CP3_REL = Path("RC_K0/child_designs/fortification/r0c_cp3")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return pt.sha256_path(path)


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows[path.relative_to(root).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
    return rows


def inventory_digest(rows: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for rel, row in sorted(rows.items()):
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(str(row["sha256"])))
        digest.update(str(row["bytes"]).encode("ascii") + b"\0")
        digest.update(str(row["mode"]).encode("ascii") + b"\0")
    return "sha256:" + digest.hexdigest()


def tree_digest(root: Path) -> str:
    return inventory_digest(inventory(root))


def copy_overlay(source_root: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        rel = source.relative_to(source_root)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel.as_posix())
    return copied


def verify_json_csv(current: Path) -> dict[str, Any]:
    cp3 = current / CP3_REL
    required = {
        "README.md",
        "docs/CP3_SCOPE_FREEZE.md",
        "contracts/BATTLEMENT_RHYTHM_CONTRACT.json",
        "data/CP3_FIXTURE_MATRIX.csv",
        "provenance/CHANGESET.json",
        "reports/CP3A_SOURCE_RECEIPT.json",
    }
    observed = {
        path.relative_to(cp3).as_posix()
        for path in cp3.rglob("*")
        if path.is_file()
    }
    missing = sorted(required - observed)
    require(not missing, f"missing CP3-A source files: {missing}")

    contract = json.loads((cp3 / "contracts/BATTLEMENT_RHYTHM_CONTRACT.json").read_text(encoding="utf-8"))
    changeset = json.loads((cp3 / "provenance/CHANGESET.json").read_text(encoding="utf-8"))
    receipt = json.loads((cp3 / "reports/CP3A_SOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    require(contract.get("schema") == "royal-capital.fortification.battlement-rhythm/1", "contract schema mismatch")
    require(contract.get("authority", {}).get("accepted_parent_sha256") == C2_SHA256, "contract authority mismatch")
    require(contract.get("result", {}).get("partial_output_published") is False, "contract partial output policy changed")
    require(changeset.get("accepted_parent", {}).get("sha256") == C2_SHA256, "changeset authority mismatch")
    require(changeset.get("new_framework") is False, "unexpected framework introduction")
    require(receipt.get("accepted_parent", {}).get("sha256") == C2_SHA256, "source receipt authority mismatch")

    with (cp3 / "data/CP3_FIXTURE_MATRIX.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = ["R0C_CP3_FX01", "R0C_CP3_FX02", "R0C_CP3_FX03", "R0C_CP3_FX04"]
    require([row["fixture_id"] for row in rows] == expected_ids, f"fixture IDs changed: {rows}")
    require(len(rows) == 4, "fixture matrix must remain finite at four rows")
    require(all(row["functional_status"] == "NOT_STARTED" for row in rows), "CP3-A may not claim geometry PASS")

    forbidden_outputs = [
        path.relative_to(cp3).as_posix()
        for path in cp3.rglob("*")
        if path.is_file() and ("outputs/" in path.relative_to(cp3).as_posix() or path.suffix.lower() in {".step", ".brep"})
    ]
    require(not forbidden_outputs, f"CP3-A unexpectedly contains geometry output: {forbidden_outputs}")
    return {
        "status": "PASS",
        "required_files": sorted(required),
        "fixture_ids": expected_ids,
        "fixture_count": len(rows),
        "contract_schema": contract["schema"],
        "geometry_outputs": forbidden_outputs,
        "functional_geometry_qualification": "NOT_STARTED",
    }


def make_checkpoint(
    *,
    baseline: Path,
    parent: Path,
    current: Path,
    output: Path,
    work: Path,
    prefix: str,
    stage: str,
) -> dict[str, Any]:
    full = output / f"{prefix}.zip"
    patch = output / f"{prefix}P.zip"
    source = output / f"{prefix}S.zip"
    source_meta = pt.create_source_delta(parent, current, source, parent_name="C2.zip", stage=stage)
    patch_meta = cp2d_driver.create_direct_patch(baseline, current, patch, stage=stage)
    pt.deterministic_zip(current, full)
    full_reopen = pt.verify_zip_tree(full, current, work / f"verify_{prefix}_full")
    baseline_replay = cp2d_driver.verify_baseline_plus_patch(
        baseline, patch, current, work / f"verify_{prefix}_patch"
    )
    parent_replay = pt.verify_parent_plus_delta(
        parent, source, current, work / f"verify_{prefix}_source"
    )
    payload = pt.verify_patch_payload(
        patch, current, source, work / f"verify_{prefix}_payload"
    )
    validations = {
        "full_reopen_equals_tree": full_reopen,
        "fixed_baseline_plus_cumulative_patch_equals_full": baseline_replay,
        "C2_plus_source_equals_full": parent_replay,
        "patch_current_delta_equals_source_payload": payload,
    }
    for name, value in validations.items():
        require(value.get("status") == "PASS", f"{prefix} {name} failed: {value}")
    artifacts = {
        full.name: cp2d_driver.zip_artifact(full),
        patch.name: cp2d_driver.zip_artifact(patch),
        source.name: cp2d_driver.zip_artifact(source),
    }
    require(all(value.get("status") == "PASS" for value in artifacts.values()), f"{prefix} ZIP integrity failed")
    return {
        "source_delta": source_meta,
        "cumulative_patch": patch_meta,
        "artifacts": artifacts,
        "validation": {"status": "PASS", **validations},
    }


def build_receipts(current: Path, output: Path, pre_check: Path, validation: Path) -> dict[str, Any]:
    root = output / "C3AR.tree"
    shutil.rmtree(root, ignore_errors=True)
    cp3 = current / CP3_REL
    for rel in (
        "README.md",
        "docs/CP3_SCOPE_FREEZE.md",
        "contracts/BATTLEMENT_RHYTHM_CONTRACT.json",
        "data/CP3_FIXTURE_MATRIX.csv",
        "provenance/CHANGESET.json",
        "reports/CP3A_SOURCE_RECEIPT.json",
        "reports/CP3A_VALIDATION.json",
    ):
        source = cp3 / rel
        target = root / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(pre_check, root / pre_check.name)
    shutil.copy2(validation, root / validation.name)
    destination = output / "C3AR.zip"
    pt.deterministic_zip(root, destination)
    shutil.rmtree(root)
    result = cp2d_driver.zip_artifact(destination)
    require(result.get("status") == "PASS", f"C3AR integrity failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-zip", required=True)
    parser.add_argument("--c2-tree", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_zip = Path(args.baseline_zip).resolve()
    parent = Path(args.c2_tree).resolve()
    source_root = Path(args.source_root).resolve()
    work = Path(args.work).resolve()
    output = Path(args.output).resolve()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    work.mkdir(parents=True)
    output.mkdir(parents=True)

    require(sha256(baseline_zip) == BASELINE_SHA256, "fixed baseline SHA mismatch")
    parent_registry = pt.validate_registry(parent)
    require(parent_registry.get("status") == "PASS", f"accepted C2 registry failed: {parent_registry}")
    require(parent_registry.get("tree_files_including_registry") == C2_FILES, f"accepted C2 file count mismatch: {parent_registry}")
    require(tree_digest(parent) == C2_TREE_DIGEST, f"accepted C2 tree digest mismatch: {tree_digest(parent)}")

    baseline = work / "baseline"
    pt.safe_extract(baseline_zip, baseline)
    current = work / "current"
    shutil.copytree(parent, current, copy_function=shutil.copy2)
    removed_caches = cp2d_driver.remove_disposable_caches(current)
    copied = copy_overlay(source_root, current)
    require(copied, "CP3-A source overlay is empty")
    pt.write_registry(current)
    pre_registry = pt.validate_registry(current)
    require(pre_registry.get("status") == "PASS", f"prevalidation registry failed: {pre_registry}")

    pre = make_checkpoint(
        baseline=baseline,
        parent=parent,
        current=current,
        output=output,
        work=work,
        prefix="C3A_pre",
        stage="ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP3_CP3-A_PREVALIDATION",
    )
    pre_check = {
        "schema": "royal-capital.fortification.r0c-cp3.cp3a-pre-checkpoint/1",
        "task": TASK,
        "checkpoint": CHECKPOINT,
        "status": "PRESERVED_AND_REOPENED_BEFORE_VALIDATION",
        "accepted_parent": {"name": "C2.zip", "sha256": C2_SHA256, "tree_digest": C2_TREE_DIGEST},
        "fixed_baseline": {"name": "RCF_D0_full.zip", "sha256": BASELINE_SHA256},
        "source_overlay_paths": copied,
        "removed_disposable_caches": removed_caches,
        "registry": pre_registry,
        **pre,
        "source_preservation": "PASS",
        "functional_qualification": "NOT_STARTED_CP3_GEOMETRY",
        "stage_completion": "OPEN_PENDING_FOCUSED_VALIDATION",
        "godot_product": "NOT_STARTED",
    }
    pre_check_path = output / "C3A_pre_check.json"
    write_json(pre_check_path, pre_check)

    focused = verify_json_csv(current)
    validation_path = current / CP3_REL / "reports/CP3A_VALIDATION.json"
    validation = {
        "schema": "royal-capital.fortification.r0c-cp3.cp3a-validation/1",
        "task": TASK,
        "checkpoint": CHECKPOINT,
        "status": "PASS",
        "accepted_C2_reopen": "PASS_EXACT_TREE",
        "scope_freeze": "PASS",
        "contract_freeze": "PASS",
        "fixture_matrix_freeze": "PASS_FINITE_4",
        "focused_parse": focused,
        "functional_geometry_qualification": "NOT_STARTED",
        "partial_output_published": false,
        "accepted_parent_mutated": false,
        "godot_product": "NOT_STARTED"
    }
    write_json(validation_path, validation)

    receipt_path = current / CP3_REL / "reports/CP3A_SOURCE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "status": "PASS",
            "scope_freeze": "PASS",
            "contract_freeze": "PASS",
            "fixture_matrix_freeze": "PASS_FINITE_4",
            "source_preservation": "PASS",
            "functional_qualification": "NOT_STARTED_CP3_GEOMETRY",
            "stage_completion": "CP3_A_COMPLETE_CLOSED",
            "partial_output_published": False,
            "accepted_parent_mutated": False,
            "godot_product": "NOT_STARTED",
        }
    )
    write_json(receipt_path, receipt)
    pt.write_registry(current)
    final_registry = pt.validate_registry(current)
    require(final_registry.get("status") == "PASS", f"final registry failed: {final_registry}")

    final = make_checkpoint(
        baseline=baseline,
        parent=parent,
        current=current,
        output=output,
        work=work,
        prefix="C3A",
        stage="ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP3_CP3-A",
    )
    receipt_artifact = build_receipts(current, output, pre_check_path, validation_path)
    core_artifacts = {**final["artifacts"], "C3AR.zip": receipt_artifact}
    final_check = {
        "schema": "royal-capital.fortification.r0c-cp3.cp3a-check/1",
        "task": TASK,
        "official_scope": OFFICIAL_SCOPE,
        "checkpoint": CHECKPOINT,
        "status": "PASS",
        "source_preservation": "PASS",
        "functional_qualification": "NOT_STARTED_CP3_GEOMETRY",
        "stage_completion": "CP3_A_COMPLETE_CLOSED",
        "r0c_cp3": "OPEN",
        "godot_product": "NOT_STARTED",
        "accepted_parent": {
            "name": "C2.zip",
            "sha256": C2_SHA256,
            "tree_digest": C2_TREE_DIGEST,
            "files": C2_FILES,
        },
        "fixed_baseline": {"name": "RCF_D0_full.zip", "sha256": BASELINE_SHA256},
        "focused_validation": focused,
        "registry": final_registry,
        "tree": {"files": len(inventory(current)), "digest": tree_digest(current)},
        "source_delta": final["source_delta"],
        "cumulative_patch": final["cumulative_patch"],
        "artifacts": core_artifacts,
        "validation": final["validation"],
        "roadmap_progress": {
            "r0c": "2/4",
            "accepted_implementation_checkpoints": "11/25",
            "cp3a_subcheckpoint": "COMPLETE_CLOSED",
            "next": "R0C-CP3_CP3-B_R0C_CP3_FX01_LINEAR_BATTLEMENT_PILOT",
            "auto_continuation": False,
        },
    }
    check_path = output / "C3A_check.json"
    write_json(check_path, final_check)

    report = f"""# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP3 — CP3-A

## Verdict

```text
accepted C2 authority          PASS_EXACT_TREE
scope freeze                   PASS
rhythm contract                PASS
finite fixture matrix          PASS — 4 rows
source preservation            PASS
functional geometry            NOT_STARTED
stage completion               CP3_A_COMPLETE / CLOSED
R0C-CP3                        OPEN
Godot product                  NOT_STARTED
```

## Direct replay

```text
RCF_D0_full.zip + C3AP.zip = C3A.zip  PASS
C2.zip          + C3AS.zip = C3A.zip  PASS
```

## Next

```text
R0C-CP3 CP3-B — R0C_CP3_FX01 LINEAR BATTLEMENT PILOT
```
"""
    report_path = output / "C3A_report.md"
    report_path.write_text(report, encoding="utf-8")
    delivery = {
        "schema": "royal-capital.fortification.r0c-cp3.cp3a-delivery/1",
        "task": TASK,
        "checkpoint": CHECKPOINT,
        "status": "PASS",
        "source_preservation": "PASS",
        "functional_qualification": "NOT_STARTED_CP3_GEOMETRY",
        "stage_completion": "CP3_A_COMPLETE_CLOSED",
        "r0c_cp3": "OPEN",
        "godot_product": "NOT_STARTED",
        "artifacts": core_artifacts,
        "tree": final_check["tree"],
        "direct_replay": {
            "RCF_D0_plus_C3AP_equals_C3A": "PASS",
            "C2_plus_C3AS_equals_C3A": "PASS",
            "C3AP_current_payload_equals_C3AS": "PASS",
        },
        "progress": final_check["roadmap_progress"],
    }
    delivery_path = output / "C3A_delivery.json"
    write_json(delivery_path, delivery)

    checksum_names = [
        "C3A.zip",
        "C3AP.zip",
        "C3AS.zip",
        "C3AR.zip",
        "C3A_check.json",
        "C3A_delivery.json",
        "C3A_report.md",
        "C3A_pre.zip",
        "C3A_preP.zip",
        "C3A_preS.zip",
        "C3A_pre_check.json",
    ]
    lines = [f"{sha256(output / name)}  {name}" for name in checksum_names]
    (output / "C3A.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(final_check, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
