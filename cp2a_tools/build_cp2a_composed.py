from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import build_cp2a as base

BASELINE_RECEIPT_SHA256 = "eafa0579f11a72ecd85eb5996266c7bc2970c822a8b4788521791aa74f8332ac"
ACCEPTED_REPLAY_ARTIFACT_SHA256 = "17be24bd7c8868e5ba14ea100162b747df5225d3ec7145b2324ebef49ee4d2cb"
ACCEPTED_REPLAY_RUN_ID = 36192768002
ACCEPTED_REPLAY_ARTIFACT_ID = 10888433130

FAILED_ATTEMPTS = [
    {
        "run_id": 36211770604,
        "preserved_reason": "historical C1.sha256 listed implementation/pre checkpoints not carried by the accepted transport container",
    },
    {
        "run_id": 36211879157,
        "preserved_reason": "C1_check.json is a byte-equal delivery sidecar but is not independently listed by historical C1.sha256",
    },
    {
        "run_id": 36212007672,
        "preserved_reason": "obsolete extracted-file count 1417 was used instead of the accepted 1418-entry C1 authority",
    },
    {
        "run_id": 36212512613,
        "preserved_reason": "fixed-baseline conversation transport URL expired with HTTP 403 before checkpoint generation",
    },
    {
        "run_id": 36212994542,
        "preserved_reason": "accepted fixed-baseline replay receipt was searched inside C1 transport although it is a separate accepted Actions artifact",
    },
]


def require(condition: bool, message: str) -> None:
    base.require(condition, message)


def verify_accepted_replay(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"accepted replay receipt missing: {path}")
    actual_sha = base.sha256_path(path)
    require(actual_sha == BASELINE_RECEIPT_SHA256, f"accepted replay receipt SHA mismatch: {actual_sha}")
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data.get("schema") == "royal-capital.fortification.r0c-cp1-baseline-patch-replay/1", "accepted replay schema mismatch")
    require(data.get("status") == "PASS", "accepted replay status is not PASS")
    require(data.get("baseline_plus_patch_equals_full") is True, "accepted replay equality is not true")
    comparison = data.get("comparison", {})
    for key in ("different", "extra", "missing", "mode_different"):
        require(comparison.get(key) == [], f"accepted replay comparison {key} is not empty")
    fixed = data.get("fixed_baseline", {})
    patch = data.get("cumulative_patch", {})
    expected = data.get("expected_full", {})
    reconstructed = data.get("reconstructed", {})
    require(fixed.get("name") == "RCF_D0_full.zip", "accepted replay baseline name mismatch")
    require(fixed.get("sha256") == base.BASELINE_SHA256, "accepted replay baseline SHA mismatch")
    require(fixed.get("files") == 1418, "accepted replay baseline file count mismatch")
    require(patch.get("name") == "C1P.zip", "accepted replay patch name mismatch")
    require(patch.get("sha256") == base.AUTHORITY_SHA256["C1P.zip"], "accepted replay C1P SHA mismatch")
    require(patch.get("deleted_paths") == 0, "accepted replay C1P deletion count mismatch")
    require(expected.get("name") == "C1.zip", "accepted replay full name mismatch")
    require(expected.get("sha256") == base.AUTHORITY_SHA256["C1.zip"], "accepted replay C1 SHA mismatch")
    require(expected.get("files") == 1418, "accepted replay C1 file count mismatch")
    require(reconstructed.get("files") == 1418, "accepted replay reconstructed file count mismatch")
    require(
        expected.get("closed_world_digest") == reconstructed.get("closed_world_digest"),
        "accepted replay closed-world digest mismatch",
    )
    return {
        "status": "PASS_ACCEPTED_DIRECT_REPLAY",
        "receipt": path.name,
        "receipt_sha256": actual_sha,
        "accepted_workflow_run_id": ACCEPTED_REPLAY_RUN_ID,
        "accepted_artifact_id": ACCEPTED_REPLAY_ARTIFACT_ID,
        "accepted_artifact_sha256": ACCEPTED_REPLAY_ARTIFACT_SHA256,
        "baseline_sha256": fixed["sha256"],
        "cumulative_patch_sha256": patch["sha256"],
        "expected_full_sha256": expected["sha256"],
        "files": expected["files"],
        "closed_world_digest": expected["closed_world_digest"],
        "comparison": comparison,
    }


def validate_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = [row for row in rows if row["required_at_r0c_cp2"] == "true"]
    require(len(rows) == 6, f"fixture matrix row count mismatch: {len(rows)}")
    require(len(required) == 3, f"required fixture count mismatch: {len(required)}")
    require(all(row["status"] == "FROZEN_PENDING_IMPLEMENTATION" for row in required), "fixture matrix prematurely claims implementation")
    return {
        "status": "PASS",
        "rows": len(rows),
        "required_rows": len(required),
        "required_fixture_ids": [row["fixture_id"] for row in required],
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_receipt(source: Path, target: Path) -> None:
    require(source.is_file(), f"required receipt missing: {source}")
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--accepted-replay", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transport = args.transport.resolve()
    accepted_replay = args.accepted_replay.resolve()
    source_root = args.source_root.resolve()
    work = args.work.resolve()
    output = args.output.resolve()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)

    # Authority intake. This precedes source modification and does not mutate accepted bytes.
    require(transport.stat().st_size == base.TRANSFER_BYTES, "C1_transfer byte size mismatch")
    require(base.sha256_path(transport) == base.TRANSFER_SHA256, "C1_transfer SHA-256 mismatch")
    transport_info = base.artifact(transport)
    accepted_replay_info = verify_accepted_replay(accepted_replay)

    transport_tree = work / "transport"
    base.pt.safe_extract(transport, transport_tree)
    manifest = base.parse_sha_manifest(transport_tree / "C1.sha256")
    authority_zip_info: dict[str, Any] = {}
    for name, expected_sha in base.AUTHORITY_SHA256.items():
        path = transport_tree / name
        require(path.is_file(), f"missing accepted authority file: {name}")
        require(base.sha256_path(path) == expected_sha, f"{name} exact SHA-256 mismatch")
        require(manifest.get(name) == expected_sha, f"{name} does not match delivered C1.sha256")
        authority_zip_info[name] = base.artifact(path)
    require(
        (transport_tree / "C1_check.json").read_bytes()
        == (transport_tree / "C1_delivery.json").read_bytes(),
        "C1_check.json must be byte-equal to C1_delivery.json",
    )

    parent = work / "parent_c1"
    current = work / "current_c2a"
    base.pt.safe_extract(transport_tree / "C1.zip", parent)
    shutil.copytree(parent, current, copy_function=shutil.copy2)
    parent_registry = base.pt.validate_registry(parent)
    require(parent_registry["status"] == "PASS", f"C1 FILES.sha256 failed: {parent_registry}")
    require(parent_registry["tree_files_including_registry"] == 1418, f"C1 file count mismatch: {parent_registry}")

    # Implementation unit: publish only the three frozen CP2-A source files.
    for rel in base.TARGETS:
        source = source_root / rel
        target = current / rel
        require(source.is_file(), f"staged freeze source missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    base.pt.write_registry(current)

    diff = base.pt.tree_diff(parent, current)

    # Preserve source, fixed-baseline cumulative patch, and full checkpoint before focused validation.
    pre_full = output / "C2A_pre.zip"
    pre_patch = output / "C2A_preP.zip"
    pre_source = output / "C2A_preS.zip"
    source_meta = base.pt.create_source_delta(
        parent,
        current,
        pre_source,
        parent_name="C1.zip",
        stage="ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2_CP2-A_FREEZE_PUBLICATION",
    )
    patch_meta = base.build_cumulative_patch(
        transport_tree / "C1P.zip",
        pre_source,
        pre_patch,
        work / "patch_build",
    )
    base.pt.deterministic_zip(current, pre_full)
    pre_checkpoint = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-pre-checkpoint/1",
        "status": "PRESERVED_BEFORE_FOCUSED_VALIDATION",
        "authority": base.AUTHORITY_SHA256,
        "fixed_baseline_sha256": base.BASELINE_SHA256,
        "accepted_fixed_baseline_replay": accepted_replay_info,
        "source_diff": diff,
        "artifacts": {
            "full": base.zip_info(pre_full),
            "patch": base.zip_info(pre_patch),
            "source": base.zip_info(pre_source),
        },
    }
    write_json(output / "C2A_pre_check.json", pre_checkpoint)

    # Focused validation starts only after all three checkpoint ZIPs exist.
    current_registry = base.pt.validate_registry(current)
    require(current_registry["status"] == "PASS", f"CP2-A FILES.sha256 failed: {current_registry}")
    require(current_registry["tree_files_including_registry"] == 1421, f"CP2-A file count mismatch: {current_registry}")
    require(diff["added"] == sorted(base.TARGETS), f"unexpected added paths: {diff['added']}")
    require(diff["modified"] == ["FILES.sha256"], f"unexpected modified paths: {diff['modified']}")
    require(diff["deleted"] == [], f"unexpected deleted paths: {diff['deleted']}")

    parent_wheels = base.wheel_inventory(parent)
    current_wheels = base.wheel_inventory(current)
    require(len(parent_wheels) == 58, f"accepted C1 wheelhouse count mismatch: {len(parent_wheels)}")
    require(parent_wheels == current_wheels, "CP2-A changed the accepted 58-wheel runtime closure")

    freeze_validation = base.verify_freeze_files(current)
    matrix_validation = validate_matrix(current / base.TARGETS[2])
    full_reopen = base.pt.verify_zip_tree(pre_full, current, work / "verify_full")
    parent_plus_source = base.pt.verify_parent_plus_delta(parent, pre_source, current, work / "verify_source")
    patch_current_delta = base.pt.verify_patch_payload(pre_patch, current, pre_source, work / "verify_patch_payload")

    require(full_reopen["status"] == "PASS", f"full reopen failed: {full_reopen}")
    require(parent_plus_source["status"] == "PASS", f"C1 + source equality failed: {parent_plus_source}")
    require(patch_current_delta["status"] == "PASS", f"cumulative patch current-delta equality failed: {patch_current_delta}")

    compositional_equality = {
        "status": "PASS_COMPOSITIONAL_WITH_ACCEPTED_DIRECT_REPLAY",
        "premises": {
            "fixed_baseline_plus_C1P_equals_C1": accepted_replay_info,
            "C1_plus_C2AS_equals_C2A": parent_plus_source,
            "C2AP_current_delta_equals_C2AS_payload": patch_current_delta,
            "C2AP_constructed_from_exact_C1P_plus_exact_C2AS": {
                "status": "PASS",
                "C1P_sha256": base.AUTHORITY_SHA256["C1P.zip"],
                "C2AS_sha256": base.sha256_path(pre_source),
                "C2AP_sha256": base.sha256_path(pre_patch),
            },
        },
        "conclusion": "RCF_D0_full.zip + C2AP.zip = C2A.zip",
        "direct_fixed_baseline_replay_this_run": "NOT_REPEATED_ACCEPTED_DIRECT_REPLAY_RECEIPT_REOPENED",
        "reason": "The immutable fixed baseline transport URL expired; the accepted direct replay receipt was freshly reopened and exact current composition was freshly verified.",
    }

    validation = {
        "status": "PASS",
        "authority_transfer": transport_info,
        "authority_inner_zips": authority_zip_info,
        "accepted_fixed_baseline_replay": accepted_replay_info,
        "parent_registry": parent_registry,
        "current_registry": current_registry,
        "freeze_files": freeze_validation,
        "fixture_matrix": matrix_validation,
        "source_diff": diff,
        "wheelhouse": {
            "status": "PASS",
            "count": len(current_wheels),
            "unchanged": parent_wheels == current_wheels,
        },
        "full_reopen": full_reopen,
        "C1_plus_source_equals_full": parent_plus_source,
        "patch_current_delta_equals_source_payload": patch_current_delta,
        "fixed_baseline_plus_cumulative_patch_equals_full": compositional_equality,
    }

    # Final artifacts are byte-identical copies of the pre-validation checkpoint.
    final_full = output / "C2A.zip"
    final_patch = output / "C2AP.zip"
    final_source = output / "C2AS.zip"
    shutil.copy2(pre_full, final_full)
    shutil.copy2(pre_patch, final_patch)
    shutil.copy2(pre_source, final_source)
    require(final_full.read_bytes() == pre_full.read_bytes(), "final full differs from pre full")
    require(final_patch.read_bytes() == pre_patch.read_bytes(), "final patch differs from pre patch")
    require(final_source.read_bytes() == pre_source.read_bytes(), "final source differs from pre source")

    core_artifacts = {
        "C2A.zip": base.artifact(final_full),
        "C2AP.zip": base.artifact(final_patch),
        "C2AS.zip": base.artifact(final_source),
    }
    check = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-freeze-publication-check/2",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "checkpoint": "CP2-A_FREEZE_PUBLICATION_CHECKPOINT",
        "official_subtitle": "TOWER JOINS",
        "source_preservation": "PASS",
        "exact_authority_admission": "PASS",
        "specification_freeze": "PASS",
        "functional_qualification": "NOT_STARTED_PRODUCT",
        "stage_completion": "CP2_A_COMPLETE_CLOSED",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "roadmap_progress": {
            "r0c": "1/4",
            "accepted_checkpoints": "10/25",
            "next_exact_checkpoint": "R0C-CP2 CP2-B",
        },
        "failed_attempts_preserved": FAILED_ATTEMPTS,
        "authority": {
            "transport": {
                "sha256": base.TRANSFER_SHA256,
                "bytes": base.TRANSFER_BYTES,
            },
            "inner": base.AUTHORITY_SHA256,
            "fixed_baseline": {
                "name": "RCF_D0_full.zip",
                "sha256": base.BASELINE_SHA256,
            },
            "accepted_direct_replay": accepted_replay_info,
        },
        "source_changes": diff,
        "pre_checkpoint": pre_checkpoint,
        "artifacts": core_artifacts,
        "source_delta": source_meta,
        "cumulative_patch": patch_meta,
        "validation": validation,
        "pre_final_byte_equality": {
            "full": True,
            "patch": True,
            "source": True,
        },
    }
    write_json(output / "C2A_check.json", check)

    delivery = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-delivery/2",
        "status": "CP2_A_COMPLETE_CLOSED",
        "task": check["task"],
        "official_subtitle": check["official_subtitle"],
        "source_preservation": "PASS",
        "functional_qualification": "NOT_STARTED_PRODUCT",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "artifacts": core_artifacts,
        "equality": {
            "RCF_D0_full_plus_C2AP_equals_C2A": "PASS_COMPOSITIONAL_WITH_ACCEPTED_DIRECT_REPLAY",
            "C1_plus_C2AS_equals_C2A": "PASS_DIRECT",
            "C2AP_current_delta_equals_C2AS_payload": "PASS_DIRECT",
        },
        "failed_attempts_preserved": FAILED_ATTEMPTS,
    }
    write_json(output / "C2A_delivery.json", delivery)

    report = f"""# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — CP2-A FREEZE PUBLICATION

## Verdict

```text
source preservation                         PASS
exact C1 authority admission                PASS
C1 FILES.sha256 closed-world replay         PASS
scope / contract / fixture freeze           PASS
fixed baseline + cumulative patch equality  PASS_COMPOSITIONAL_WITH_ACCEPTED_DIRECT_REPLAY
C1 + source delta equality                  PASS_DIRECT
ZIP reopen / CRC                            PASS
functional qualification                    NOT_STARTED_PRODUCT
stage completion                            CP2_A_COMPLETE / CLOSED
R0C-CP2                                     OPEN
Godot product                               NOT_STARTED
```

## Frozen authority

```text
title               tower joins
output              TowerJoinPlan
focused validation  tangent/wall-walk
scope               corner/tangent/wall-penetrating tower joins
```

## Source changes

```text
added      {len(diff['added'])}
modified   {len(diff['modified'])}
deleted    {len(diff['deleted'])}
wheelhouse 58 / unchanged
```

Only the three CP2-A freeze files were added; root `FILES.sha256` was regenerated. No accepted tower, span, runtime, OSS, license, provenance, STEP, or BREP byte was modified.

## Equality proof

```text
accepted direct replay:
  RCF_D0_full.zip + exact C1P.zip = exact C1.zip  PASS
fresh current replay:
  exact C1.zip + C2AS.zip = C2A.zip               PASS
fresh patch composition:
  C2AP current delta = C2AS payload               PASS
conclusion:
  RCF_D0_full.zip + C2AP.zip = C2A.zip            PASS_COMPOSITIONAL
pre checkpoint bytes = final bytes                PASS
```

The fixed-baseline conversation transport URL expired before this run. No baseline was regenerated. The separately accepted direct replay artifact was freshly reopened and validated, then combined with fresh current-delta equality.

## Roadmap

```text
R0C progress          1 / 4
accepted checkpoints  10 / 25
R0C-CP2               OPEN
next                   R0C-CP2 CP2-B — explicit start only
```
"""
    (output / "C2A_report.md").write_text(report, encoding="utf-8")

    base.write_core_sha(
        output,
        [
            "C2A.zip",
            "C2AP.zip",
            "C2AS.zip",
            "C2A_check.json",
            "C2A_delivery.json",
            "C2A_report.md",
            "C2A_pre_check.json",
        ],
    )

    receipts_tree = work / "receipts"
    receipts_tree.mkdir(parents=True)
    for name in [
        "C2A_check.json",
        "C2A_delivery.json",
        "C2A_report.md",
        "C2A.sha256",
        "C2A_pre_check.json",
    ]:
        copy_receipt(output / name, receipts_tree / name)
    for name in ["AUTHORITY_INSPECTION.json", "AUTHORITY_INSPECTION.md"]:
        copy_receipt(base.REPO_ROOT / "cp2a_authority_snapshot" / name, receipts_tree / name)
    copy_receipt(accepted_replay, receipts_tree / accepted_replay.name)
    copy_receipt(base.REPO_ROOT / "cp2a_baseline_receipt" / "INSPECTION.json", receipts_tree / "BASELINE_REPLAY_INSPECTION.json")
    receipts = output / "C2AR.zip"
    base.pt.deterministic_zip(receipts_tree, receipts)
    receipts_info = base.artifact(receipts)

    release = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-release/2",
        "status": "CP2_A_COMPLETE_CLOSED",
        "core_artifacts": core_artifacts,
        "receipts": receipts_info,
        "source_preservation": "PASS",
        "functional_qualification": "NOT_STARTED_PRODUCT",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "next": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 CP2-B",
    }
    write_json(output / "C2A_release.json", release)
    notes = """# R0C-CP2 CP2-A freeze publication

CP2-A specification publication only. `R0C_CP2_PASS` is not claimed.

- exact C1 authority: PASS
- source preservation: PASS
- scope/contract/fixture freeze: PASS
- fixed-baseline cumulative patch equality: PASS_COMPOSITIONAL_WITH_ACCEPTED_DIRECT_REPLAY
- C1 source-delta equality: PASS_DIRECT
- functional join geometry: NOT_STARTED_PRODUCT
- Godot product: NOT_STARTED
- next: CP2-B by explicit instruction
"""
    (output / "C2A_RELEASE_NOTES.md").write_text(notes, encoding="utf-8")

    small_tree = work / "small_delivery"
    small_tree.mkdir(parents=True)
    for name in [
        "C2AS.zip",
        "C2AR.zip",
        "C2A_check.json",
        "C2A_delivery.json",
        "C2A_report.md",
        "C2A.sha256",
        "C2A_release.json",
        "C2A_pre_check.json",
        "C2A_RELEASE_NOTES.md",
    ]:
        copy_receipt(output / name, small_tree / name)
    small = output / "C2A_SMALL.zip"
    base.pt.deterministic_zip(small_tree, small)
    small_info = base.artifact(small)
    (output / "C2A_SMALL.sha256").write_text(
        f"{small_info['sha256']}  C2A_SMALL.zip\n", encoding="utf-8"
    )

    transfer_tree = work / "transfer"
    transfer_tree.mkdir(parents=True)
    for name in [
        "C2A.zip",
        "C2AP.zip",
        "C2AS.zip",
        "C2AR.zip",
        "C2A_check.json",
        "C2A_delivery.json",
        "C2A_report.md",
        "C2A.sha256",
        "C2A_release.json",
        "C2A_pre_check.json",
    ]:
        copy_receipt(output / name, transfer_tree / name)
    transfer_manifest = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-transfer/1",
        "status": "CP2_A_COMPLETE_CLOSED",
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": base.sha256_path(path),
            }
            for path in sorted(transfer_tree.iterdir())
            if path.is_file()
        },
    }
    write_json(transfer_tree / "transport_manifest.json", transfer_manifest)
    transfer = output / "C2A_transfer.zip"
    base.pt.deterministic_zip(transfer_tree, transfer)
    transfer_info = base.artifact(transfer)
    (output / "C2A_transfer.sha256").write_text(
        f"{transfer_info['sha256']}  C2A_transfer.zip\n", encoding="utf-8"
    )

    summary = {
        "status": "PASS",
        "core": core_artifacts,
        "receipts": receipts_info,
        "small_delivery": small_info,
        "transfer": transfer_info,
        "validation": validation,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
