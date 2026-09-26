from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "r0c_cp1"))
import package_tools as pt  # noqa: E402

TRANSFER_SHA256 = "b8f4f6788c4295770364f722d9e40403dcdab652487404f50226f0c1995dae53"
TRANSFER_BYTES = 514_002_792
BASELINE_SHA256 = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
AUTHORITY_SHA256 = {
    "C1.zip": "91cf2025f311772b0a137d3509efa0b1d65b96832a78f1e4f46ba9cf3b06e878",
    "C1P.zip": "dddb960aa467689eecf1e312724ce06c0c9175e0fdc902c534c4e12ffdca77b4",
    "C1S.zip": "be7c76b3d936b78d844a3fa3afef7282bdb0ea05c2fe1689db3990d753548114",
    "C1R.zip": "a1ae24769d97c6fafdf37650170268b66645ad435f2df5bead0004fe2f10b937",
}
TARGETS = [
    "RC_K0/child_designs/fortification/r0c_cp2/docs/CP2_SCOPE_FREEZE.md",
    "RC_K0/child_designs/fortification/r0c_cp2/contracts/TOWER_JOIN_CONTRACT.json",
    "RC_K0/child_designs/fortification/r0c_cp2/data/CP2_FIXTURE_MATRIX.csv",
]
FAILED_ATTEMPTS = [
    {
        "run_id": 36211770604,
        "preserved_reason": "C1.sha256 also listed historical implementation/pre checkpoints absent from the transport container",
    },
    {
        "run_id": 36211879157,
        "preserved_reason": "C1_check.json is a byte-equal delivery sidecar but is not independently listed by historical C1.sha256",
    },
    {
        "run_id": 36212007672,
        "preserved_reason": "fixed extracted-file count used an obsolete 1417 value instead of the accepted 1418-entry C1 authority",
    },
]
TRANSPORT_METADATA = {"D.txt", "SOURCE_DELTA.json"}


def sha256_path(path: Path) -> str:
    return pt.sha256_path(path)


def zip_info(path: Path) -> dict[str, Any]:
    names: set[str] = set()
    duplicates: list[str] = []
    unsafe: list[str] = []
    symlinks: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name in names:
                duplicates.append(name)
            names.add(name)
            if name.startswith("/") or ".." in Path(name).parts:
                unsafe.append(name)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                symlinks.append(name)
        bad_crc = zf.testzip()
        entries = len(zf.infolist())
        file_entries = sum(1 for info in zf.infolist() if not info.is_dir())
    result = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "entries": entries,
        "file_entries": file_entries,
        "bad_crc": bad_crc,
        "duplicates": duplicates,
        "unsafe": unsafe,
        "symlinks": symlinks,
    }
    result["status"] = (
        "PASS"
        if bad_crc is None and not duplicates and not unsafe and not symlinks
        else "FAIL"
    )
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_sha_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64})\s+[* ]?(.+)$")
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(raw.strip())
        if match:
            rows[match.group(2)] = match.group(1).lower()
    return rows


def copy_overlay(write_root: Path, destination: Path) -> int:
    count = 0
    for source in sorted(write_root.rglob("*")):
        rel = source.relative_to(write_root)
        target = destination / rel
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def apply_cumulative_patch(
    baseline_tree: Path, patch_zip: Path, destination: Path, scratch: Path
) -> dict[str, Any]:
    shutil.rmtree(destination, ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)
    shutil.copytree(baseline_tree, destination, copy_function=shutil.copy2)
    pt.safe_extract(patch_zip, scratch)
    deleted: list[str] = []
    d_file = scratch / "D.txt"
    if d_file.is_file():
        deleted = [
            line.strip()
            for line in d_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    for rel in deleted:
        target = destination / rel
        if target.is_file() or target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    write_root = scratch / "W"
    require(write_root.is_dir(), f"{patch_zip.name} is missing W/ overlay")
    overlay_files = copy_overlay(write_root, destination)
    return {"overlay_files": overlay_files, "deleted": deleted}


def build_cumulative_patch(
    c1_patch: Path, source_delta: Path, destination: Path, scratch: Path
) -> dict[str, Any]:
    shutil.rmtree(scratch, ignore_errors=True)
    pt.safe_extract(c1_patch, scratch)
    deleted_paths: set[str] = set()
    d_file = scratch / "D.txt"
    if d_file.is_file():
        deleted_paths.update(
            line.strip()
            for line in d_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    pt._overlay_source_delta_into_patch(scratch, source_delta, deleted_paths)
    d_file.write_text(
        "\n".join(sorted(deleted_paths)) + ("\n" if deleted_paths else ""),
        encoding="utf-8",
    )
    metadata = {
        "schema": "royal-capital.cumulative-patch/1",
        "stage": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2_CP2-A_FREEZE_PUBLICATION",
        "baseline_file": "RCF_D0_full.zip",
        "baseline_sha256": BASELINE_SHA256,
        "accepted_parent_full": "C1.zip",
        "accepted_parent_sha256": AUTHORITY_SHA256["C1.zip"],
        "accepted_parent_cumulative_patch": "C1P.zip",
        "accepted_parent_cumulative_patch_sha256": AUTHORITY_SHA256["C1P.zip"],
        "current_delta": source_delta.name,
        "deleted_count": len(deleted_paths),
        "proof": "RCF_D0_full+C1P=C1; C1+current source delta=CP2-A full; overlaying the same current delta on C1P reconstructs CP2-A full from the fixed baseline",
    }
    (scratch / "PATCH_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pt.deterministic_zip(scratch, destination)
    return metadata


def verify_freeze_files(current: Path) -> dict[str, Any]:
    scope = current / TARGETS[0]
    contract = current / TARGETS[1]
    matrix = current / TARGETS[2]
    for path in (scope, contract, matrix):
        require(path.is_file(), f"missing required freeze file: {path}")

    scope_text = scope.read_text(encoding="utf-8")
    require(
        "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — TOWER JOINS" in scope_text,
        "scope file does not preserve official title",
    )
    require(
        "corner/tangent/wall-penetrating tower joins" in scope_text,
        "scope file does not preserve finite-roadmap CP2 scope",
    )

    contract_data = json.loads(contract.read_text(encoding="utf-8"))
    require(contract_data["authority"]["roadmap"]["title"] == "tower joins", "contract roadmap title mismatch")
    require(contract_data["authority"]["roadmap"]["output"] == "TowerJoinPlan", "contract output mismatch")
    require(contract_data["result"]["partial_output_published"] is False, "contract must fail closed")

    with matrix.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = [row for row in rows if row["required_at_r0c_cp2"] == "true"]
    require(len(required) == 3, "fixture matrix must contain exactly three required representative fixtures")
    require(
        {(row["tower_family"], row["join_kind"], row["span_family"]) for row in required}
        == {
            ("ROUND", "TANGENT", "STRAIGHT_SPAN"),
            ("SQUARE", "CORNER", "STRAIGHT_SPAN"),
            ("POLYGONAL", "WALL_PENETRATING", "STRAIGHT_SPAN"),
        },
        "fixture matrix required set mismatch",
    )
    require(all(row["status"] == "FROZEN_PENDING_IMPLEMENTATION" for row in required), "required fixtures must not claim PASS")

    roadmap = current / "RC_K0/child_designs/fortification/data/ROADMAP.csv"
    with roadmap.open("r", encoding="utf-8", newline="") as handle:
        roadmap_rows = list(csv.DictReader(handle))
    match = [row for row in roadmap_rows if row["stage"] == "R0C" and row["checkpoint"] == "CP2"]
    require(len(match) == 1, "authoritative R0C/CP2 roadmap row missing or duplicated")
    require(
        match[0]
        == {
            "stage": "R0C",
            "checkpoint": "CP2",
            "title": "tower joins",
            "outputs": "TowerJoinPlan",
            "focused_validation": "tangent/wall-walk",
            "non_goals": "gates",
            "gate": "R0C_CP2_PASS",
            "next": "R0C-CP3",
        },
        f"authoritative R0C/CP2 roadmap row changed: {match[0]}",
    )
    return {
        "status": "PASS",
        "scope_markdown": "PASS",
        "contract_json": "PASS",
        "fixture_csv": "PASS",
        "required_fixtures": len(required),
        "roadmap_row": match[0],
    }


def wheel_inventory(tree: Path) -> dict[str, str]:
    return {
        path.relative_to(tree).as_posix(): sha256_path(path)
        for path in sorted(tree.rglob("*.whl"))
        if path.is_file()
    }


def artifact(path: Path) -> dict[str, Any]:
    info = zip_info(path)
    require(info["status"] == "PASS", f"ZIP integrity failed: {json.dumps(info, indent=2)}")
    return info


def write_core_sha(output: Path, names: list[str]) -> None:
    rows = [f"{sha256_path(output / name)}  {name}" for name in names]
    (output / "C2A.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transport = args.transport.resolve()
    baseline = args.baseline.resolve()
    source_root = args.source_root.resolve()
    work = args.work.resolve()
    output = args.output.resolve()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)

    require(transport.stat().st_size == TRANSFER_BYTES, "C1_transfer byte size mismatch")
    require(sha256_path(transport) == TRANSFER_SHA256, "C1_transfer SHA-256 mismatch")
    transport_info = artifact(transport)
    require(sha256_path(baseline) == BASELINE_SHA256, "fixed baseline SHA-256 mismatch")
    baseline_info = artifact(baseline)

    transport_tree = work / "transport"
    pt.safe_extract(transport, transport_tree)
    manifest = parse_sha_manifest(transport_tree / "C1.sha256")
    for name, expected in AUTHORITY_SHA256.items():
        path = transport_tree / name
        require(path.is_file(), f"missing accepted authority file: {name}")
        require(sha256_path(path) == expected, f"{name} exact SHA-256 mismatch")
        require(zip_info(path)["status"] == "PASS", f"{name} ZIP integrity failure")
        require(manifest.get(name) == expected, f"{name} does not match delivered C1.sha256")
    require(
        (transport_tree / "C1_check.json").read_bytes()
        == (transport_tree / "C1_delivery.json").read_bytes(),
        "C1_check.json must be byte-equal to C1_delivery.json",
    )

    parent = work / "parent_c1"
    current = work / "current_c2a"
    baseline_tree = work / "baseline"
    pt.safe_extract(transport_tree / "C1.zip", parent)
    pt.safe_extract(baseline, baseline_tree)
    shutil.copytree(parent, current, copy_function=shutil.copy2)

    parent_registry = pt.validate_registry(parent)
    require(parent_registry["status"] == "PASS", f"C1 FILES.sha256 failed: {parent_registry}")
    require(parent_registry["tree_files_including_registry"] == 1418, f"C1 file count mismatch: {parent_registry}")

    copied: list[str] = []
    for rel in TARGETS:
        source = source_root / rel
        target = current / rel
        require(source.is_file(), f"staged freeze source missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel)
    pt.write_registry(current)

    current_registry = pt.validate_registry(current)
    require(current_registry["status"] == "PASS", f"CP2-A FILES.sha256 failed: {current_registry}")
    require(current_registry["tree_files_including_registry"] == 1421, f"CP2-A file count mismatch: {current_registry}")

    diff = pt.tree_diff(parent, current)
    require(diff["added"] == sorted(TARGETS), f"unexpected added paths: {diff['added']}")
    require(diff["modified"] == ["FILES.sha256"], f"unexpected modified paths: {diff['modified']}")
    require(diff["deleted"] == [], f"unexpected deleted paths: {diff['deleted']}")

    parent_wheels = wheel_inventory(parent)
    current_wheels = wheel_inventory(current)
    require(len(parent_wheels) == 58, f"accepted C1 wheelhouse count mismatch: {len(parent_wheels)}")
    require(parent_wheels == current_wheels, "CP2-A changed the accepted 58-wheel runtime closure")

    freeze_validation = verify_freeze_files(current)

    pre_full = output / "C2A_pre.zip"
    pre_patch = output / "C2A_preP.zip"
    pre_source = output / "C2A_preS.zip"
    source_meta = pt.create_source_delta(
        parent,
        current,
        pre_source,
        parent_name="C1.zip",
        stage="ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2_CP2-A_FREEZE_PUBLICATION",
    )
    patch_meta = build_cumulative_patch(
        transport_tree / "C1P.zip",
        pre_source,
        pre_patch,
        work / "patch_build",
    )
    pt.deterministic_zip(current, pre_full)

    pre_checkpoint = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-pre-checkpoint/1",
        "status": "PRESERVED_BEFORE_FOCUSED_VALIDATION",
        "authority": AUTHORITY_SHA256,
        "fixed_baseline_sha256": BASELINE_SHA256,
        "source_diff": diff,
        "artifacts": {
            "full": zip_info(pre_full),
            "patch": zip_info(pre_patch),
            "source": zip_info(pre_source),
        },
    }
    (output / "C2A_pre_check.json").write_text(
        json.dumps(pre_checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Focused packaging validation starts only after pre checkpoint bytes exist.
    full_reopen = pt.verify_zip_tree(pre_full, current, work / "verify_full")
    parent_plus_source = pt.verify_parent_plus_delta(parent, pre_source, current, work / "verify_source")
    patch_current_delta = pt.verify_patch_payload(pre_patch, current, pre_source, work / "verify_patch_payload")

    reconstructed_c1 = work / "reconstructed_c1"
    c1_patch_application = apply_cumulative_patch(
        baseline_tree,
        transport_tree / "C1P.zip",
        reconstructed_c1,
        work / "c1_patch_extract",
    )
    baseline_plus_c1p = pt.compare_trees(parent, reconstructed_c1)

    reconstructed_c2a = work / "reconstructed_c2a"
    c2a_patch_application = apply_cumulative_patch(
        baseline_tree,
        pre_patch,
        reconstructed_c2a,
        work / "c2a_patch_extract",
    )
    baseline_plus_c2ap = pt.compare_trees(current, reconstructed_c2a)

    validation = {
        "authority_transfer": transport_info,
        "fixed_baseline": baseline_info,
        "parent_registry": parent_registry,
        "current_registry": current_registry,
        "freeze_files": freeze_validation,
        "source_diff": diff,
        "wheelhouse": {
            "status": "PASS",
            "count": len(current_wheels),
            "unchanged": parent_wheels == current_wheels,
        },
        "full_reopen": full_reopen,
        "c1_plus_source_equals_full": parent_plus_source,
        "patch_current_delta_equals_source_payload": patch_current_delta,
        "baseline_plus_c1p_equals_c1": baseline_plus_c1p,
        "baseline_plus_c2ap_equals_full": baseline_plus_c2ap,
        "c1_patch_application": c1_patch_application,
        "c2a_patch_application": c2a_patch_application,
    }
    required_passes = [
        full_reopen["status"],
        parent_plus_source["status"],
        patch_current_delta["status"],
        baseline_plus_c1p["status"],
        baseline_plus_c2ap["status"],
        parent_registry["status"],
        current_registry["status"],
        freeze_validation["status"],
    ]
    validation["status"] = "PASS" if all(value == "PASS" for value in required_passes) else "FAIL"
    require(validation["status"] == "PASS", f"focused validation failed: {json.dumps(validation, indent=2)}")

    final_full = output / "C2A.zip"
    final_patch = output / "C2AP.zip"
    final_source = output / "C2AS.zip"
    shutil.copy2(pre_full, final_full)
    shutil.copy2(pre_patch, final_patch)
    shutil.copy2(pre_source, final_source)
    require(final_full.read_bytes() == pre_full.read_bytes(), "final full differs from preserved pre full")
    require(final_patch.read_bytes() == pre_patch.read_bytes(), "final patch differs from preserved pre patch")
    require(final_source.read_bytes() == pre_source.read_bytes(), "final source differs from preserved pre source")

    core_artifacts = {
        "C2A.zip": artifact(final_full),
        "C2AP.zip": artifact(final_patch),
        "C2AS.zip": artifact(final_source),
    }
    check = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-freeze-publication-check/1",
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
            "next_exact_checkpoint": "R0C-CP2 CP2-B"
        },
        "failed_attempts_preserved": FAILED_ATTEMPTS,
        "authority": {
            "transport": {
                "sha256": TRANSFER_SHA256,
                "bytes": TRANSFER_BYTES
            },
            "inner": AUTHORITY_SHA256,
            "fixed_baseline": {
                "name": "RCF_D0_full.zip",
                "sha256": BASELINE_SHA256
            }
        },
        "source_changes": {
            "added": diff["added"],
            "modified": diff["modified"],
            "deleted": diff["deleted"]
        },
        "pre_checkpoint": pre_checkpoint,
        "artifacts": core_artifacts,
        "source_delta": source_meta,
        "cumulative_patch": patch_meta,
        "validation": validation,
        "pre_final_byte_equality": {
            "full": True,
            "patch": True,
            "source": True
        }
    }
    (output / "C2A_check.json").write_text(
        json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    delivery = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-delivery/1",
        "status": "CP2_A_COMPLETE_CLOSED",
        "task": check["task"],
        "official_subtitle": check["official_subtitle"],
        "source_preservation": "PASS",
        "functional_qualification": "NOT_STARTED_PRODUCT",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "artifacts": core_artifacts,
        "equality": {
            "RCF_D0_full_plus_C2AP_equals_C2A": True,
            "C1_plus_C2AS_equals_C2A": True,
            "C2AP_current_delta_equals_C2AS_payload": True
        },
        "failed_attempts_preserved": FAILED_ATTEMPTS
    }
    (output / "C2A_delivery.json").write_text(
        json.dumps(delivery, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — CP2-A FREEZE PUBLICATION

## Verdict

```text
source preservation                         PASS
exact C1 authority admission                PASS
C1 FILES.sha256 closed-world replay         PASS
scope / contract / fixture freeze           PASS
fixed baseline + cumulative patch equality  PASS
C1 + source delta equality                  PASS
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

## Equality

```text
RCF_D0_full.zip + C2AP.zip = C2A.zip  PASS
C1.zip + C2AS.zip          = C2A.zip  PASS
C2AP current delta         = C2AS payload PASS
pre checkpoint bytes       = final bytes PASS
```

## Roadmap

```text
R0C progress          1 / 4
accepted checkpoints  10 / 25
R0C-CP2               OPEN
next                   R0C-CP2 CP2-B — explicit start only
```
"""
    (output / "C2A_report.md").write_text(report, encoding="utf-8")

    write_core_sha(
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
        shutil.copy2(output / name, receipts_tree / name)
    for name in ["AUTHORITY_INSPECTION.json", "AUTHORITY_INSPECTION.md"]:
        source = REPO_ROOT / "cp2a_authority_snapshot" / name
        require(source.is_file(), f"authority receipt missing: {source}")
        shutil.copy2(source, receipts_tree / name)
    receipts = output / "C2AR.zip"
    pt.deterministic_zip(receipts_tree, receipts)
    receipts_info = artifact(receipts)

    release = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2a-release/1",
        "status": "CP2_A_COMPLETE_CLOSED",
        "core_artifacts": core_artifacts,
        "receipts": receipts_info,
        "source_preservation": "PASS",
        "functional_qualification": "NOT_STARTED_PRODUCT",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "next": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 CP2-B"
    }
    (output / "C2A_release.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    notes = """# R0C-CP2 CP2-A freeze publication

CP2-A specification publication only. `R0C_CP2_PASS` is not claimed.

- exact C1 authority: PASS
- source preservation: PASS
- scope/contract/fixture freeze: PASS
- fixed-baseline cumulative patch equality: PASS
- C1 source-delta equality: PASS
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
        shutil.copy2(output / name, small_tree / name)
    small = output / "C2A_SMALL.zip"
    pt.deterministic_zip(small_tree, small)
    small_info = artifact(small)
    (output / "C2A_SMALL.sha256").write_text(
        f"{small_info['sha256']}  C2A_SMALL.zip\n", encoding="utf-8"
    )

    final_summary = {
        "status": "PASS",
        "core": core_artifacts,
        "receipts": receipts_info,
        "small_delivery": small_info,
        "validation": validation,
    }
    print(json.dumps(final_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
