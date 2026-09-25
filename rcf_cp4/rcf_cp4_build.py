#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import zipfile
from typing import Any

BASELINE_FILE = "RCF_D0_full.zip"
BASELINE_SHA256 = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
A3_SHA256 = "ffbe0be757beaed2b4b5f72125bd31e490b64c6cc0d2524da9957154048c729e"
A3P_SHA256 = "b958a30b41842ac367e7b770f9ea8fa068f444ea30e7e289a9a2d3fd4b56cfa4"
A3S_SHA256 = "6f022d9d1eda1a9bfee4ee9935172f1d343bf57fe0b53b8e0ff61b276656339b"
A3_SUMMARY_SHA256 = "a3fabe919f77e13d616d6f9a41ec1ad7ba5f54adc27b0491adc46b523fb5c975"
ZIP_TIME = (2026, 9, 25, 0, 0, 0)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError(f"unsafe archive path: {info.filename}")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"symlink is not allowed: {info.filename}")
            archive.extract(info, destination)
            target = destination / name.as_posix()
            if target.is_file() and unix_mode:
                target.chmod(stat.S_IMODE(unix_mode))


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def file_map(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        result[rel] = {
            "path": path,
            "sha256": sha256_path(path),
            "bytes": path.stat().st_size,
            "mode": file_mode(path),
        }
    return result


def manifest_digest(records: dict[str, dict[str, Any]], selected: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    names = sorted(selected if selected is not None else records)
    for rel in names:
        row = records[rel]
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(f"{row['mode']:04o}".encode("ascii") + b"\0")
        digest.update(row["sha256"].encode("ascii") + b"\n")
    return digest.hexdigest()


def compare_maps(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left_names, right_names = set(left), set(right)
    missing = sorted(left_names - right_names)
    extra = sorted(right_names - left_names)
    different = sorted(
        rel for rel in left_names & right_names
        if left[rel]["sha256"] != right[rel]["sha256"] or left[rel]["mode"] != right[rel]["mode"]
    )
    return {
        "equal": not missing and not extra and not different,
        "missing": missing,
        "extra": extra,
        "different": different,
        "left_count": len(left),
        "right_count": len(right),
    }


def copy_record(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def zip_tree(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for rel, row in file_map(root).items():
            info = zipfile.ZipInfo(rel)
            info.date_time = ZIP_TIME
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (row["mode"] & 0xFFFF) << 16
            archive.writestr(info, row["path"].read_bytes())
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"archive CRC failure {output}: {bad}")


def write_registry(tree: Path) -> None:
    registry = tree / "FILES.sha256"
    rows = []
    for rel, row in file_map(tree).items():
        if rel == "FILES.sha256":
            continue
        rows.append(f"{row['sha256']}  {rel}")
    registry.write_text("\n".join(rows) + "\n", encoding="utf-8")


def compute_delta(parent: Path, final: Path) -> dict[str, list[str]]:
    old, new = file_map(parent), file_map(final)
    added = sorted(set(new) - set(old))
    modified = sorted(
        rel for rel in set(new) & set(old)
        if new[rel]["sha256"] != old[rel]["sha256"] or new[rel]["mode"] != old[rel]["mode"]
    )
    deleted = sorted(set(old) - set(new))
    return {"added": added, "modified": modified, "deleted": deleted}


def overlay_tree(parent: Path, overlay: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(parent, destination, copy_function=shutil.copy2)
    for rel, row in file_map(overlay).items():
        copy_record(row["path"], destination / rel)


def ensure_input_hash(path: Path, expected: str) -> None:
    actual = sha256_path(path)
    if actual != expected:
        raise RuntimeError(f"input hash mismatch {path.name}: {actual} != {expected}")


def load_parent_summary(path: Path) -> dict[str, Any]:
    ensure_input_hash(path, A3_SUMMARY_SHA256)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or value.get("transitive_patch_proof") != "PASS":
        raise RuntimeError("accepted A3 package proof is not PASS")
    if value.get("full", {}).get("sha256") != A3_SHA256:
        raise RuntimeError("A3 full hash in summary is inconsistent")
    if value.get("patch", {}).get("sha256") != A3P_SHA256:
        raise RuntimeError("A3 patch hash in summary is inconsistent")
    if value.get("source", {}).get("sha256") != A3S_SHA256:
        raise RuntimeError("A3 source hash in summary is inconsistent")
    return value


def make_package_set(
    *,
    parent: Path,
    final: Path,
    parent_patch_zip: Path,
    parent_summary: dict[str, Any],
    deliver: Path,
    work: Path,
    prefix: str,
) -> dict[str, Any]:
    write_registry(final)
    delta = compute_delta(parent, final)
    if delta["deleted"]:
        raise RuntimeError(f"CP4 does not permit deletions: {delta['deleted']}")
    delta_paths = sorted(delta["added"] + delta["modified"])
    final_map = file_map(final)
    parent_map = file_map(parent)

    full_zip = deliver / f"{prefix}.zip"
    patch_zip = deliver / f"{prefix}P.zip"
    source_zip = deliver / f"{prefix}S.zip"
    zip_tree(final, full_zip)

    patch_tree = work / f"{prefix}_patch_tree"
    if patch_tree.exists():
        shutil.rmtree(patch_tree)
    safe_extract(parent_patch_zip, patch_tree)
    if not (patch_tree / "W").is_dir() or not (patch_tree / "D.txt").is_file():
        raise RuntimeError("accepted parent cumulative patch layout is invalid")
    for rel in delta_paths:
        copy_record(final_map[rel]["path"], patch_tree / "W" / rel)

    old_meta_path = patch_tree / "PATCH_METADATA.json"
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8")) if old_meta_path.is_file() else {}
    metadata = {
        **old_meta,
        "schema": "royal-capital.cumulative-patch/1",
        "baseline_file": BASELINE_FILE,
        "baseline_sha256": BASELINE_SHA256,
        "parent_full": "A3.zip",
        "parent_full_sha256": A3_SHA256,
        "parent_patch": "A3P.zip",
        "parent_patch_sha256": A3P_SHA256,
        "parent_source": "A3S.zip",
        "parent_source_sha256": A3S_SHA256,
        "parent_transitive_patch_proof": parent_summary["transitive_patch_proof"],
        "checkpoint": "R0A-CP4",
        "delta_added": len(delta["added"]),
        "delta_modified": len(delta["modified"]),
        "delta_deleted": 0,
        "delta_manifest_sha256": manifest_digest(final_map, delta_paths),
        "proof": "accepted A3 baseline proof + exact A3-to-final delta overlay",
    }
    old_meta_path.write_text(pretty_json(metadata), encoding="utf-8")
    zip_tree(patch_tree, patch_zip)

    source_tree = work / f"{prefix}_source_tree"
    if source_tree.exists():
        shutil.rmtree(source_tree)
    source_tree.mkdir(parents=True)
    for rel in delta_paths:
        copy_record(final_map[rel]["path"], source_tree / rel)
    zip_tree(source_tree, source_zip)

    verify_root = work / f"{prefix}_verify"
    if verify_root.exists():
        shutil.rmtree(verify_root)
    verify_root.mkdir(parents=True)
    full_reopen = verify_root / "full"
    source_reopen = verify_root / "source"
    patch_reopen = verify_root / "patch"
    safe_extract(full_zip, full_reopen)
    safe_extract(source_zip, source_reopen)
    safe_extract(patch_zip, patch_reopen)

    full_comparison = compare_maps(final_map, file_map(full_reopen))
    if not full_comparison["equal"]:
        raise RuntimeError(f"full reopen mismatch: {full_comparison}")

    source_map = file_map(source_reopen)
    if set(source_map) != set(delta_paths):
        raise RuntimeError(f"source entry set mismatch missing={sorted(set(delta_paths)-set(source_map))} extra={sorted(set(source_map)-set(delta_paths))}")
    for rel in delta_paths:
        if source_map[rel]["sha256"] != final_map[rel]["sha256"] or source_map[rel]["mode"] != final_map[rel]["mode"]:
            raise RuntimeError(f"source/full mismatch at {rel}")

    reconstructed = verify_root / "parent_plus_source"
    overlay_tree(parent, source_reopen, reconstructed)
    source_reconstruction = compare_maps(final_map, file_map(reconstructed))
    if not source_reconstruction["equal"]:
        raise RuntimeError(f"parent + source != final: {source_reconstruction}")

    patch_w = patch_reopen / "W"
    cumulative_w_map = file_map(patch_w)
    deleted = {
        line.strip() for line in (patch_reopen / "D.txt").read_text(encoding="utf-8").splitlines() if line.strip()
    }
    bad_cumulative = []
    for rel, row in cumulative_w_map.items():
        if rel not in final_map or row["sha256"] != final_map[rel]["sha256"] or row["mode"] != final_map[rel]["mode"]:
            bad_cumulative.append(rel)
    for rel in deleted:
        if rel in final_map:
            bad_cumulative.append(f"deleted-but-present:{rel}")
    if bad_cumulative:
        raise RuntimeError(f"cumulative patch/full mismatch: {bad_cumulative[:30]}")
    for rel in delta_paths:
        patch_row = cumulative_w_map.get(rel)
        if patch_row is None or patch_row["sha256"] != source_map[rel]["sha256"] or patch_row["mode"] != source_map[rel]["mode"]:
            raise RuntimeError(f"patch/source delta mismatch at {rel}")

    result = {
        "schema": "royal-capital.fortification.cp4-package-equality/1",
        "status": "PASS",
        "fixed_baseline": {"file": BASELINE_FILE, "sha256": BASELINE_SHA256},
        "accepted_parent": {
            "full_sha256": A3_SHA256,
            "patch_sha256": A3P_SHA256,
            "source_sha256": A3S_SHA256,
            "transitive_patch_proof": parent_summary["transitive_patch_proof"],
        },
        "delta": {
            "added": len(delta["added"]),
            "modified": len(delta["modified"]),
            "deleted": 0,
            "paths": delta_paths,
            "manifest_sha256": manifest_digest(final_map, delta_paths),
        },
        "checks": {
            "full_zip_reopen_exact": full_comparison["equal"],
            "source_entry_set_exact": set(source_map) == set(delta_paths),
            "parent_plus_source_equals_full": source_reconstruction["equal"],
            "cumulative_patch_payload_matches_full": not bad_cumulative,
            "patch_delta_equals_source_delta": True,
            "parent_baseline_proof_carried": parent_summary["transitive_patch_proof"] == "PASS",
        },
        "full": {"file": full_zip.name, "sha256": sha256_path(full_zip), "bytes": full_zip.stat().st_size, "entries": len(final_map)},
        "patch": {"file": patch_zip.name, "sha256": sha256_path(patch_zip), "bytes": patch_zip.stat().st_size, "entries": len(file_map(patch_reopen)), "cumulative_payload_entries": len(cumulative_w_map)},
        "source": {"file": source_zip.name, "sha256": sha256_path(source_zip), "bytes": source_zip.stat().st_size, "entries": len(source_map)},
    }
    return result


def write_prepare_sources(tree: Path, template: Path) -> None:
    fort = tree / "RC_K0/child_designs/fortification"
    destination = fort / "r0a_cp4"
    if destination.exists():
        raise RuntimeError("R0A CP4 already exists in parent checkpoint")
    shutil.copytree(template, destination, copy_function=shutil.copy2)
    (destination / "docs").mkdir(parents=True, exist_ok=True)
    (destination / "reports").mkdir(parents=True, exist_ok=True)
    (destination / "provenance").mkdir(parents=True, exist_ok=True)
    (destination / "docs/CP4_IMPLEMENTATION.md").write_text(
        "# R0A-CP4 implementation checkpoint\n\n"
        "The closeout source and replay tool are preserved before fresh runtime materialization. "
        "Stage completion remains open until exact A/B replay and final package equality pass.\n",
        encoding="utf-8",
    )
    source_preservation = {
        "schema": "royal-capital.fortification.cp4-source-preservation/1",
        "status": "PASS",
        "parent_full": {"file": "A3.zip", "sha256": A3_SHA256},
        "parent_patch": {"file": "A3P.zip", "sha256": A3P_SHA256},
        "parent_source": {"file": "A3S.zip", "sha256": A3S_SHA256},
        "fixed_baseline": {"file": BASELINE_FILE, "sha256": BASELINE_SHA256},
        "runtime_replay": "PENDING",
        "stage_completion": "OPEN",
    }
    (destination / "reports/source_preservation.json").write_text(pretty_json(source_preservation), encoding="utf-8")
    changeset = {
        "schema": "royal-capital.fortification.cp4-changeset/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP4",
        "parent_checkpoint": "A3.zip",
        "initial_implementation_baseline": BASELINE_FILE,
        "scope": ["exact runtime replay", "final full/patch/source equality", "R0A closeout"],
        "geometry_changes": False,
        "source_preserved_before_validation": True,
    }
    (destination / "provenance/CHANGESET.json").write_text(pretty_json(changeset), encoding="utf-8")


def update_cp_status(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    fieldnames = list(rows[0])
    rows = [row for row in rows if not (row.get("stage") == "R0A" and row.get("checkpoint") == "CP4")]
    rows.append({
        "stage": "R0A",
        "checkpoint": "CP4",
        "source_preservation": "PASS",
        "remote_self_admission": "NOT_APPLICABLE",
        "local_runtime_admission": "PASS_X2",
        "functional_qualification": "PASS",
        "stage_completion": "R0A_COMPLETE",
        "next": "START_R0B_CP1_BY_EXPLICIT_APPROVAL",
        "blocking_reason": "NONE",
    })
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def update_roadmap_header(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacement = """```text
CURRENT                 RC-FORT-R0A COMPLETE
NEXT                    RC-FORT-R0B-CP1 — EXPLICIT APPROVAL REQUIRED
TERMINAL                RC-FORT-R0F-CP4
AUTO_CONTINUATION       FORBIDDEN
KERNEL CONSUMER         ROYAL-CAPITAL R0D-CP1
```"""
    updated, count = re.subn(r"```text\nCURRENT.*?```", replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("roadmap header replacement failed")
    path.write_text(updated, encoding="utf-8")


def runtime_replay_summary(tree: Path, replay: Path) -> dict[str, Any]:
    fort = tree / "RC_K0/child_designs/fortification"
    accepted = fort / "r0a_cp3/outputs/reference"
    a = replay / "A"
    b = replay / "B"
    a_product, b_product, accepted_product = file_map(a / "product"), file_map(b / "product"), file_map(accepted)
    ab = compare_maps(a_product, b_product)
    ar = compare_maps(a_product, accepted_product)
    br = compare_maps(b_product, accepted_product)
    capability_equal = (a / "capability/canonical.json").read_bytes() == (b / "capability/canonical.json").read_bytes()
    freeze_equal = (replay / "pip-freeze-A.txt").read_bytes() == (replay / "pip-freeze-B.txt").read_bytes()
    receipts = [json.loads((root / "runtime-receipt.json").read_text(encoding="utf-8")) for root in (a, b)]
    identity_fields = [
        (row["python"]["version"], row["python"]["implementation"], row["python"]["cache_tag"], row["python"]["isolated_venv"],
         row["runtime"]["build123d_version"], row["runtime"]["ocp_version"], row["runtime"]["wheel_count"],
         row["runtime"]["wheelhouse_manifest_sha256"], row["runtime"]["requirements_lock_sha256"], row["runtime"]["global_install"])
        for row in receipts
    ]
    identity_equal = identity_fields[0] == identity_fields[1]
    exact_identity = all(
        row["status"] == "PASS"
        and row["python"]["version"] == "3.13.5"
        and row["python"]["isolated_venv"] is True
        and row["runtime"]["build123d_version"] == "0.13.1.dev12+ge22d34dae"
        and row["runtime"]["ocp_version"] == "8.0.1.0.0"
        and row["runtime"]["wheel_count"] == 58
        and row["runtime"]["global_install"] is False
        for row in receipts
    )
    pip_checks = all("No broken requirements found." in (replay / f"pip-check-{label}.log").read_text(encoding="utf-8") for label in ("A", "B"))
    checks = {
        "runtime_identity_exact": exact_identity,
        "runtime_identity_A_equals_B": identity_equal,
        "pip_check_A_B": pip_checks,
        "pip_freeze_A_equals_B": freeze_equal,
        "capability_canonical_A_equals_B": capability_equal,
        "product_A_equals_B": ab["equal"],
        "product_A_equals_accepted_reference": ar["equal"],
        "product_B_equals_accepted_reference": br["equal"],
    }
    result = {
        "schema": "royal-capital.fortification.cp4-exact-runtime-replay/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "comparisons": {"A_vs_B": ab, "A_vs_reference": ar, "B_vs_reference": br},
        "product_files_compared": len(accepted_product),
        "capability_canonical_sha256": sha256_path(a / "capability/canonical.json"),
        "runtime_receipts": receipts,
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"exact runtime replay failed: {result}")
    return result


def copy_runtime_receipts(tree: Path, replay: Path) -> None:
    destination = tree / "RC_K0/child_designs/fortification/r0a_cp4/reports/runtime"
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "install-A.log", "install-B.log", "pip-check-A.log", "pip-check-B.log",
        "pip-freeze-A.txt", "pip-freeze-B.txt",
    ]
    for name in names:
        copy_record(replay / name, destination / name)
    for label in ("A", "B"):
        copy_record(replay / label / "runtime-receipt.json", destination / f"runtime-{label}.json")
        copy_record(replay / label / "capability/result.json", destination / f"capability-{label}.json")
        copy_record(replay / label / "capability/canonical.json", destination / f"capability-{label}-canonical.json")


def validate_final_tree(tree: Path, replay_result: dict[str, Any]) -> dict[str, Any]:
    fort = tree / "RC_K0/child_designs/fortification"
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    check("exact_runtime_replay", replay_result["status"] == "PASS")
    check("wheel_count_retained", len(list((fort / "r0a_cp0/runtime/wheelhouse").glob("*.whl"))) == 58)
    required = [
        fort / "r0a_cp4/README.md",
        fort / "r0a_cp4/docs/CP4_IMPLEMENTATION.md",
        fort / "r0a_cp4/docs/CP4_CLOSEOUT.md",
        fort / "r0a_cp4/reports/source_preservation.json",
        fort / "r0a_cp4/reports/exact_runtime_replay.json",
        fort / "r0a_cp4/reports/r0a_closeout.json",
        fort / "r0a_cp4/provenance/CHANGESET.json",
    ]
    check("required_closeout_files", all(path.is_file() for path in required), [str(path) for path in required if not path.is_file()])
    cp_rows = list(csv.DictReader((fort / "data/CP_STATUS.csv").open(encoding="utf-8", newline="")))
    cp4_rows = [row for row in cp_rows if row["stage"] == "R0A" and row["checkpoint"] == "CP4"]
    check("cp4_status_row", len(cp4_rows) == 1 and cp4_rows[0]["stage_completion"] == "R0A_COMPLETE", cp4_rows)
    check("r0a_rows_complete", {row["checkpoint"] for row in cp_rows if row["stage"] == "R0A"} == {"CP0", "CP1", "CP2", "CP3", "CP4"})
    parse_failures = []
    for path in fort.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_failures.append({"path": str(path), "error": str(exc)})
    for path in fort.rglob("*.csv"):
        try:
            list(csv.reader(path.open(encoding="utf-8", newline="")))
        except Exception as exc:
            parse_failures.append({"path": str(path), "error": str(exc)})
    check("machine_readable_parse", not parse_failures, parse_failures)
    caches = [
        path.relative_to(tree).as_posix() for path in tree.rglob("*")
        if (path.is_file() and path.suffix in {".pyc", ".pyo"}) or (path.is_dir() and path.name in {"__pycache__", ".godot"})
    ]
    check("no_generated_cache", not caches, caches)
    check("godot_product_not_claimed", "Godot product                NOT_STARTED" in (fort / "docs/00_STATUS.md").read_text(encoding="utf-8"))
    result = {
        "schema": "royal-capital.fortification.cp4-final-tree-validation/1",
        "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
        "checks": checks,
        "summary": {"passed": sum(row["pass"] for row in checks), "failed": sum(not row["pass"] for row in checks)},
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"final tree validation failed: {result}")
    return result


def prepare(args: argparse.Namespace) -> None:
    a3, a3p, a3s, a3_summary = map(Path, (args.a3, args.a3p, args.a3s, args.a3_summary))
    ensure_input_hash(a3, A3_SHA256)
    ensure_input_hash(a3p, A3P_SHA256)
    ensure_input_hash(a3s, A3S_SHA256)
    parent_summary = load_parent_summary(a3_summary)
    with zipfile.ZipFile(a3s) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("A3 source archive CRC failed")

    work = Path(args.work).resolve()
    deliver = Path(args.deliver).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    deliver.mkdir(parents=True, exist_ok=True)
    parent = work / "parent"
    tree = work / "tree"
    safe_extract(a3, parent)
    shutil.copytree(parent, tree, copy_function=shutil.copy2)
    write_prepare_sources(tree, Path(args.template).resolve())

    package = make_package_set(
        parent=parent,
        final=tree,
        parent_patch_zip=a3p,
        parent_summary=parent_summary,
        deliver=deliver,
        work=work,
        prefix="A4_pre",
    )
    check = {
        "schema": "royal-capital.fortification.cp4-prevalidation-checkpoint/1",
        "status": "PASS",
        "source_preservation": "PASS",
        "runtime_replay": "NOT_STARTED",
        "stage_completion": "OPEN",
        "package_equality": package,
    }
    (deliver / "A4_pre_check.json").write_text(pretty_json(check), encoding="utf-8")
    print(pretty_json(check), end="")


def finalize(args: argparse.Namespace) -> None:
    work = Path(args.work).resolve()
    deliver = Path(args.deliver).resolve()
    parent = work / "parent"
    tree = work / "tree"
    replay = Path(args.replay).resolve()
    a3p = Path(args.a3p).resolve()
    parent_summary = load_parent_summary(Path(args.a3_summary).resolve())
    fort = tree / "RC_K0/child_designs/fortification"
    cp4 = fort / "r0a_cp4"

    replay_result = runtime_replay_summary(tree, replay)
    copy_runtime_receipts(tree, replay)
    (cp4 / "reports/exact_runtime_replay.json").write_text(pretty_json(replay_result), encoding="utf-8")

    update_cp_status(fort / "data/CP_STATUS.csv")
    update_roadmap_header(fort / "docs/06_ROADMAP.md")
    (fort / "docs/00_STATUS.md").write_text(
        "# 상태\n\n```text\n"
        "task                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP4\n"
        "resume checkpoint            A3.zip\n"
        "initial implementation base  RCF_D0_full.zip\n"
        "CP0 runtime                  COMPLETE / ADMITTED\n"
        "CP1 provider adapter         COMPLETE / QUALIFIED\n"
        "CP2 straight wall-span       COMPLETE / QUALIFIED\n"
        "CP3 coverage/gates           COMPLETE / QUALIFIED\n"
        "exact runtime A/B            PASS / CPYTHON 3.13.5 / 58 WHEELS\n"
        "capability replay            PASS / BYTE-IDENTICAL CANONICAL\n"
        "product replay               A == B == ACCEPTED CP3 REFERENCE\n"
        "full/patch/source equality   PASS\n"
        "stage completion             R0A_COMPLETE / CLOSED\n"
        "next                         R0B-CP1 — EXPLICIT APPROVAL REQUIRED\n"
        "auto continuation            FORBIDDEN\n"
        "```\n\n```text\n"
        "source preservation          PASS\n"
        "runtime admission            PASS_X2\n"
        "functional qualification     PASS\n"
        "package equality             PASS\n"
        "stage completion             COMPLETE\n"
        "Godot product                NOT_STARTED\n"
        "```\n",
        encoding="utf-8",
    )
    (fort / "README.md").write_text(
        "# RCF — ROYAL-CAPITAL FORTIFICATION CAD 상세 설계·구현 체크포인트\n\n"
        "## 현재 판정\n\n```text\n"
        "child design                  RC-FORT-R0-DESIGN COMPLETE\n"
        "R0A                           COMPLETE / CLOSED\n"
        "R0A checkpoints               5 / 5 COMPLETE\n"
        "exact runtime replay          PASS_X2\n"
        "final package equality        PASS\n"
        "accepted implementation CP    5 / 25\n"
        "R0B-CP1 start                 ALLOWED BY EXPLICIT APPROVAL ONLY\n"
        "auto continuation             FORBIDDEN\n"
        "```\n\n"
        "## 공식 다음 작업\n\n```text\n"
        "ROYAL-CAPITAL-FORTIFICATION-CAD-R0B-CP1\n"
        "— STRAIGHT AND CURVED SPANS,\n"
        "  CANONICAL CENTERLINE\n"
        "  & LOCAL FRAME\n"
        "```\n",
        encoding="utf-8",
    )
    (cp4 / "docs/CP4_CLOSEOUT.md").write_text(
        "# R0A-CP4 closeout\n\n```text\n"
        "source preservation          PASS\n"
        "fresh exact runtimes         2 / 2 PASS\n"
        "capability canonical A/B     BYTE-IDENTICAL\n"
        "CP1→CP2→CP3 product A/B      BYTE-IDENTICAL\n"
        "accepted reference replay    BYTE-IDENTICAL\n"
        "full/patch/source equality   PASS\n"
        "R0A stage                    COMPLETE / CLOSED\n"
        "next                         R0B-CP1 BY EXPLICIT APPROVAL\n"
        "```\n",
        encoding="utf-8",
    )
    closeout = {
        "schema": "royal-capital.fortification.r0a-closeout/1",
        "status": "PASS",
        "stage": "R0A",
        "completed_checkpoints": ["CP0", "CP1", "CP2", "CP3", "CP4"],
        "source_preservation": "PASS",
        "exact_runtime_replay": "PASS_X2",
        "functional_qualification": "PASS",
        "package_equality": "PASS",
        "stage_completion": "R0A_COMPLETE",
        "godot_product": "NOT_STARTED",
        "next": "R0B-CP1",
        "next_requires_explicit_approval": True,
        "auto_continuation": False,
    }
    (cp4 / "reports/r0a_closeout.json").write_text(pretty_json(closeout), encoding="utf-8")
    changeset_path = cp4 / "provenance/CHANGESET.json"
    changeset = json.loads(changeset_path.read_text(encoding="utf-8"))
    changeset.update({"runtime_replay": "PASS_X2", "stage_completion": "R0A_COMPLETE", "next": "R0B-CP1_EXPLICIT_APPROVAL"})
    changeset_path.write_text(pretty_json(changeset), encoding="utf-8")

    # Remove generated caches before final validation and packaging.
    for path in sorted(tree.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
    for path in list(tree.rglob("*.pyc")) + list(tree.rglob("*.pyo")):
        path.unlink(missing_ok=True)

    validation = validate_final_tree(tree, replay_result)
    (cp4 / "reports/cp4_validation.json").write_text(pretty_json(validation), encoding="utf-8")
    # Revalidate after adding the validation receipt itself.
    validation = validate_final_tree(tree, replay_result)

    package = make_package_set(
        parent=parent,
        final=tree,
        parent_patch_zip=a3p,
        parent_summary=parent_summary,
        deliver=deliver,
        work=work,
        prefix="A4",
    )
    summary = {
        "schema": "royal-capital.fortification.cp4-package/1",
        "status": "PASS",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP4",
        "fixed_baseline": {"file": BASELINE_FILE, "sha256": BASELINE_SHA256},
        "source_preservation": "PASS",
        "exact_runtime_replay": replay_result,
        "final_tree_validation": validation,
        "package_equality": package,
        "stage_completion": "R0A_COMPLETE",
        "next": "R0B-CP1_BY_EXPLICIT_APPROVAL",
        "auto_continuation": "FORBIDDEN",
    }
    (deliver / "A4.json").write_text(pretty_json(summary), encoding="utf-8")
    (deliver / "A4.md").write_text(
        "# R0A-CP4 complete\n\n"
        "Two fresh CPython 3.13.5 runtimes were materialized from the retained 58-wheel offline lock. "
        "Their capability results and complete CP1→CP2→CP3 product outputs are byte-identical to each other and to the accepted CP3 reference. "
        "The final full, cumulative fixed-baseline patch and source-delta bundles were generated from one final tree and passed reopen, reconstruction and payload-equality checks. "
        "R0A is complete; R0B-CP1 requires explicit approval.\n",
        encoding="utf-8",
    )
    (deliver / "A4.txt").write_text(
        "R0A_COMPLETE\nEXACT_RUNTIME_REPLAY_PASS_X2\nFULL_PATCH_SOURCE_EQUALITY_PASS\nR0B_CP1_EXPLICIT_APPROVAL_REQUIRED\nAUTO_CONTINUATION_FORBIDDEN\n",
        encoding="utf-8",
    )
    (deliver / "A4_package_equality.json").write_text(pretty_json(package), encoding="utf-8")
    names = [
        "A4.zip", "A4P.zip", "A4S.zip", "A4.md", "A4.json", "A4.txt", "A4_package_equality.json",
        "A4_pre.zip", "A4_preP.zip", "A4_preS.zip", "A4_pre_check.json",
    ]
    (deliver / "A4.sha256").write_text(
        "\n".join(f"{sha256_path(deliver / name)}  {name}" for name in names) + "\n",
        encoding="utf-8",
    )
    print(pretty_json(summary), end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "finalize"))
    parser.add_argument("--a3", required=True)
    parser.add_argument("--a3p", required=True)
    parser.add_argument("--a3s", required=True)
    parser.add_argument("--a3-summary", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--deliver", required=True)
    parser.add_argument("--replay")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    else:
        if not args.replay:
            parser.error("--replay is required for finalize")
        finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
