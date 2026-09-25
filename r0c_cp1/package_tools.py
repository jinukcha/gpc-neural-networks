from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable
import zipfile

A4_SHA256 = "8f0d0a6667c8521c6f2720f2580304c4e06b480bd9e470384bc2828516edf751"
A4P_SHA256 = "bd4b32d7c35f413bae9aa9bf781bffdfe820d1a3c8f04d57c078a71d1425e335"
B4_CLAIMED_SHA256 = "4f66c61f4f2280fbfdd81c0d3056f3f2b705dab4baece5c21a65b93e3bdfc641"
BASELINE_SHA256 = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
CHAIN_SHA256 = {
    "B1S.zip": "17991d4cf4f273c2927bf585900310940f8c091ba724cbe8ac2882ab4f9e708b",
    "B2S.zip": "4a9c7251ba0bb1e67eb73fb83bb9d96866424960ef6e141bd9ec5b1ccb53662d",
    "B3S.zip": "ee3c3d7cb85b374830f530f2c8e2410f8b23f5a94fa9746e418b99b4de8a8c58",
    "B4S.zip": "a2f241fc0b3cb8fd391843ef04458b482b12e8e42da9ea63d00c3379cb2bb1ac",
}
TRANSPORT_METADATA = {"D.txt", "SOURCE_DELTA.json"}


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe ZIP member: {name}")
            zf.extract(info, destination)
            if not info.is_dir():
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if unix_mode:
                    os.chmod(destination / name, stat.S_IMODE(unix_mode))


def files(root: Path, *, include_registry: bool = True) -> dict[str, Path]:
    result = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not include_registry and rel == "FILES.sha256":
            continue
        result[rel] = path
    return result


def normalized_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def file_equal(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes() and normalized_mode(a) == normalized_mode(b)


def write_registry(tree: Path) -> None:
    rows = []
    for rel, path in sorted(files(tree, include_registry=False).items()):
        rows.append(f"{sha256_path(path)}  {rel}")
    (tree / "FILES.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate_registry(tree: Path) -> dict[str, Any]:
    registry = tree / "FILES.sha256"
    expected: dict[str, str] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            expected[rel] = digest
    actual = {rel: sha256_path(path) for rel, path in files(tree, include_registry=False).items()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(rel for rel in set(expected) & set(actual) if expected[rel] != actual[rel])
    return {
        "status": "PASS" if not missing and not extra and not mismatched else "FAIL",
        "registry_entries": len(expected),
        "tree_files_including_registry": len(actual) + 1,
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
    }


def _copy_zip_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> None:
    target = destination / info.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(zf.read(info))
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if unix_mode:
        os.chmod(target, stat.S_IMODE(unix_mode))


def apply_source_delta(tree: Path, delta_zip: Path) -> dict[str, Any]:
    with zipfile.ZipFile(delta_zip) as zf:
        names = zf.namelist()
        deleted = []
        if "D.txt" in names:
            deleted = [line.strip() for line in zf.read("D.txt").decode("utf-8").splitlines() if line.strip()]
        payload = [info for info in zf.infolist() if not info.is_dir() and info.filename not in TRANSPORT_METADATA]
        for info in payload:
            _copy_zip_member(zf, info, tree)
        for rel in deleted:
            path = tree / rel
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    return {"archive": delta_zip.name, "archive_entries": len(names), "payload_entries": len(payload), "deleted": deleted}


def reconstruct_b4(inputs: Path, work: Path) -> tuple[Path, dict[str, Any]]:
    a4 = inputs / "A4.zip"
    if sha256_path(a4) != A4_SHA256:
        raise RuntimeError("A4 SHA-256 mismatch")
    for name, expected in CHAIN_SHA256.items():
        if sha256_path(inputs / name) != expected:
            raise RuntimeError(f"{name} SHA-256 mismatch")
    tree = work / "parent_b4"
    if tree.exists():
        shutil.rmtree(tree)
    safe_extract(a4, tree)
    stages = [apply_source_delta(tree, inputs / name) for name in CHAIN_SHA256]
    validation = validate_registry(tree)
    validation["expected_file_count"] = 1301
    validation["file_count_match"] = validation["tree_files_including_registry"] == 1301
    validation["accepted_parent_claimed_zip_sha256"] = B4_CLAIMED_SHA256
    validation["reconstruction_basis"] = "A4_ACCEPTED_FULL_PLUS_B1S_B2S_B3S_B4S_EXACT_DELTAS"
    validation["stages"] = stages
    if validation["status"] != "PASS" or not validation["file_count_match"]:
        raise RuntimeError(f"B4 reconstruction failed: {json.dumps(validation, indent=2)}")
    return tree, validation


def deterministic_zip(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for rel, path in sorted(files(root).items()):
            info = zipfile.ZipInfo(rel)
            info.date_time = (2026, 9, 25, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (normalized_mode(path) & 0xFFFF) << 16
            zf.writestr(info, path.read_bytes())


def tree_diff(parent: Path, current: Path) -> dict[str, list[str]]:
    before = files(parent)
    after = files(current)
    added = sorted(set(after) - set(before))
    deleted = sorted(set(before) - set(after))
    modified = sorted(rel for rel in set(before) & set(after) if not file_equal(before[rel], after[rel]))
    return {"added": added, "modified": modified, "deleted": deleted}


def create_source_delta(parent: Path, current: Path, destination: Path, *, parent_name: str, stage: str) -> dict[str, Any]:
    diff = tree_diff(parent, current)
    temporary = destination.with_suffix(".tree")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for rel in diff["added"] + diff["modified"]:
        target = temporary / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current / rel, target)
    (temporary / "D.txt").write_text("\n".join(diff["deleted"]) + ("\n" if diff["deleted"] else ""), encoding="utf-8")
    metadata = {
        "schema": "royal-capital.source-delta/1",
        "stage": stage,
        "parent": parent_name,
        "added": len(diff["added"]),
        "modified": len(diff["modified"]),
        "deleted": len(diff["deleted"]),
        "payload_entries": len(diff["added"]) + len(diff["modified"]),
        "raw_wheels_included": False,
    }
    (temporary / "SOURCE_DELTA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    deterministic_zip(temporary, destination)
    shutil.rmtree(temporary)
    return {**metadata, "paths": diff}


def _overlay_source_delta_into_patch(patch_root: Path, source_delta: Path, deleted_paths: set[str]) -> None:
    write_root = patch_root / "W"
    with zipfile.ZipFile(source_delta) as zf:
        names = zf.namelist()
        if "D.txt" in names:
            deleted_paths.update(line.strip() for line in zf.read("D.txt").decode("utf-8").splitlines() if line.strip())
        for info in zf.infolist():
            if info.is_dir() or info.filename in TRANSPORT_METADATA:
                continue
            target = write_root / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode:
                os.chmod(target, stat.S_IMODE(unix_mode))
            deleted_paths.discard(info.filename)


def create_cumulative_patch(inputs: Path, current_source_delta: Path, destination: Path, *, stage: str) -> dict[str, Any]:
    a4p = inputs / "A4P.zip"
    if sha256_path(a4p) != A4P_SHA256:
        raise RuntimeError("A4P SHA-256 mismatch")
    patch_root = destination.with_suffix(".tree")
    if patch_root.exists():
        shutil.rmtree(patch_root)
    safe_extract(a4p, patch_root)
    deleted_paths: set[str] = set()
    d_file = patch_root / "D.txt"
    if d_file.is_file():
        deleted_paths.update(line.strip() for line in d_file.read_text(encoding="utf-8").splitlines() if line.strip())
    for name in CHAIN_SHA256:
        _overlay_source_delta_into_patch(patch_root, inputs / name, deleted_paths)
    _overlay_source_delta_into_patch(patch_root, current_source_delta, deleted_paths)
    d_file.write_text("\n".join(sorted(deleted_paths)) + ("\n" if deleted_paths else ""), encoding="utf-8")
    metadata = {
        "schema": "royal-capital.cumulative-patch/1",
        "stage": stage,
        "baseline_file": "RCF_D0_full.zip",
        "baseline_sha256": BASELINE_SHA256,
        "accepted_parent_full": "B4.zip",
        "accepted_parent_claimed_sha256": B4_CLAIMED_SHA256,
        "accepted_ancestor_full": "A4.zip",
        "accepted_ancestor_sha256": A4_SHA256,
        "accepted_ancestor_cumulative_patch": "A4P.zip",
        "accepted_ancestor_cumulative_patch_sha256": A4P_SHA256,
        "chain_deltas": CHAIN_SHA256,
        "current_delta": current_source_delta.name,
        "deleted_count": len(deleted_paths),
        "proof": "RCF_D0+A4P=A4 accepted; A4+B1S+B2S+B3S+B4S=B4 closed-world registry; B4+current delta=current; cumulative W overlay therefore reconstructs current from fixed baseline",
    }
    (patch_root / "PATCH_METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    deterministic_zip(patch_root, destination)
    shutil.rmtree(patch_root)
    return metadata


def compare_trees(expected: Path, observed: Path) -> dict[str, Any]:
    a = files(expected)
    b = files(observed)
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    different = sorted(rel for rel in set(a) & set(b) if not file_equal(a[rel], b[rel]))
    return {"status": "PASS" if not missing and not extra and not different else "FAIL", "missing": missing, "extra": extra, "different": different, "expected_files": len(a), "observed_files": len(b)}


def verify_zip_tree(archive: Path, expected_tree: Path, scratch: Path) -> dict[str, Any]:
    if scratch.exists():
        shutil.rmtree(scratch)
    with zipfile.ZipFile(archive) as zf:
        bad_crc = zf.testzip()
        zf.extractall(scratch)
        for info in zf.infolist():
            if info.is_dir():
                continue
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode:
                os.chmod(scratch / info.filename, stat.S_IMODE(unix_mode))
    comparison = compare_trees(expected_tree, scratch)
    comparison["bad_crc"] = bad_crc
    comparison["status"] = "PASS" if comparison["status"] == "PASS" and bad_crc is None else "FAIL"
    shutil.rmtree(scratch)
    return comparison


def verify_parent_plus_delta(parent: Path, source_delta: Path, expected_tree: Path, scratch: Path) -> dict[str, Any]:
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(parent, scratch, copy_function=shutil.copy2)
    apply_source_delta(scratch, source_delta)
    comparison = compare_trees(expected_tree, scratch)
    shutil.rmtree(scratch)
    return comparison


def verify_patch_payload(patch_zip: Path, expected_tree: Path, source_delta: Path, scratch: Path) -> dict[str, Any]:
    if scratch.exists():
        shutil.rmtree(scratch)
    safe_extract(patch_zip, scratch)
    source_changed = set()
    source_deleted = set()
    with zipfile.ZipFile(source_delta) as zf:
        for name in zf.namelist():
            if name == "D.txt":
                source_deleted.update(line.strip() for line in zf.read(name).decode("utf-8").splitlines() if line.strip())
            elif name not in TRANSPORT_METADATA and not name.endswith("/"):
                source_changed.add(name)
    mismatched = []
    for rel in sorted(source_changed):
        path = scratch / "W" / rel
        if not path.is_file() or not file_equal(path, expected_tree / rel):
            mismatched.append(rel)
    patch_deleted = set(line.strip() for line in (scratch / "D.txt").read_text(encoding="utf-8").splitlines() if line.strip())
    missing_deletions = sorted(source_deleted - patch_deleted)
    result = {"status": "PASS" if not mismatched and not missing_deletions else "FAIL", "current_delta_payload_mismatch": mismatched, "missing_deletions": missing_deletions, "current_delta_files": len(source_changed)}
    shutil.rmtree(scratch)
    return result


def package_checkpoint(
    *,
    parent: Path,
    current: Path,
    inputs: Path,
    deliver: Path,
    prefix: str,
    stage: str,
    scratch_root: Path,
) -> dict[str, Any]:
    deliver.mkdir(parents=True, exist_ok=True)
    write_registry(current)
    full = deliver / f"{prefix}.zip"
    source = deliver / f"{prefix}S.zip"
    patch = deliver / f"{prefix}P.zip"
    source_meta = create_source_delta(parent, current, source, parent_name="B4.zip", stage=stage)
    patch_meta = create_cumulative_patch(inputs, source, patch, stage=stage)
    deterministic_zip(current, full)
    verification = {
        "full_reopen": verify_zip_tree(full, current, scratch_root / f"{prefix}_full"),
        "parent_plus_source": verify_parent_plus_delta(parent, source, current, scratch_root / f"{prefix}_source"),
        "patch_current_delta": verify_patch_payload(patch, current, source, scratch_root / f"{prefix}_patch"),
    }
    verification["status"] = "PASS" if all(row["status"] == "PASS" for row in verification.values()) else "FAIL"
    result = {
        "schema": "royal-capital.fortification.package-checkpoint/1",
        "stage": stage,
        "status": verification["status"],
        "full": {"name": full.name, "bytes": full.stat().st_size, "sha256": sha256_path(full), "entries": len(zipfile.ZipFile(full).infolist())},
        "patch": {"name": patch.name, "bytes": patch.stat().st_size, "sha256": sha256_path(patch), "entries": len(zipfile.ZipFile(patch).infolist())},
        "source": {"name": source.name, "bytes": source.stat().st_size, "sha256": sha256_path(source), "entries": len(zipfile.ZipFile(source).infolist())},
        "source_delta": source_meta,
        "cumulative_patch": patch_meta,
        "verification": verification,
    }
    (deliver / f"{prefix}_check.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise RuntimeError(f"package checkpoint failed: {json.dumps(result, indent=2)}")
    return result
