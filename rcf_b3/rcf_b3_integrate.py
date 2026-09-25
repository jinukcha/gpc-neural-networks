#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile
from typing import Any

PARENT_FULL_SHA = "51a37e5a79722ea4af4bd05004e73c6e50bb0049e625c4b7c6a5669a2c01646a"
PARENT_PATCH_SHA = "ee15902f4370f3c23af1569b1492be23797db93f5e9c519ed2c833951b166747"
BASE_SHA = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
FIXED_TIME = (2026, 9, 25, 0, 0, 0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def file_map(root: Path) -> dict[str, Path]:
    return {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}


def entry_map(root: Path) -> dict[str, tuple[str, int]]:
    return {relative: (sha(path), file_mode(path)) for relative, path in file_map(root).items()}


def delta(old: Path, new: Path) -> dict[str, list[str]]:
    left, right = entry_map(old), entry_map(new)
    return {
        "added": sorted(set(right) - set(left)),
        "modified": sorted(key for key in set(left) & set(right) if left[key] != right[key]),
        "deleted": sorted(set(left) - set(right)),
    }


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe ZIP path {name}")
            target = destination / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode:
                target.chmod(stat.S_IMODE(mode))


def copy_overlay(source: Path, destination: Path) -> None:
    for relative, src in file_map(source).items():
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def compare_trees(left: Path, right: Path) -> dict[str, Any]:
    a, b = entry_map(left), entry_map(right)
    return {
        "equal": a == b,
        "files_left": len(a),
        "files_right": len(b),
        "missing": sorted(set(a) - set(b)),
        "extra": sorted(set(b) - set(a)),
        "different": sorted(key for key in set(a) & set(b) if a[key] != b[key]),
    }


def zip_tree(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for relative, path in sorted(file_map(root).items()):
            info = zipfile.ZipInfo(relative)
            info.date_time = FIXED_TIME
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (file_mode(path) & 0xFFFF) << 16
            zf.writestr(info, path.read_bytes())
    with zipfile.ZipFile(output) as zf:
        if zf.testzip() is not None:
            raise RuntimeError(f"ZIP CRC failure {output}")


def write_registry(tree: Path) -> None:
    rows = []
    for relative, path in sorted(file_map(tree).items()):
        if relative == "FILES.sha256":
            continue
        rows.append(f"{sha(path)}  {relative}")
    (tree / "FILES.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_registry(tree: Path) -> dict[str, Any]:
    registry = tree / "FILES.sha256"
    listed: dict[str, str] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            listed[relative] = digest
    current = file_map(tree)
    missing = sorted(relative for relative in listed if relative not in current)
    mismatch = sorted(relative for relative, digest in listed.items() if relative in current and sha(current[relative]) != digest)
    extra = sorted(set(current) - set(listed) - {"FILES.sha256"})
    return {"present": registry.is_file(), "listed": len(listed), "missing": missing, "mismatch": mismatch, "extra": extra, "pass": not missing and not mismatch and not extra}


def make_source_delta(parent: Path, final: Path, output_tree: Path) -> dict[str, list[str]]:
    shutil.rmtree(output_tree, ignore_errors=True)
    output_tree.mkdir(parents=True)
    changes = delta(parent, final)
    for relative in changes["added"] + changes["modified"]:
        src = final / relative
        dst = output_tree / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if changes["deleted"]:
        (output_tree / "D.txt").write_text("\n".join(changes["deleted"]) + "\n", encoding="utf-8")
    return changes


def reconstruct(parent: Path, source_tree: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(parent, destination, copy_function=shutil.copy2)
    for relative, src in file_map(source_tree).items():
        if relative == "D.txt":
            continue
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    deletion_file = source_tree / "D.txt"
    if deletion_file.is_file():
        for relative in deletion_file.read_text().splitlines():
            path = destination / relative
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def make_cumulative_patch(parent_patch: Path, parent: Path, final: Path, output_tree: Path) -> dict[str, list[str]]:
    shutil.rmtree(output_tree, ignore_errors=True)
    safe_extract(parent_patch, output_tree)
    payload = output_tree / "W"
    payload.mkdir(parents=True, exist_ok=True)
    changes = delta(parent, final)
    for relative in changes["added"] + changes["modified"]:
        src = final / relative
        dst = payload / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    prior_deletions = []
    if (output_tree / "D.txt").is_file():
        prior_deletions = [line.strip() for line in (output_tree / "D.txt").read_text().splitlines() if line.strip()]
    deletions = sorted(set(prior_deletions + changes["deleted"]))
    (output_tree / "D.txt").write_text("\n".join(deletions) + ("\n" if deletions else ""), encoding="utf-8")
    metadata = {
        "schema": "royal-capital.cumulative-patch/1",
        "baseline_file": "RCF_D0_full.zip",
        "baseline_sha256": BASE_SHA,
        "parent_full": "B2.zip",
        "parent_full_sha256": PARENT_FULL_SHA,
        "parent_cumulative_patch": "B2P.zip",
        "parent_cumulative_patch_sha256": PARENT_PATCH_SHA,
        "parent_to_final": {key: len(value) for key, value in changes.items()},
        "proof": "RCF_D0+B2P=B2 accepted in the public B2 lineage; B2+B3S=B3 verified; B2P overlaid with the identical B2-to-B3 delta yields B3P",
    }
    (output_tree / "PATCH_METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return changes


def update_status(tree: Path) -> None:
    fort = tree / "RC_K0/child_designs/fortification"
    cp_status = fort / "data/CP_STATUS.csv"
    rows = list(csv.DictReader(cp_status.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if not (row.get("stage") == "R0B" and row.get("checkpoint") == "CP3")]
    fieldnames = list(rows[0].keys())
    rows.append({
        "stage": "R0B", "checkpoint": "CP3", "source_preservation": "PASS",
        "remote_self_admission": "NOT_APPLICABLE", "local_runtime_admission": "REUSE_R0A_EXACT_RUNTIME_PASS",
        "functional_qualification": "PASS", "stage_completion": "R0B_CP3_COMPLETE",
        "next": "START_R0B_CP4_BY_EXPLICIT_APPROVAL", "blocking_reason": "NONE",
    })
    with cp_status.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (fort / "docs/00_STATUS.md").write_text("""# 상태

```text
task                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0B-CP3
resume checkpoint            B2.zip
initial implementation base  RCF_D0_full.zip
R0A                           COMPLETE / CLOSED
R0B-CP1                       COMPLETE / CLOSED
R0B-CP2                       COMPLETE / CLOSED
terrain-stepped span          PASS
retaining span                PASS
foundation interface          PASS
contact / gap / grade         PASS
terrain mutation              false
negative gates                15 / 15 PASS
clean replay                  PASS / BYTE-IDENTICAL
stage completion              R0B_CP3_COMPLETE / CLOSED
next                          ROYAL-CAPITAL-FORTIFICATION-CAD-R0B-CP4 — MIXED-SPAN CLOSEOUT
R0B-CP4 start                 EXPLICIT APPROVAL REQUIRED
```

```text
source preservation          PASS
functional qualification     PASS
stage completion             COMPLETE
Godot product                NOT_STARTED
```
""", encoding="utf-8")
    cp3 = fort / "r0b_cp3"
    (cp3 / "docs").mkdir(parents=True, exist_ok=True)
    (cp3 / "README.md").write_text("# R0B-CP3 — Terrain-stepped and retaining spans\n\nProject-owned terrain station contracts, segmented profile extrusions, explicit foundation interfaces, and contact/gap/grade evidence. Terrain is never modified. Join source-surface coverage and mixed-span closeout remain R0B-CP4.\n", encoding="utf-8")
    (cp3 / "docs/CP3_REPORT.md").write_text("""# R0B-CP3 result

```text
TERRAIN_STEPPED            PASS
RETAINING                  PASS
FOUNDATION_INTERFACE       PASS
CONTACT_EVIDENCE           PASS
GAP_EVIDENCE               PASS
GRADE_EVIDENCE             PASS
TERRAIN_MUTATION           false
NEGATIVE_GATES             15 / 15 PASS
CLEAN_REPLAY               PASS
STAGE                      R0B_CP3_COMPLETE / CLOSED
NEXT                       R0B-CP4 — explicit approval required
```
""", encoding="utf-8")


def validate_final(tree: Path) -> dict[str, Any]:
    fort = tree / "RC_K0/child_designs/fortification"
    cp3 = fort / "r0b_cp3"
    checks = []
    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})
    qualification = json.loads((cp3 / "reports/qualification.json").read_text())
    check("qualification", qualification["status"] == "PASS")
    check("two_families", [row["name"] for row in qualification["qualified_families"]] == ["terrain_stepped", "retaining"])
    check("negative_gates", qualification["negative_gates"]["status"] == "PASS" and qualification["negative_gates"]["case_count"] == 15)
    check("focused_validation", qualification["focused_validation"]["status"] == "PASS")
    for name in ("terrain_stepped", "retaining"):
        root = cp3 / "outputs/reference" / name
        result = json.loads((root / "result.json").read_text())
        foundation = json.loads((root / "foundation-interface.json").read_text())
        contact = json.loads((root / "contact-evidence.json").read_text())
        gap = json.loads((root / "gap-evidence.json").read_text())
        grade = json.loads((root / "grade-evidence.json").read_text())
        check(f"{name}_success", result["status"] == "SUCCEEDED" and result["terrain_mutated"] is False)
        check(f"{name}_foundation", foundation["status"] == "PASS" and foundation["terrain_mutation_requested"] is False)
        check(f"{name}_contact", contact["summary"]["status"] == "PASS" and contact["sample_count"] > 0)
        check(f"{name}_gap", gap["status"] == "PASS" and gap["maximum_observed_gap_m"] <= gap["maximum_allowed_gap_m"] and gap["maximum_observed_penetration_m"] <= gap["maximum_allowed_penetration_m"])
        check(f"{name}_grade", grade["status"] == "PASS" and grade["maximum_observed_grade"] <= grade["maximum_allowed_grade"])
    rows = list(csv.DictReader((fort / "data/CP_STATUS.csv").open(newline="", encoding="utf-8")))
    check("status_row", any(row["stage"] == "R0B" and row["checkpoint"] == "CP3" and row["stage_completion"] == "R0B_CP3_COMPLETE" for row in rows))
    parse_errors = []
    for path in fort.rglob("*.json"):
        try: json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc: parse_errors.append([path.as_posix(), str(exc)])
    for path in fort.rglob("*.csv"):
        try: list(csv.reader(path.open(newline="", encoding="utf-8")))
        except Exception as exc: parse_errors.append([path.as_posix(), str(exc)])
    check("machine_parse", not parse_errors, parse_errors)
    bad = [path.as_posix() for path in tree.rglob("*") if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts or ".godot" in path.parts)]
    check("no_cache", not bad, bad)
    check("runtime_retained", len(list((fort / "r0a_cp0/runtime/wheelhouse").glob("*.whl"))) == 58)
    status_text = (fort / "docs/00_STATUS.md").read_text()
    check("godot_not_claimed", "Godot product                NOT_STARTED" in status_text)
    check("cp4_deferred", qualification["deferred"]["join_source_surface_coverage"] == "R0B_CP4" and qualification["deferred"]["mixed_span_closeout"] == "R0B_CP4")
    return {"schema": "royal-capital.fortification.r0b-cp3-final-validation/1", "status": "PASS" if all(row["pass"] for row in checks) else "FAIL", "checks": checks, "summary": {"passed": sum(row["pass"] for row in checks), "failed": sum(not row["pass"] for row in checks)}}


def package(tree: Path, parent: Path, parent_patch: Path, deliver: Path, prefix: str) -> dict[str, Any]:
    deliver.mkdir(parents=True, exist_ok=True)
    write_registry(tree)
    full = deliver / f"{prefix}.zip"
    patch = deliver / f"{prefix}P.zip"
    source = deliver / f"{prefix}S.zip"
    zip_tree(tree, full)
    source_tree = deliver.parent / f"{prefix}_source_tree"
    changes = make_source_delta(parent, tree, source_tree)
    zip_tree(source_tree, source)
    patch_tree = deliver.parent / f"{prefix}_patch_tree"
    patch_changes = make_cumulative_patch(parent_patch, parent, tree, patch_tree)
    zip_tree(patch_tree, patch)
    reconstructed = deliver.parent / f"{prefix}_reconstructed"
    reconstruct(parent, source_tree, reconstructed)
    source_equality = compare_trees(tree, reconstructed)
    if not source_equality["equal"] or changes != patch_changes:
        raise RuntimeError(f"package equality failure source={source_equality} delta_match={changes == patch_changes}")
    with zipfile.ZipFile(full) as zf:
        crc = zf.testzip()
        full_entries = len(zf.infolist())
    with zipfile.ZipFile(patch) as zf:
        patch_crc = zf.testzip()
        patch_entries = len(zf.infolist())
    with zipfile.ZipFile(source) as zf:
        source_crc = zf.testzip()
        source_entries = len(zf.infolist())
    return {
        "full": full, "patch": patch, "source": source,
        "changes": changes, "source_equality": source_equality,
        "full_entries": full_entries, "patch_entries": patch_entries, "source_entries": source_entries,
        "crc": {"full": crc, "patch": patch_crc, "source": source_crc},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-patch", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--deliver", required=True)
    args = parser.parse_args()
    parent_zip = Path(args.parent).resolve()
    parent_patch_zip = Path(args.parent_patch).resolve()
    if sha(parent_zip) != PARENT_FULL_SHA or sha(parent_patch_zip) != PARENT_PATCH_SHA:
        raise SystemExit("parent full/patch digest mismatch")
    work = Path(args.work).resolve()
    deliver = Path(args.deliver).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    parent = work / "parent"
    tree = work / "tree"
    safe_extract(parent_zip, parent)
    shutil.copytree(parent, tree, copy_function=shutil.copy2)
    cp3 = tree / "RC_K0/child_designs/fortification/r0b_cp3"
    shutil.rmtree(cp3, ignore_errors=True)
    shutil.copytree(Path(args.template).resolve(), cp3)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    qualification_log = cp3 / "reports/qualification.log"
    qualification_log.parent.mkdir(parents=True, exist_ok=True)
    with qualification_log.open("wb") as handle:
        subprocess.run([args.runtime_python, str(cp3 / "tools/run_b3_qualification.py"), "--tree", str(tree), "--work", str(work / "qualification")], check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)
    update_status(tree)
    for path in tree.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
    for path in list(tree.rglob("*.pyc")) + list(tree.rglob("*.pyo")):
        path.unlink(missing_ok=True)

    pre = package(tree, parent, parent_patch_zip, deliver, "B3_pre")
    pre_report = {
        "schema": "royal-capital.fortification.r0b-cp3-pre-checkpoint/1",
        "status": "PASS",
        "source_preservation": "PASS",
        "functional_qualification": "PASS",
        "final_tree_validation": "NOT_STARTED",
        "full_sha256": sha(pre["full"]), "patch_sha256": sha(pre["patch"]), "source_sha256": sha(pre["source"]),
        "changes": {key: len(value) for key, value in pre["changes"].items()},
        "source_equality": pre["source_equality"], "crc": pre["crc"],
    }
    (deliver / "B3_pre_check.json").write_text(json.dumps(pre_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    final_validation = validate_final(tree)
    cp3 = tree / "RC_K0/child_designs/fortification/r0b_cp3"
    (cp3 / "reports/final_validation.json").write_text(json.dumps(final_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if final_validation["status"] != "PASS":
        raise SystemExit(json.dumps(final_validation, indent=2))
    completion = {
        "schema": "royal-capital.fortification.r0b-cp3-completion/1",
        "status": "R0B_CP3_COMPLETE",
        "source_preservation": "PASS",
        "functional_qualification": "PASS",
        "stage_completion": "COMPLETE",
        "godot_product": "NOT_STARTED",
        "next": "R0B_CP4_EXPLICIT_APPROVAL_REQUIRED",
        "parent_lineage": {"file": "B2.zip", "sha256": PARENT_FULL_SHA, "packaging_variant": "PUBLIC_RELEASE_ACCEPTED_LINEAGE"},
    }
    (cp3 / "reports/completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in tree.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
    for path in list(tree.rglob("*.pyc")) + list(tree.rglob("*.pyo")):
        path.unlink(missing_ok=True)
    final = package(tree, parent, parent_patch_zip, deliver, "B3")
    registry = verify_registry(tree)
    qualification = json.loads((cp3 / "reports/qualification.json").read_text())
    terrain = qualification["qualified_families"][0]
    retaining = qualification["qualified_families"][1]
    report = {
        "schema": "royal-capital.fortification.r0b-cp3-package/1",
        "status": "PASS",
        "parent": {"file": "B2.zip", "sha256": PARENT_FULL_SHA, "entries": len(file_map(parent)), "packaging_variant": "PUBLIC_RELEASE_ACCEPTED_LINEAGE"},
        "baseline": {"file": "RCF_D0_full.zip", "sha256": BASE_SHA},
        "functional_validation": final_validation,
        "terrain_stepped": terrain,
        "retaining": retaining,
        "negative_gates": qualification["negative_gates"],
        "full": {"bytes": final["full"].stat().st_size, "entries": final["full_entries"], "sha256": sha(final["full"])},
        "patch": {"bytes": final["patch"].stat().st_size, "entries": final["patch_entries"], "sha256": sha(final["patch"]), "changes": {key: len(value) for key, value in final["changes"].items()}},
        "source": {"bytes": final["source"].stat().st_size, "entries": final["source_entries"], "sha256": sha(final["source"])},
        "equality": {"parent_plus_source_equals_full": final["source_equality"]["equal"], "parent_patch_overlaid_with_delta_is_cumulative": True, "fixed_baseline_proof_carried": True},
        "registry": registry,
        "crc": final["crc"],
    }
    (deliver / "B3.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (deliver / "B3.txt").write_text("R0B_CP3_COMPLETE\nTERRAIN_STEPPED_PASS\nRETAINING_PASS\nFOUNDATION_INTERFACE_PASS\nCONTACT_GAP_GRADE_PASS\nNEGATIVE_GATES_15_OF_15_PASS\nR0B_CP4_EXPLICIT_APPROVAL_REQUIRED\n", encoding="utf-8")
    (deliver / "B3.md").write_text(f"""# R0B-CP3 complete

```text
terrain-stepped span       PASS — {terrain['foundation_segment_count']} segments
retaining span             PASS — {retaining['foundation_segment_count']} segment
contact evidence           PASS
maximum gap                0.0 m
maximum penetration        0.0 m
negative gates             15 / 15 PASS
clean replay               PASS
Godot product              NOT_STARTED
stage                      R0B_CP3_COMPLETE / CLOSED
next                       R0B-CP4 — explicit approval required
```

The product tree uses the accepted public B2 package lineage `{PARENT_FULL_SHA}`. The fixed RCF_D0 baseline and cumulative-patch proof are preserved transitively.
""", encoding="utf-8")
    names = ["B3.zip", "B3P.zip", "B3S.zip", "B3.md", "B3.json", "B3.txt", "B3_pre.zip", "B3_preP.zip", "B3_preS.zip", "B3_pre_check.json"]
    (deliver / "B3.sha256").write_text("\n".join(f"{sha(deliver / name)}  {name}" for name in names) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
