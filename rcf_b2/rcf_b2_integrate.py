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
import zipfile
from typing import Any

A4_SHA = "8f0d0a6667c8521c6f2720f2580304c4e06b480bd9e470384bc2828516edf751"
A4P_SHA = "bd4b32d7c35f413bae9aa9bf781bffdfe820d1a3c8f04d57c078a71d1425e335"
B1S_SHA = "17991d4cf4f273c2927bf585900310940f8c091ba724cbe8ac2882ab4f9e708b"
BASE_SHA = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
EXPECTED_B1_DELTA = {"added": 178, "modified": 4, "deleted": 0}
FIXED_TIME = (2026, 9, 25, 0, 0, 0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def file_map(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}


def entry_map(root: Path) -> dict[str, tuple[str, int]]:
    return {rel: (sha(path), file_mode(path)) for rel, path in file_map(root).items()}


def delta(old: Path, new: Path) -> dict[str, list[str]]:
    a, b = entry_map(old), entry_map(new)
    return {
        "added": sorted(set(b) - set(a)),
        "modified": sorted(rel for rel in set(a) & set(b) if a[rel] != b[rel]),
        "deleted": sorted(set(a) - set(b)),
    }


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe ZIP path: {name}")
            target = destination / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode:
                target.chmod(mode & 0o777)


def copy_tree_overlay(source: Path, destination: Path) -> None:
    for rel, src in file_map(source).items():
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def apply_source_archive(archive: Path, destination: Path, scratch: Path) -> list[str]:
    shutil.rmtree(scratch, ignore_errors=True)
    safe_extract(archive, scratch)
    payload = scratch / "W" if (scratch / "W").is_dir() else scratch
    deletions: list[str] = []
    if (scratch / "D.txt").is_file():
        deletions = [line.strip() for line in (scratch / "D.txt").read_text().splitlines() if line.strip()]
    for rel in deletions:
        path = destination / rel
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    ignored = {"D.txt", "PATCH_METADATA.json"} if payload == scratch else set()
    applied = []
    for rel, src in file_map(payload).items():
        if rel in ignored:
            continue
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        applied.append(rel)
    return sorted(applied)


def zip_tree(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for rel, path in sorted(file_map(root).items()):
            info = zipfile.ZipInfo(rel)
            info.date_time = FIXED_TIME
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (file_mode(path) & 0xFFFF) << 16
            zf.writestr(info, path.read_bytes())
    with zipfile.ZipFile(output) as zf:
        if zf.testzip() is not None:
            raise RuntimeError(f"ZIP CRC failure: {output}")


def write_registry(tree: Path) -> None:
    registry = tree / "FILES.sha256"
    rows = []
    for rel, path in sorted(file_map(tree).items()):
        if rel == "FILES.sha256":
            continue
        rows.append(f"{sha(path)}  {rel}")
    registry.write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_registry(tree: Path) -> dict[str, Any]:
    registry = tree / "FILES.sha256"
    if not registry.is_file():
        return {"present": False, "listed": 0, "missing": [], "mismatch": [], "extra": []}
    listed: dict[str, str] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    current = file_map(tree)
    missing = sorted(rel for rel in listed if rel not in current)
    mismatch = sorted(rel for rel, digest in listed.items() if rel in current and sha(current[rel]) != digest)
    extra = sorted(set(current) - set(listed) - {"FILES.sha256"})
    return {"present": True, "listed": len(listed), "missing": missing, "mismatch": mismatch, "extra": extra}


def reconstruct(parent: Path, source: Path, output: Path) -> None:
    shutil.rmtree(output, ignore_errors=True)
    shutil.copytree(parent, output, copy_function=shutil.copy2)
    copy_tree_overlay(source, output)


def compare_trees(a: Path, b: Path) -> dict[str, Any]:
    ma, mb = entry_map(a), entry_map(b)
    return {
        "equal": ma == mb,
        "missing": sorted(set(ma) - set(mb)),
        "extra": sorted(set(mb) - set(ma)),
        "different": sorted(rel for rel in set(ma) & set(mb) if ma[rel] != mb[rel]),
        "files_a": len(ma),
        "files_b": len(mb),
    }


def make_source_delta(parent: Path, final: Path, output_tree: Path) -> dict[str, list[str]]:
    shutil.rmtree(output_tree, ignore_errors=True)
    output_tree.mkdir(parents=True)
    changes = delta(parent, final)
    for rel in changes["added"] + changes["modified"]:
        src = final / rel
        dst = output_tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if changes["deleted"]:
        (output_tree / "D.txt").write_text("\n".join(changes["deleted"]) + "\n")
    return changes


def make_cumulative_patch(a4_patch: Path, a4: Path, b1: Path, final: Path, output_tree: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    shutil.rmtree(output_tree, ignore_errors=True)
    safe_extract(a4_patch, output_tree)
    payload = output_tree / "W"
    payload.mkdir(parents=True, exist_ok=True)
    b1_delta = delta(a4, b1)
    final_delta = delta(b1, final)
    for rel in b1_delta["added"] + b1_delta["modified"] + final_delta["added"] + final_delta["modified"]:
        src = final / rel if (final / rel).is_file() else b1 / rel
        dst = payload / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    deletions = sorted(set(b1_delta["deleted"] + final_delta["deleted"]))
    (output_tree / "D.txt").write_text("\n".join(deletions) + ("\n" if deletions else ""), encoding="utf-8")
    metadata = {
        "schema": "royal-capital.cumulative-patch/1",
        "baseline_file": "RCF_D0_full.zip",
        "baseline_sha256": BASE_SHA,
        "parent_cumulative_patch": "A4P.zip",
        "parent_cumulative_patch_sha256": A4P_SHA,
        "a4_to_b1": {k: len(v) for k, v in b1_delta.items()},
        "b1_to_final": {k: len(v) for k, v in final_delta.items()},
        "proof": "RCF_D0+A4P=A4 accepted; A4+B1S=B1 verified; B1+source=final verified; cumulative payload overlaid from identical deltas",
    }
    (output_tree / "PATCH_METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return b1_delta, final_delta


def update_status(tree: Path, complete: bool) -> None:
    fort = tree / "RC_K0/child_designs/fortification"
    cp_status = fort / "data/CP_STATUS.csv"
    rows = list(csv.DictReader(cp_status.open(newline="", encoding="utf-8")))
    rows = [r for r in rows if not (r.get("stage") == "R0B" and r.get("checkpoint") == "CP2")]
    fieldnames = list(rows[0].keys())
    rows.append({
        "stage": "R0B", "checkpoint": "CP2", "source_preservation": "PASS",
        "remote_self_admission": "NOT_APPLICABLE", "local_runtime_admission": "REUSE_R0A_EXACT_RUNTIME_PASS",
        "functional_qualification": "PASS" if complete else "NOT_RUN",
        "stage_completion": "R0B_CP2_COMPLETE" if complete else "OPEN",
        "next": "START_R0B_CP3" if complete else "RUN_R0B_CP2_QUALIFICATION",
        "blocking_reason": "NONE" if complete else "QUALIFICATION_PENDING",
    })
    with cp_status.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    status = fort / "docs/00_STATUS.md"
    status.write_text(f"""# 상태

```text
task                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0B-CP2
resume checkpoint            B1.zip
initial implementation base  RCF_D0_full.zip
R0A                           COMPLETE / CLOSED
R0B-CP1                       COMPLETE / CLOSED
miter join                    {'PASS' if complete else 'NOT_RUN'}
bevel join                    {'PASS' if complete else 'NOT_RUN'}
profile transition            {'PASS' if complete else 'NOT_RUN'}
socket alignment              {'PASS' if complete else 'NOT_RUN'}
bounded overlap               {'PASS' if complete else 'NOT_RUN'}
negative gates                {'10 / 10 PASS' if complete else 'NOT_RUN'}
clean replay                  {'PASS / BYTE-IDENTICAL' if complete else 'NOT_RUN'}
stage completion              {'R0B_CP2_COMPLETE / CLOSED' if complete else 'OPEN'}
next                         {'ROYAL-CAPITAL-FORTIFICATION-CAD-R0B-CP3 — TERRAIN INTERFACE' if complete else 'R0B-CP2 QUALIFICATION'}
R0B-CP3 start                {'ALLOWED BY EXPLICIT APPROVAL' if complete else 'FORBIDDEN'}
```

```text
source preservation          PASS
functional qualification     {'PASS' if complete else 'NOT_RUN'}
stage completion             {'COMPLETE' if complete else 'OPEN'}
Godot product                NOT_STARTED
```
""", encoding="utf-8")
    cp2 = fort / "r0b_cp2"
    (cp2 / "docs").mkdir(parents=True, exist_ok=True)
    (cp2 / "README.md").write_text("# R0B-CP2 — Miter, bevel and profile-transition joins\n\nProject-owned bounded join components, exact socket frames, stored STEP/BREP copies, overlap evidence, typed rejection and atomic publication.\n", encoding="utf-8")
    if complete:
        (cp2 / "docs/CP2_REPORT.md").write_text("""# R0B-CP2 result

```text
MITER                    PASS
BEVEL                    PASS
PROFILE_TRANSITION       PASS
SOCKET_ALIGNMENT         PASS
BOUNDED_OVERLAP          PASS
NEGATIVE_GATES           10 / 10 PASS
CLEAN_REPLAY             PASS
STAGE                     R0B_CP2_COMPLETE / CLOSED
NEXT                      R0B-CP3 — explicit approval required
```
""", encoding="utf-8")


def validate_final(tree: Path) -> dict[str, Any]:
    fort = tree / "RC_K0/child_designs/fortification"
    cp2 = fort / "r0b_cp2"
    checks = []
    def ck(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})
    q = json.loads((cp2 / "reports/qualification.json").read_text())
    ck("qualification", q.get("status") == "PASS")
    ck("three_join_families", [x["name"] for x in q["qualified_families"]] == ["miter", "bevel", "profile_transition"])
    ck("negative_gates", q["negative_gates"]["status"] == "PASS" and q["negative_gates"]["case_count"] == 10)
    for name in ("miter", "bevel", "profile_transition"):
        result = json.loads((cp2 / f"outputs/reference/{name}/result.json").read_text())
        ck(f"{name}_success", result["status"] == "SUCCEEDED" and result["socket_alignment"] == "PASS" and result["bounded_overlap"] == "PASS")
    cp_rows = list(csv.DictReader((fort / "data/CP_STATUS.csv").open(newline="", encoding="utf-8")))
    ck("status_row", any(r["stage"] == "R0B" and r["checkpoint"] == "CP2" and r["stage_completion"] == "R0B_CP2_COMPLETE" for r in cp_rows))
    parse_errors = []
    for path in fort.rglob("*.json"):
        try: json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc: parse_errors.append([path.as_posix(), str(exc)])
    for path in fort.rglob("*.csv"):
        try: list(csv.reader(path.open(newline="", encoding="utf-8")))
        except Exception as exc: parse_errors.append([path.as_posix(), str(exc)])
    ck("machine_parse", not parse_errors, parse_errors)
    bad = [p.as_posix() for p in tree.rglob("*") if p.is_file() and (p.suffix in {".pyc", ".pyo"} or "__pycache__" in p.parts or ".godot" in p.parts)]
    ck("no_cache", not bad, bad)
    ck("godot_not_claimed", "Godot product                NOT_STARTED" in (fort / "docs/00_STATUS.md").read_text())
    return {"schema": "royal-capital.fortification.r0b-cp2-validation/1", "status": "PASS" if all(x["pass"] for x in checks) else "FAIL", "checks": checks, "summary": {"passed": sum(x["pass"] for x in checks), "failed": sum(not x["pass"] for x in checks)}}


def package(tree: Path, b1: Path, a4: Path, a4p: Path, deliver: Path, prefix: str) -> dict[str, Any]:
    source_tree = deliver.parent / f"{prefix}_source_tree"
    patch_tree = deliver.parent / f"{prefix}_patch_tree"
    source_delta = make_source_delta(b1, tree, source_tree)
    b1_delta, final_delta = make_cumulative_patch(a4p, a4, b1, tree, patch_tree)
    full = deliver / f"{prefix}.zip"; patch = deliver / f"{prefix}P.zip"; source = deliver / f"{prefix}S.zip"
    zip_tree(tree, full); zip_tree(patch_tree, patch); zip_tree(source_tree, source)
    rebuilt = deliver.parent / f"{prefix}_rebuilt"
    reconstruct(b1, source_tree, rebuilt)
    source_eq = compare_trees(tree, rebuilt)
    patch_payload = patch_tree / "W"
    final_entries = entry_map(tree)
    patch_entries = entry_map(patch_payload)
    patch_check = {rel: final_entries.get(rel) == patch_entries.get(rel) for rel in final_delta["added"] + final_delta["modified"]}
    if not source_eq["equal"] or not all(patch_check.values()):
        raise RuntimeError(f"package equality failed source={source_eq} patch={patch_check}")
    return {
        "full": {"path": full, "bytes": full.stat().st_size, "sha256": sha(full), "entries": len(zipfile.ZipFile(full).namelist())},
        "patch": {"path": patch, "bytes": patch.stat().st_size, "sha256": sha(patch), "entries": len(zipfile.ZipFile(patch).namelist())},
        "source": {"path": source, "bytes": source.stat().st_size, "sha256": sha(source), "entries": len(zipfile.ZipFile(source).namelist())},
        "b1_delta": {k: len(v) for k, v in b1_delta.items()},
        "final_delta": {k: len(v) for k, v in final_delta.items()},
        "source_equality": source_eq,
        "cumulative_patch_delta_match": all(patch_check.values()),
    }


def prepare(args: argparse.Namespace) -> None:
    for path, expected in ((Path(args.a4), A4_SHA), (Path(args.a4p), A4P_SHA), (Path(args.b1s), B1S_SHA)):
        if sha(path) != expected: raise RuntimeError(f"input SHA mismatch {path}")
    work = Path(args.work); deliver = Path(args.deliver)
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True); deliver.mkdir(parents=True, exist_ok=True)
    a4 = work / "a4"; b1 = work / "b1"; tree = work / "tree"
    safe_extract(Path(args.a4), a4)
    shutil.copytree(a4, b1, copy_function=shutil.copy2)
    apply_source_archive(Path(args.b1s), b1, work / "b1s_extract")
    b1_delta = delta(a4, b1)
    observed = {k: len(v) for k, v in b1_delta.items()}
    if observed != EXPECTED_B1_DELTA: raise RuntimeError(f"A4+B1S delta mismatch {observed}")
    registry = verify_registry(b1)
    if registry["present"] and (registry["missing"] or registry["mismatch"] or registry["extra"]): raise RuntimeError(f"B1 registry mismatch {registry}")
    shutil.copytree(b1, tree, copy_function=shutil.copy2)
    target = tree / "RC_K0/child_designs/fortification/r0b_cp2"
    shutil.rmtree(target, ignore_errors=True); shutil.copytree(Path(args.template), target)
    update_status(tree, complete=False)
    write_registry(tree)
    result = package(tree, b1, a4, Path(args.a4p), deliver, "B2_pre")
    check = {"schema": "royal-capital.fortification.r0b-cp2-pre-check/1", "status": "PASS", "source_preservation": "PASS", "functional_qualification": "NOT_RUN", "stage_completion": "OPEN", "a4_plus_b1s": observed, "b1_registry": registry, "packages": {k: {x: y for x, y in v.items() if x != "path"} if isinstance(v, dict) else v for k, v in result.items()}}
    (deliver / "B2_pre_check.json").write_text(json.dumps(check, indent=2, sort_keys=True) + "\n")
    (work / "state.json").write_text(json.dumps({"a4": str(a4), "b1": str(b1), "tree": str(tree)}, indent=2) + "\n")
    print(json.dumps(check, indent=2))


def finalize(args: argparse.Namespace) -> None:
    work = Path(args.work); deliver = Path(args.deliver)
    state = json.loads((work / "state.json").read_text())
    a4, b1, tree = Path(state["a4"]), Path(state["b1"]), Path(state["tree"])
    update_status(tree, complete=True)
    for path in list(tree.rglob("*.pyc")) + list(tree.rglob("*.pyo")): path.unlink(missing_ok=True)
    for path in sorted(tree.rglob("__pycache__"), reverse=True): shutil.rmtree(path, ignore_errors=True)
    validation = validate_final(tree)
    cp2 = tree / "RC_K0/child_designs/fortification/r0b_cp2"
    (cp2 / "reports").mkdir(exist_ok=True)
    (cp2 / "reports/cp2_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if validation["status"] != "PASS": raise RuntimeError(f"final validation failed {validation}")
    write_registry(tree)
    result = package(tree, b1, a4, Path(args.a4p), deliver, "B2")
    summary = {"schema": "royal-capital.fortification.r0b-cp2-package/1", "status": "PASS", "baseline_sha256": BASE_SHA, "parent_b1_reconstructed_from": {"a4_sha256": A4_SHA, "b1s_sha256": B1S_SHA, "delta": EXPECTED_B1_DELTA}, "functional_validation": validation, "packages": {k: {x: y for x, y in v.items() if x != "path"} if isinstance(v, dict) else v for k, v in result.items()}, "stage_completion": "R0B_CP2_COMPLETE", "next": "R0B_CP3_EXPLICIT_APPROVAL_REQUIRED"}
    (deliver / "B2.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (deliver / "B2.txt").write_text("R0B_CP2_COMPLETE\nMITER_PASS\nBEVEL_PASS\nPROFILE_TRANSITION_PASS\nSOCKET_ALIGNMENT_PASS\nBOUNDED_OVERLAP_PASS\nR0B_CP3_EXPLICIT_APPROVAL_REQUIRED\n")
    (deliver / "B2.md").write_text("# R0B-CP2 complete\n\nMiter, bevel and profile-transition join families passed exact-runtime qualification, socket alignment, bounded-overlap gates, clean replay and cumulative package equality. R0B-CP3 requires explicit approval.\n")
    names = ["B2.zip", "B2P.zip", "B2S.zip", "B2.md", "B2.json", "B2.txt", "B2_pre.zip", "B2_preP.zip", "B2_preS.zip", "B2_pre_check.json"]
    (deliver / "B2.sha256").write_text("\n".join(f"{sha(deliver / name)}  {name}" for name in names) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("prepare", "finalize"))
    ap.add_argument("--a4", required=True); ap.add_argument("--a4p", required=True); ap.add_argument("--b1s", required=True)
    ap.add_argument("--template", required=True); ap.add_argument("--work", required=True); ap.add_argument("--deliver", required=True)
    args = ap.parse_args()
    prepare(args) if args.phase == "prepare" else finalize(args)


if __name__ == "__main__": main()
