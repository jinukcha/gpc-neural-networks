#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rcf_cp2_build_core", HERE / "rcf_cp2_build.py")
if SPEC is None or SPEC.loader is None:
    raise SystemExit("cannot load CP2 core builder")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

FIRST_FAILURE = {
    "schema": "royal-capital.fortification.cp2-preserved-failure/1",
    "workflow_run": 36102144460,
    "job": 107966722309,
    "status": "PRESERVED_RECONSTRUCTION_FAILURE",
    "failure_type": "SOURCE_DIET_STORED_COPY_OMISSION",
    "message": "A1S source package intentionally excluded four CP1 STEP/BREP stored-copy bytes while the A1 closed-world registry retained them.",
    "missing_paths": [
        "child_designs/fortification/r0a_cp1/reports/A/shape.brep",
        "child_designs/fortification/r0a_cp1/reports/A/shape.step",
        "child_designs/fortification/r0a_cp1/reports/B/shape.brep",
        "child_designs/fortification/r0a_cp1/reports/B/shape.step",
    ],
    "repair": "Regenerate only the four deterministic stored copies through the accepted CP1 adapter and exact CP0 wheelhouse, then require the original A1 FILES.sha256 registry to pass without modification.",
    "source_rollback": False,
    "preexisting_checkpoint_deleted": False,
}

SECOND_FAILURE = {
    "schema": "royal-capital.fortification.cp2-preserved-failure/1",
    "workflow_run": 36102444045,
    "job": 107967639739,
    "status": "PRESERVED_RECONSTRUCTION_FAILURE",
    "failure_type": "SOURCE_REPLAY_CACHE_RESIDUE",
    "message": "The deterministic CP1 stored-copy replay succeeded, but Python import caches were created inside the reconstructed source tree and correctly rejected as extra files by the unchanged A1 registry.",
    "extra_paths": [
        "child_designs/fortification/r0a_cp1/src/rcf_fortification_cad/__pycache__/__init__.cpython-313.pyc",
        "child_designs/fortification/r0a_cp1/src/rcf_fortification_cad/__pycache__/canonical.cpython-313.pyc",
        "child_designs/fortification/r0a_cp1/src/rcf_fortification_cad/__pycache__/contract.cpython-313.pyc",
        "child_designs/fortification/r0a_cp1/src/rcf_fortification_cad/__pycache__/provider.cpython-313.pyc",
    ],
    "repair": "Remove only generated __pycache__/pyc/pyo residue after replay and before immutable registry verification; preserve regenerated STEP/BREP bytes and all failure logs.",
    "source_rollback": False,
    "preexisting_checkpoint_deleted": False,
}


def clean_generated_python_cache(root: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(root.rglob("*.pyc")) + sorted(root.rglob("*.pyo")):
        if path.exists():
            removed.append(path.relative_to(root).as_posix())
            path.unlink()
    caches = sorted((p for p in root.rglob("__pycache__") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
    for cache in caches:
        if cache.exists():
            removed.append(cache.relative_to(root).as_posix() + "/")
            shutil.rmtree(cache)
    return removed


def restore_a1_stored_copies(tree_root: Path, work: Path) -> dict[str, object]:
    cp0 = tree_root / "child_designs/fortification/r0a_cp0"
    cp1 = tree_root / "child_designs/fortification/r0a_cp1"
    runtime = work / "reconstruct_cp1_runtime"
    log = work / "reconstruct_cp1_runtime.log"
    core.materialize_runtime(Path(sys.executable), runtime, cp0, log)
    replay_work = work / "reconstruct_cp1_work"
    replay_reports = work / "reconstruct_cp1_reports"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cp1 / "src")
    command = [
        str(runtime / "bin/python"),
        str(cp1 / "tools/run_cp1_qualification.py"),
        "--cp1-root", str(cp1),
        "--cp0-root", str(cp0),
        "--work-root", str(replay_work),
        "--reports-root", str(replay_reports),
    ]
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (work / "reconstruct_cp1_qualification.log").write_bytes(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"CP1 stored-copy replay failed with exit {result.returncode}")
    restored = []
    for label in ("A", "B"):
        for name in ("shape.step", "shape.brep"):
            source = replay_reports / label / name
            target = cp1 / "reports" / label / name
            if target.exists():
                raise RuntimeError(f"unexpected preexisting stored copy: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            restored.append(target.relative_to(tree_root).as_posix())
    removed_cache = clean_generated_python_cache(cp1)
    return {
        "schema": "royal-capital.fortification.a1-stored-copy-reconstruction/1",
        "status": "PASS",
        "method": "EXACT_CP1_ADAPTER_REPLAY",
        "restored": restored,
        "removed_generated_cache": removed_cache,
        "original_registry_preserved": True,
    }


def reconstruct_fixed(inputs: Path, wheel_artifact: Path, work: Path):
    for name, expected in (("RC_K0.zip", core.RCK0_SHA), ("RCF_D0_patch.zip", core.D0_PATCH_SHA), ("A1S.zip", core.A1S_SHA)):
        actual = core.sha_file(inputs / name)
        if actual != expected:
            raise RuntimeError(f"{name} SHA mismatch {actual}")
    base_extract = work / "base_extract"
    patch_extract = work / "d0_patch_extract"
    a1s_extract = work / "a1s_extract"
    core.extract_safe(inputs / "RC_K0.zip", base_extract)
    core.extract_safe(inputs / "RCF_D0_patch.zip", patch_extract)
    core.extract_safe(inputs / "A1S.zip", a1s_extract)
    baseline_parent = work / "baseline_parent"
    tree_parent = work / "tree_parent"
    baseline_parent.mkdir(parents=True)
    tree_parent.mkdir(parents=True)
    core.copy_overlay(base_extract, baseline_parent)
    core.copy_overlay(base_extract, tree_parent)
    w_candidates = [p for p in patch_extract.rglob("W") if p.is_dir()]
    if len(w_candidates) != 1:
        raise RuntimeError(f"expected one W directory, found {w_candidates}")
    core.copy_overlay(w_candidates[0], baseline_parent)
    core.copy_overlay(w_candidates[0], tree_parent)
    baseline_root = baseline_parent / "RC_K0"
    baseline_registry = core.verify_registry(baseline_root, 165)
    a1_root = a1s_extract / "RC_K0"
    if not a1_root.is_dir():
        raise RuntimeError("A1S RC_K0 root missing")
    tree_root = tree_parent / "RC_K0"
    core.copy_overlay(a1_root, tree_root)
    wh_source = wheel_artifact / "wheelhouse"
    wh_target = tree_root / "child_designs/fortification/r0a_cp0/runtime/wheelhouse"
    if not wh_source.is_dir() or len(list(wh_source.glob("*.whl"))) != 58:
        raise RuntimeError("exact 58-wheel artifact missing")
    wh_target.mkdir(parents=True, exist_ok=True)
    core.copy_overlay(wh_source, wh_target)
    repair = restore_a1_stored_copies(tree_root, work)
    a1_registry = core.verify_registry(tree_root, 427)
    return baseline_parent, tree_parent, {"baseline": baseline_registry, "a1": a1_registry, "a1_source_diet_repair": repair}


_original_cp2_files = core.cp2_files

def cp2_files_with_history():
    files = _original_cp2_files()
    files["reports/history/reconstruction_attempt_01_failure.json"] = json.dumps(FIRST_FAILURE, indent=2, sort_keys=True) + "\n"
    files["reports/history/reconstruction_attempt_02_failure.json"] = json.dumps(SECOND_FAILURE, indent=2, sort_keys=True) + "\n"
    files["reports/history/README.md"] = (
        "# Preserved CP2 recovery history\n\n"
        "Attempt 1 stopped before CP2 implementation because the compact A1 source package omitted four deterministic CP1 STEP/BREP evidence files still listed in the immutable A1 registry. Attempt 2 regenerated those files successfully but was stopped because Python import cache residue appeared in the source tree. No source, checkpoint or unique modification was deleted or rolled back. The final resume regenerates only the stored copies, removes only generated cache residue, and requires the unchanged A1 registry to pass before CP2 implementation begins.\n"
    )
    return files

core.reconstruct = reconstruct_fixed
core.cp2_files = cp2_files_with_history
raise SystemExit(core.main())
