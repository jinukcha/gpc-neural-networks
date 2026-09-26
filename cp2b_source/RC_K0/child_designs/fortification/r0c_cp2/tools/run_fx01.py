#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree_digest(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for path, digest in sorted(hashes.items()):
        h.update(path.encode("utf-8") + b"\0" + bytes.fromhex(digest))
    return "sha256:" + h.hexdigest()


def _update_matrix(path: Path, evidence_ref: str) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise SystemExit("fixture matrix header missing")
    matched = 0
    for row in rows:
        if row["fixture_id"] == "R0C_CP2_FX01":
            matched += 1
            if row["status"] != "FROZEN_PENDING_IMPLEMENTATION":
                raise SystemExit(f"unexpected pre-run FX01 status: {row['status']}")
            row["exact_source_binding"] = "BOUND_EXACT_ACCEPTED_IDS_DIGESTS_AND_SOCKETS"
            row["status"] = "PASS_CP2_B"
            row["authority_note"] = (
                "CP2-B accepted ROUND×TANGENT×STRAIGHT_SPAN representative fixture; "
                f"evidence={evidence_ref}"
            )
    if matched != 1:
        raise SystemExit(f"FX01 matrix row count mismatch: {matched}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    work = Path(args.work).resolve()
    fort = tree / "RC_K0/child_designs/fortification"
    cp1 = fort / "r0a_cp1"
    cp2 = fort / "r0c_cp2"
    sys.path[:0] = [str(cp1 / "src"), str(cp2 / "src")]

    from rcf_fortification_tower_join import Fx01TowerJoinProducer

    fixture_path = cp2 / "fixtures/R0C_CP2_FX01.fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    runs = work / "runs"
    failures = work / "failures"
    if runs.exists():
        shutil.rmtree(runs)
    if failures.exists():
        shutil.rmtree(failures)
    runs.mkdir(parents=True)
    failures.mkdir(parents=True)

    reference = cp2 / "outputs/fx01/reference"
    if reference.exists():
        raise SystemExit(f"accepted reference path must be fresh: {reference}")

    producer = Fx01TowerJoinProducer(tree)
    run_a = runs / "A"
    run_b = runs / "B"
    result_a = producer.execute(fixture, run_a, failures, failure_case="positive_A")
    result_b = producer.execute(fixture, run_b, failures, failure_case="positive_B")
    if result_a.get("status") != "SUCCEEDED" or result_b.get("status") != "SUCCEEDED":
        raise SystemExit(f"FX01 positive run failed: A={result_a} B={result_b}")

    hashes_a = _file_hashes(run_a)
    hashes_b = _file_hashes(run_b)
    clean = {
        "schema": "royal-capital.fortification.r0c-cp2.fx01-clean-replay/1",
        "fixture_id": "R0C_CP2_FX01",
        "status": "PASS" if hashes_a == hashes_b else "FAIL",
        "files_compared": len(hashes_a),
        "byte_identical": hashes_a == hashes_b,
        "missing_in_b": sorted(set(hashes_a) - set(hashes_b)),
        "extra_in_b": sorted(set(hashes_b) - set(hashes_a)),
        "different": sorted(
            path
            for path in set(hashes_a) & set(hashes_b)
            if hashes_a[path] != hashes_b[path]
        ),
        "run_a_tree_digest": _tree_digest(hashes_a),
        "run_b_tree_digest": _tree_digest(hashes_b),
    }
    if clean["status"] != "PASS":
        raise SystemExit(f"FX01 clean replay mismatch: {clean}")

    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_a, reference)
    reference_hashes = _file_hashes(reference)
    if reference_hashes != hashes_a:
        raise SystemExit("published reference does not match clean run A")

    expected = {
        "part_count": 3,
        "stored_copy_count": 6,
        "volume_m3": 45.92,
        "socket_position_error_m": 0.0,
        "tangent_error": 0.0,
        "up_axis_error": 0.0,
        "inside_frame_error": 0.0,
        "wall_walk_position_error_m": 0.0,
        "foundation_source_offset_m": 0.5,
        "foundation_transition_residual_m": 0.0,
        "unsupported_gap_m": 0.0,
        "outside_projection_m": 0.0,
        "inside_projection_m": 0.0,
        "accepted_sources_unchanged": True,
        "stored_copies_reopened": True,
        "partial_output_published": False,
    }
    for key, value in expected.items():
        if result_a.get(key) != value:
            raise SystemExit(f"FX01 evidence mismatch {key}: {result_a.get(key)!r} != {value!r}")

    evidence_rel = "RC_K0/child_designs/fortification/r0c_cp2/reports/CP2B_FX01_qualification.json"
    reports = cp2 / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    clean_path = reports / "CP2B_FX01_clean_replay.json"
    clean_path.write_text(_pretty(clean), encoding="utf-8")

    qualification = {
        "schema": "royal-capital.fortification.r0c-cp2.fx01-qualification/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "official_subtitle": "TOWER JOINS",
        "checkpoint": "CP2-B_R0C_CP2_FX01",
        "fixture_id": "R0C_CP2_FX01",
        "status": "PASS",
        "source_preservation": "PASS",
        "functional_qualification": "PASS_FX01",
        "stage_completion": "CP2_B_COMPLETE_CLOSED",
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "implemented": {
            "tower_family": "ROUND",
            "join_kind": "TANGENT",
            "span_family": "STRAIGHT_SPAN",
            "realization": "BOUNDED_TRANSITION_PIECE",
            "source_geometry_mutated": False,
        },
        "binding": {
            "tower_socket_id": "span_out",
            "span_socket_id": "span_start",
            "span_translation_m": [7.45359421, 0.0, 0.0],
        },
        "evidence": {key: result_a[key] for key in expected},
        "bounds_m": result_a["bounds_m"],
        "vertex_count": result_a["vertex_count"],
        "triangle_count": result_a["triangle_count"],
        "reference_tree_digest": _tree_digest(reference_hashes),
        "clean_replay": clean,
        "negative_gates": "DEFERRED_TO_CP2_D",
        "remaining_required_fixtures": ["R0C_CP2_FX02", "R0C_CP2_FX03"],
    }
    qualification_path = reports / "CP2B_FX01_qualification.json"
    qualification_path.write_text(_pretty(qualification), encoding="utf-8")

    completion = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2b-completion/1",
        "status": "PASS",
        "checkpoint": "CP2-B",
        "fixture_id": "R0C_CP2_FX01",
        "completion": "CP2_B_COMPLETE_CLOSED",
        "r0c_cp2": "OPEN",
        "accepted_fixture_count": 1,
        "required_fixture_count": 3,
        "next_exact_checkpoint": "R0C-CP2 CP2-C",
    }
    (reports / "CP2B_FX01_completion.json").write_text(_pretty(completion), encoding="utf-8")

    _update_matrix(cp2 / "data/CP2_FIXTURE_MATRIX.csv", evidence_rel)
    shutil.rmtree(runs)
    print(_pretty(qualification), end="")


if __name__ == "__main__":
    main()
