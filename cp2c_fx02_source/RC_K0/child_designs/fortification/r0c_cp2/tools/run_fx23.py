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
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8") + b"\0" + bytes.fromhex(value))
    return "sha256:" + digest.hexdigest()


def _read_matrix(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise SystemExit("fixture matrix header missing")
    return fieldnames, rows


def _update_matrix(path: Path, fixture_id: str, evidence_ref: str) -> list[dict[str, str]]:
    fieldnames, rows = _read_matrix(path)
    by_id = {row["fixture_id"]: row for row in rows}
    required = ("R0C_CP2_FX01", "R0C_CP2_FX02", "R0C_CP2_FX03")
    if any(key not in by_id for key in required):
        raise SystemExit("required fixture row missing")
    if by_id["R0C_CP2_FX01"]["status"] != "PASS_CP2_B":
        raise SystemExit(f"FX01 accepted status changed: {by_id['R0C_CP2_FX01']['status']}")
    row = by_id[fixture_id]
    if fixture_id == "R0C_CP2_FX02":
        if row["status"] != "FROZEN_PENDING_IMPLEMENTATION":
            raise SystemExit(f"unexpected FX02 pre-run status: {row['status']}")
        if by_id["R0C_CP2_FX03"]["status"] != "FROZEN_PENDING_IMPLEMENTATION":
            raise SystemExit("FX03 changed before its implementation unit")
        row["exact_source_binding"] = "BOUND_EXACT_ACCEPTED_IDS_DIGESTS_SOCKETS_AND_INSTANCE_TRANSFORMS"
        row["status"] = "PASS_CP2_C_FX02"
        row["authority_note"] = (
            "CP2-C accepted SQUARE×CORNER×STRAIGHT_SPAN fixture; "
            f"evidence={evidence_ref}"
        )
    elif fixture_id == "R0C_CP2_FX03":
        if by_id["R0C_CP2_FX02"]["status"] != "PASS_CP2_C_FX02":
            raise SystemExit("FX02 is not accepted before FX03")
        if row["status"] != "FROZEN_PENDING_IMPLEMENTATION":
            raise SystemExit(f"unexpected FX03 pre-run status: {row['status']}")
        by_id["R0C_CP2_FX02"]["status"] = "PASS_CP2_C"
        row["exact_source_binding"] = "BOUND_EXACT_ACCEPTED_IDS_DIGESTS_SOCKETS_AND_INSTANCE_TRANSFORMS"
        row["status"] = "PASS_CP2_C"
        row["authority_note"] = (
            "CP2-C accepted POLYGONAL×WALL_PENETRATING×STRAIGHT_SPAN fixture; "
            f"evidence={evidence_ref}"
        )
    else:
        raise SystemExit(f"unsupported fixture id: {fixture_id}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _require_equal(result: dict[str, Any], key: str, expected: Any) -> None:
    if result.get(key) != expected:
        raise SystemExit(f"evidence mismatch {key}: {result.get(key)!r} != {expected!r}")


def _require_near(result: dict[str, Any], key: str, expected: float, tolerance: float = 1e-9) -> None:
    actual = result.get(key)
    if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > tolerance:
        raise SystemExit(f"evidence mismatch {key}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--fixture-id", choices=("R0C_CP2_FX02", "R0C_CP2_FX03"), required=True)
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    work = Path(args.work).resolve()
    fixture_id = args.fixture_id
    suffix = fixture_id[-4:].lower()
    fort = tree / "RC_K0/child_designs/fortification"
    cp1 = fort / "r0a_cp1"
    cp2 = fort / "r0c_cp2"
    sys.path[:0] = [str(cp1 / "src"), str(cp2 / "src")]

    from rcf_fortification_tower_join import Fx23TowerJoinProducer

    fixture_path = cp2 / "fixtures" / f"{fixture_id}.fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    runs = work / f"{suffix}_runs"
    failures = work / "failures" / suffix
    shutil.rmtree(runs, ignore_errors=True)
    shutil.rmtree(failures, ignore_errors=True)
    runs.mkdir(parents=True)
    failures.mkdir(parents=True)
    reference = cp2 / "outputs" / suffix / "reference"
    if reference.exists():
        raise SystemExit(f"accepted reference path must be fresh: {reference}")

    producer = Fx23TowerJoinProducer(tree)
    run_a = runs / "A"
    run_b = runs / "B"
    result_a = producer.execute(fixture, run_a, failures, failure_case=f"{suffix}_positive_A")
    result_b = producer.execute(fixture, run_b, failures, failure_case=f"{suffix}_positive_B")
    if result_a.get("status") != "SUCCEEDED" or result_b.get("status") != "SUCCEEDED":
        raise SystemExit(f"{fixture_id} positive run failed: A={result_a} B={result_b}")

    hashes_a = _file_hashes(run_a)
    hashes_b = _file_hashes(run_b)
    clean = {
        "schema": f"royal-capital.fortification.r0c-cp2.{suffix}-clean-replay/1",
        "fixture_id": fixture_id,
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
        raise SystemExit(f"{fixture_id} clean replay mismatch: {clean}")

    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_a, reference)
    reference_hashes = _file_hashes(reference)
    if reference_hashes != hashes_a:
        raise SystemExit("published reference does not match clean run A")

    for key, expected in {
        "interface_count": 2,
        "part_count": 6,
        "stored_copy_count": 12,
        "socket_position_error_m": 0.0,
        "tangent_error": 0.0,
        "up_axis_error": 0.0,
        "inside_frame_error": 0.0,
        "wall_walk_position_error_m": 0.0,
        "foundation_transition_residual_m": 0.0,
        "unsupported_gap_m": 0.0,
        "outside_projection_m": 0.0,
        "inside_projection_m": 0.0,
        "accepted_sources_unchanged": True,
        "stored_copies_reopened": True,
        "partial_output_published": False,
    }.items():
        _require_equal(result_a, key, expected)
    if result_a.get("foundation_source_offsets_m") != [0.5, 0.5]:
        raise SystemExit(f"foundation offsets mismatch: {result_a.get('foundation_source_offsets_m')!r}")
    if fixture_id == "R0C_CP2_FX02":
        _require_equal(result_a, "span_instance_count", 2)
        _require_near(result_a, "maximum_source_socket_separation_m", 0.0)
        _require_near(result_a, "corner_angle_deg", 90.0)
        expected = {
            "tower_family": "SQUARE",
            "join_kind": "CORNER",
            "span_family": "STRAIGHT_SPAN",
            "realization": "BOUNDED_CORNER_TRANSITIONS",
        }
        stage_completion = "CP2_C_FX02_COMPLETE_CLOSED"
        functional = "PASS_FX02"
    else:
        _require_equal(result_a, "span_instance_count", 1)
        _require_near(result_a, "maximum_source_socket_separation_m", 4.656854249)
        _require_equal(result_a, "corner_angle_deg", None)
        expected = {
            "tower_family": "POLYGONAL",
            "join_kind": "WALL_PENETRATING",
            "span_family": "STRAIGHT_SPAN",
            "realization": "BOUNDED_PENETRATION_TRANSITIONS",
        }
        stage_completion = "CP2_C_COMPLETE_CLOSED"
        functional = "PASS_FX03"

    reports = cp2 / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    clean_path = reports / f"CP2C_{suffix.upper()}_clean_replay.json"
    clean_path.write_text(_pretty(clean), encoding="utf-8")
    evidence_rel = (
        f"RC_K0/child_designs/fortification/r0c_cp2/reports/"
        f"CP2C_{suffix.upper()}_qualification.json"
    )
    matrix_rows = _update_matrix(cp2 / "data/CP2_FIXTURE_MATRIX.csv", fixture_id, evidence_rel)
    qualification = {
        "schema": f"royal-capital.fortification.r0c-cp2.{suffix}-qualification/1",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "official_subtitle": "TOWER JOINS",
        "checkpoint": f"CP2-C_{fixture_id}",
        "fixture_id": fixture_id,
        "status": "PASS",
        "source_preservation": "PASS",
        "functional_qualification": functional,
        "stage_completion": stage_completion,
        "r0c_cp2": "OPEN",
        "godot_product": "NOT_STARTED",
        "implemented": {
            **expected,
            "source_geometry_mutated": False,
        },
        "evidence": {
            "interface_count": result_a["interface_count"],
            "span_instance_count": result_a["span_instance_count"],
            "part_count": result_a["part_count"],
            "stored_copy_count": result_a["stored_copy_count"],
            "socket_position_error_m": result_a["socket_position_error_m"],
            "maximum_source_socket_separation_m": result_a["maximum_source_socket_separation_m"],
            "tangent_error": result_a["tangent_error"],
            "up_axis_error": result_a["up_axis_error"],
            "inside_frame_error": result_a["inside_frame_error"],
            "wall_walk_position_error_m": result_a["wall_walk_position_error_m"],
            "foundation_source_offsets_m": result_a["foundation_source_offsets_m"],
            "foundation_transition_residual_m": result_a["foundation_transition_residual_m"],
            "unsupported_gap_m": result_a["unsupported_gap_m"],
            "outside_projection_m": result_a["outside_projection_m"],
            "inside_projection_m": result_a["inside_projection_m"],
            "corner_angle_deg": result_a["corner_angle_deg"],
            "accepted_sources_unchanged": result_a["accepted_sources_unchanged"],
            "stored_copies_reopened": result_a["stored_copies_reopened"],
            "partial_output_published": result_a["partial_output_published"],
        },
        "bounds_m": result_a["bounds_m"],
        "volume_m3": result_a["volume_m3"],
        "vertex_count": result_a["vertex_count"],
        "triangle_count": result_a["triangle_count"],
        "interface_evidence": result_a["interface_evidence"],
        "reference_tree_digest": _tree_digest(reference_hashes),
        "clean_replay": clean,
        "negative_gates": "DEFERRED_TO_CP2_D",
    }
    qualification_path = reports / f"CP2C_{suffix.upper()}_qualification.json"
    qualification_path.write_text(_pretty(qualification), encoding="utf-8")
    completion = {
        "schema": f"royal-capital.fortification.r0c-cp2.{suffix}-completion/1",
        "status": "PASS",
        "checkpoint": "CP2-C",
        "fixture_id": fixture_id,
        "completion": stage_completion,
        "r0c_cp2": "OPEN",
        "next_exact_checkpoint": (
            "R0C-CP2 CP2-C R0C_CP2_FX03"
            if fixture_id == "R0C_CP2_FX02"
            else "R0C-CP2 CP2-D"
        ),
    }
    (reports / f"CP2C_{suffix.upper()}_completion.json").write_text(_pretty(completion), encoding="utf-8")

    if fixture_id == "R0C_CP2_FX03":
        by_id = {row["fixture_id"]: row for row in matrix_rows}
        required_statuses = {
            "R0C_CP2_FX01": "PASS_CP2_B",
            "R0C_CP2_FX02": "PASS_CP2_C",
            "R0C_CP2_FX03": "PASS_CP2_C",
        }
        for key, value in required_statuses.items():
            if by_id[key]["status"] != value:
                raise SystemExit(f"final matrix status mismatch {key}: {by_id[key]['status']}")
        aggregate = {
            "schema": "royal-capital.fortification.r0c-cp2.cp2c-family-matrix/1",
            "status": "PASS",
            "checkpoint": "CP2-C",
            "stage_completion": "CP2_C_COMPLETE_CLOSED",
            "r0c_cp2": "OPEN",
            "required_fixtures": required_statuses,
            "continuity": {
                "wall_body": "PASS_ALL_REQUIRED_FIXTURES",
                "wall_walk": "PASS_ALL_REQUIRED_FIXTURES",
                "foundation": "PASS_ALL_REQUIRED_FIXTURES",
            },
            "clean_replay": {
                "FX01": "PASS_ACCEPTED_CP2_B",
                "FX02": "PASS",
                "FX03": "PASS",
            },
            "negative_gates": "DEFERRED_TO_CP2_D",
            "godot_product": "NOT_STARTED",
        }
        (reports / "CP2C_FAMILY_MATRIX_qualification.json").write_text(
            _pretty(aggregate), encoding="utf-8"
        )

    shutil.rmtree(runs)
    if failures.is_dir() and not any(failures.iterdir()):
        failures.rmdir()
    print(_pretty(qualification), end="")


if __name__ == "__main__":
    main()
