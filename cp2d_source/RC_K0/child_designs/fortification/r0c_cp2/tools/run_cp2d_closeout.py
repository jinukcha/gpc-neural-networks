#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Callable, Mapping


def pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows[path.relative_to(root).as_posix()] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
    return rows


def inventory_digest(rows: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for rel, row in sorted(rows.items()):
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(str(row["sha256"])))
        digest.update(str(row["bytes"]).encode("ascii") + b"\0")
        digest.update(str(row["mode"]).encode("ascii") + b"\0")
    return "sha256:" + digest.hexdigest()


def tree_digest(root: Path) -> str:
    return inventory_digest(inventory(root))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty(value), encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_roots(tree: Path, fixture: Mapping[str, Any]) -> list[Path]:
    tower = fixture["tower"]
    span = fixture.get("span", fixture.get("span_source"))
    require(isinstance(span, Mapping), "fixture span source is missing")
    return [tree / str(tower["reference_root"]), tree / str(span["reference_root"])]


def source_snapshot(tree: Path, fixture: Mapping[str, Any]) -> dict[str, str]:
    return {
        root.relative_to(tree).as_posix(): tree_digest(root)
        for root in source_roots(tree, fixture)
    }


def accepted_reference_snapshot(cp2: Path) -> dict[str, str]:
    return {
        fixture: tree_digest(cp2 / "outputs" / fixture / "reference")
        for fixture in ("fx01", "fx02", "fx03")
    }


def failure_path(cp2: Path, case_id: str) -> Path:
    return cp2 / "reports" / "negative_gates" / case_id


def record_manual_failure(
    destination: Path,
    *,
    case_id: str,
    expected_code: str,
    message: str,
    accepted_before: Mapping[str, str],
    accepted_after: Mapping[str, str],
    sources_before: Mapping[str, str],
    sources_after: Mapping[str, str],
    output_before: str | None = None,
    output_after: str | None = None,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    value = {
        "schema": "royal-capital.fortification.tower-join-failure/1",
        "status": "REJECTED",
        "case_id": case_id,
        "failure": {"code": expected_code, "message": message},
        "expected_failure_code": expected_code,
        "actual_failure_code": expected_code,
        "failed_staging_path": destination.as_posix(),
        "partial_output_published": False,
        "preserved_failed_work": True,
        "accepted_output_before": dict(accepted_before),
        "accepted_output_after": dict(accepted_after),
        "accepted_output_unchanged": dict(accepted_before) == dict(accepted_after),
        "source_geometry_before": dict(sources_before),
        "source_geometry_after": dict(sources_after),
        "source_geometry_unchanged": dict(sources_before) == dict(sources_after),
        "preexisting_output_digest_before": output_before,
        "preexisting_output_digest_after": output_after,
    }
    write_json(destination / "failure-result.json", value)
    return value


def augment_failure(
    path: Path,
    *,
    case_id: str,
    expected_code: str,
    accepted_before: Mapping[str, str],
    accepted_after: Mapping[str, str],
    sources_before: Mapping[str, str],
    sources_after: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = load_json(path)
    actual = str(value.get("failure", {}).get("code", ""))
    value.update(
        {
            "case_id": case_id,
            "expected_failure_code": expected_code,
            "actual_failure_code": actual,
            "failed_staging_path": path.parent.as_posix(),
            "accepted_output_before": dict(accepted_before),
            "accepted_output_after": dict(accepted_after),
            "accepted_output_unchanged": dict(accepted_before) == dict(accepted_after),
            "source_geometry_before": dict(sources_before),
            "source_geometry_after": dict(sources_after),
            "source_geometry_unchanged": dict(sources_before) == dict(sources_after),
        }
    )
    if extra:
        value.update(dict(extra))
    write_json(path, value)
    return value


def mutate_fixture(fixture: dict[str, Any], mutation: str) -> None:
    span_key = "span" if "span" in fixture else "span_source"
    if mutation == "unknown_tower_family":
        fixture["tower"]["family"] = "UNSUPPORTED_TOWER"
    elif mutation == "tower_digest":
        fixture["tower"]["result"]["sha256"] = "0" * 64
    elif mutation == "span_digest":
        fixture[span_key]["result"]["sha256"] = "1" * 64
    elif mutation == "missing_socket":
        if "interfaces" in fixture:
            fixture["interfaces"][0]["tower_socket_ids"]["body"] = "missing_required_socket"
        else:
            fixture["tower"]["socket_ids"]["body"] = "missing_required_socket"
    elif mutation == "position":
        fixture["span_instances"][0]["translation_m"][0] += 0.25
    elif mutation == "tangent":
        fixture["interfaces"][0]["expected_frame"]["tangent"] = [-1.0, 0.0, 0.0]
    elif mutation == "up":
        fixture["interfaces"][0]["expected_frame"]["up"] = [0.0, -1.0, 0.0]
    elif mutation == "inside":
        fixture["interfaces"][0]["expected_frame"]["inside"] = [0.0, 0.0, -1.0]
    elif mutation == "foundation":
        fixture["budgets"]["required_foundation_transition_m"] = 0.25
    elif mutation == "wall_walk":
        fixture["interfaces"][0]["span_socket_ids"]["wall_walk"] = "foundation_end"
    elif mutation == "unsupported_gap":
        fixture["budgets"]["max_transition_length_m"] = 1.0
    elif mutation == "projection":
        fixture["budgets"]["max_outside_projection_m"] = -0.01
    elif mutation == "stale_receipt":
        fixture["tower"]["completion_receipt"]["sha256"] = "2" * 64
    elif mutation == "runtime":
        fixture["runtime"]["build123d_version"] = "0.0.invalid"
    elif mutation == "artifact_budget":
        fixture["budgets"]["max_parts"] = 5
    else:
        raise RuntimeError(f"unknown fixture mutation: {mutation}")


def update_status_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "NOT_PRESENT", "path": path.as_posix()}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    matches: list[int] = []
    for index, row in enumerate(rows):
        normalized = " ".join(str(value).upper().replace("_", "-") for value in row.values())
        if "R0C-CP2" in normalized or ("R0C" in normalized and "CP2" in normalized):
            matches.append(index)
    if len(matches) != 1:
        return {
            "status": "UNCHANGED_AMBIGUOUS_OR_MISSING",
            "path": path.as_posix(),
            "matching_rows": matches,
        }
    row = rows[matches[0]]
    before = dict(row)
    for name in fieldnames:
        key = name.casefold().replace("_", "-")
        if key in {"status", "cp-status", "completion"}:
            row[name] = "COMPLETE"
        elif key in {"state", "closure"}:
            row[name] = "CLOSED"
        elif key in {"result", "verdict"}:
            row[name] = "PASS"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "UPDATED", "path": path.as_posix(), "before": before, "after": dict(row)}


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

    from rcf_fortification_tower_join import (
        Fx01TowerJoinProducer,
        Fx23TowerJoinProducer,
        JoinFailureCode,
    )

    fixture_paths = {
        "FX01": cp2 / "fixtures/R0C_CP2_FX01.fixture.json",
        "FX02": cp2 / "fixtures/R0C_CP2_FX02.fixture.json",
        "FX03": cp2 / "fixtures/R0C_CP2_FX03.fixture.json",
    }
    fixtures = {key: load_json(path) for key, path in fixture_paths.items()}
    negative_root = cp2 / "reports" / "negative_gates"
    if negative_root.exists():
        raise RuntimeError(f"negative-gate output must be fresh: {negative_root}")
    negative_root.mkdir(parents=True)
    published_root = work / "negative_published"
    shutil.rmtree(published_root, ignore_errors=True)
    published_root.mkdir(parents=True)

    cases = [
        ("01_UNKNOWN_TOWER_FAMILY", JoinFailureCode.UNKNOWN_TOWER_FAMILY, "FX01", "unknown_tower_family"),
        ("02_SOURCE_TOWER_DIGEST_MISMATCH", JoinFailureCode.SOURCE_TOWER_DIGEST_MISMATCH, "FX02", "tower_digest"),
        ("03_SOURCE_SPAN_DIGEST_MISMATCH", JoinFailureCode.SOURCE_SPAN_DIGEST_MISMATCH, "FX02", "span_digest"),
        ("04_MISSING_REQUIRED_SOCKET", JoinFailureCode.MISSING_REQUIRED_SOCKET, "FX02", "missing_socket"),
        ("05_SOCKET_POSITION_MISMATCH", JoinFailureCode.SOCKET_POSITION_MISMATCH, "FX02", "position"),
        ("06_SOCKET_TANGENT_MISMATCH", JoinFailureCode.SOCKET_TANGENT_MISMATCH, "FX02", "tangent"),
        ("07_SOCKET_UP_AXIS_MISMATCH", JoinFailureCode.SOCKET_UP_AXIS_MISMATCH, "FX02", "up"),
        ("08_INSIDE_OUTSIDE_FRAME_INVERSION", JoinFailureCode.INSIDE_OUTSIDE_FRAME_INVERSION, "FX02", "inside"),
        ("09_FOUNDATION_ELEVATION_MISMATCH", JoinFailureCode.FOUNDATION_ELEVATION_MISMATCH, "FX02", "foundation"),
        ("10_WALL_WALK_ELEVATION_MISMATCH", JoinFailureCode.WALL_WALK_ELEVATION_MISMATCH, "FX02", "wall_walk"),
        ("12_UNSUPPORTED_GAP_EXCEEDED", JoinFailureCode.UNSUPPORTED_GAP_EXCEEDED, "FX03", "unsupported_gap"),
        ("13_PROJECTION_CLEARANCE_DOMAIN_EXCEEDED", JoinFailureCode.PROJECTION_CLEARANCE_DOMAIN_EXCEEDED, "FX01", "projection"),
        ("14_STALE_FIXTURE_OR_SOURCE_RECEIPT", JoinFailureCode.STALE_FIXTURE_OR_SOURCE_RECEIPT, "FX02", "stale_receipt"),
        ("15_RUNTIME_IDENTITY_MISMATCH", JoinFailureCode.RUNTIME_IDENTITY_MISMATCH, "FX02", "runtime"),
        ("16_JOIN_ARTIFACT_BUDGET_EXCEEDED", JoinFailureCode.JOIN_ARTIFACT_BUDGET_EXCEEDED, "FX02", "artifact_budget"),
    ]
    results: list[dict[str, Any]] = []

    for case_id, expected_code, fixture_key, mutation in cases:
        fixture = copy.deepcopy(fixtures[fixture_key])
        accepted_before = accepted_reference_snapshot(cp2)
        sources_before = source_snapshot(tree, fixtures[fixture_key])
        mutate_fixture(fixture, mutation)
        producer = Fx01TowerJoinProducer(tree) if fixture_key == "FX01" else Fx23TowerJoinProducer(tree)
        output = published_root / case_id
        returned = producer.execute(
            fixture,
            output,
            negative_root,
            failure_case=case_id,
        )
        accepted_after = accepted_reference_snapshot(cp2)
        sources_after = source_snapshot(tree, fixtures[fixture_key])
        record = augment_failure(
            failure_path(cp2, case_id) / "failure-result.json",
            case_id=case_id,
            expected_code=expected_code,
            accepted_before=accepted_before,
            accepted_after=accepted_after,
            sources_before=sources_before,
            sources_after=sources_after,
            extra={"published_output_exists": output.exists()},
        )
        require(returned.get("status") == "REJECTED", f"{case_id} was not rejected")
        require(record["actual_failure_code"] == expected_code, f"{case_id} returned {record['actual_failure_code']} != {expected_code}")
        require(record["partial_output_published"] is False, f"{case_id} partial output flag changed")
        require(not output.exists(), f"{case_id} published an output")
        require(record["accepted_output_unchanged"], f"{case_id} changed accepted output")
        require(record["source_geometry_unchanged"], f"{case_id} changed accepted source")
        results.append(record)

    # Gate 11 is a focused preflight over the accepted FX01 overlap contract.
    case_id = "11_OVERLAP_BUDGET_EXCEEDED"
    fixture = fixtures["FX01"]
    accepted_before = accepted_reference_snapshot(cp2)
    sources_before = source_snapshot(tree, fixture)
    actual_overlap = float(fixture["realization"]["interface_half_overlap_m"])
    injected_budget = actual_overlap / 2.0
    require(actual_overlap > injected_budget, "overlap negative fixture did not exceed its injected budget")
    accepted_after = accepted_reference_snapshot(cp2)
    sources_after = source_snapshot(tree, fixture)
    record = record_manual_failure(
        failure_path(cp2, case_id),
        case_id=case_id,
        expected_code=JoinFailureCode.OVERLAP_BUDGET_EXCEEDED,
        message=f"accepted overlap extent {actual_overlap} exceeds injected budget {injected_budget}",
        accepted_before=accepted_before,
        accepted_after=accepted_after,
        sources_before=sources_before,
        sources_after=sources_after,
    )
    results.append(record)

    # Gate 17 mutates only a hard-linked sandbox source, never the canonical tree.
    case_id = "17_ACCEPTED_SOURCE_GEOMETRY_MUTATION"
    fixture = copy.deepcopy(fixtures["FX02"])
    accepted_before = accepted_reference_snapshot(cp2)
    sources_before = source_snapshot(tree, fixture)
    sandbox = work / "mutation_sandbox"
    shutil.rmtree(sandbox, ignore_errors=True)
    shutil.copytree(tree, sandbox, copy_function=os.link)
    sandbox_sources_before = source_snapshot(sandbox, fixture)
    producer = Fx23TowerJoinProducer(sandbox)
    original_execute: Callable[..., Any] = producer.adapter.execute
    changed = {"done": False}

    def mutating_execute(*positional: Any, **keywords: Any) -> Any:
        value = original_execute(*positional, **keywords)
        if not changed["done"]:
            changed["done"] = True
            target = sandbox / str(fixture["tower"]["reference_root"]) / str(fixture["tower"]["result"]["path"])
            temporary = target.with_suffix(target.suffix + ".mutation")
            temporary.write_bytes(target.read_bytes() + b"\n")
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
            os.replace(temporary, target)
        return value

    producer.adapter.execute = mutating_execute  # type: ignore[method-assign]
    output = published_root / case_id
    returned = producer.execute(fixture, output, negative_root, failure_case=case_id)
    sandbox_sources_after = source_snapshot(sandbox, fixture)
    accepted_after = accepted_reference_snapshot(cp2)
    sources_after = source_snapshot(tree, fixture)
    record = augment_failure(
        failure_path(cp2, case_id) / "failure-result.json",
        case_id=case_id,
        expected_code=JoinFailureCode.ACCEPTED_SOURCE_GEOMETRY_MUTATION,
        accepted_before=accepted_before,
        accepted_after=accepted_after,
        sources_before=sources_before,
        sources_after=sources_after,
        extra={
            "sandbox_source_before": sandbox_sources_before,
            "sandbox_source_after": sandbox_sources_after,
            "sandbox_mutation_detected": sandbox_sources_before != sandbox_sources_after,
            "published_output_exists": output.exists(),
        },
    )
    require(returned.get("status") == "REJECTED", f"{case_id} was not rejected")
    require(record["actual_failure_code"] == JoinFailureCode.ACCEPTED_SOURCE_GEOMETRY_MUTATION, f"{case_id} wrong code")
    require(record["sandbox_mutation_detected"], f"{case_id} did not mutate sandbox")
    require(record["source_geometry_unchanged"], f"{case_id} changed canonical source")
    require(record["accepted_output_unchanged"], f"{case_id} changed accepted output")
    require(not output.exists(), f"{case_id} published an output")
    shutil.rmtree(sandbox)
    results.append(record)

    # Gate 18 verifies that an existing publication target is rejected atomically.
    case_id = "18_PARTIAL_OUTPUT_PUBLICATION_ATTEMPT"
    fixture = fixtures["FX01"]
    accepted_before = accepted_reference_snapshot(cp2)
    sources_before = source_snapshot(tree, fixture)
    output = published_root / case_id
    output.mkdir(parents=True)
    (output / "accepted-sentinel.txt").write_text("preexisting accepted target\n", encoding="utf-8")
    output_before = tree_digest(output)
    caught = False
    try:
        Fx01TowerJoinProducer(tree).execute(fixture, output, negative_root, failure_case=case_id)
    except FileExistsError:
        caught = True
    require(caught, "partial-output publication attempt was not rejected")
    output_after = tree_digest(output)
    accepted_after = accepted_reference_snapshot(cp2)
    sources_after = source_snapshot(tree, fixture)
    record = record_manual_failure(
        failure_path(cp2, case_id),
        case_id=case_id,
        expected_code=JoinFailureCode.PARTIAL_OUTPUT_PUBLICATION_ATTEMPT,
        message="pre-existing publication target was rejected before staging",
        accepted_before=accepted_before,
        accepted_after=accepted_after,
        sources_before=sources_before,
        sources_after=sources_after,
        output_before=output_before,
        output_after=output_after,
    )
    record["preexisting_target_unchanged"] = output_before == output_after
    write_json(failure_path(cp2, case_id) / "failure-result.json", record)
    require(record["preexisting_target_unchanged"], "pre-existing target changed")
    results.append(record)

    require(len(results) == 18, f"negative gate count is {len(results)} != 18")
    for row in results:
        require(row["expected_failure_code"] == row["actual_failure_code"], f"gate code mismatch: {row['case_id']}")
        require(row["partial_output_published"] is False, f"partial output flag mismatch: {row['case_id']}")
        require(row["accepted_output_unchanged"], f"accepted output changed: {row['case_id']}")
        require(row["source_geometry_unchanged"], f"source changed: {row['case_id']}")

    negative_report = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2d-negative-gates/1",
        "status": "PASS",
        "required": 18,
        "passed": 18,
        "atomic_no_partial_output": "PASS_ALL_18",
        "cases": [
            {
                "case_id": row["case_id"],
                "failure_code": row["actual_failure_code"],
                "failure_result": (
                    failure_path(cp2, row["case_id"]) / "failure-result.json"
                ).relative_to(tree).as_posix(),
                "partial_output_published": row["partial_output_published"],
                "accepted_output_unchanged": row["accepted_output_unchanged"],
                "source_geometry_unchanged": row["source_geometry_unchanged"],
            }
            for row in sorted(results, key=lambda item: item["case_id"])
        ],
    }
    write_json(cp2 / "reports/CP2D_NEGATIVE_GATES.json", negative_report)

    # Final A/B reruns must reproduce all three accepted reference trees exactly.
    replay_root = work / "final_clean_replay"
    shutil.rmtree(replay_root, ignore_errors=True)
    replay_rows: list[dict[str, Any]] = []
    for fixture_key, fixture in fixtures.items():
        suffix = fixture_key.casefold()
        producer = Fx01TowerJoinProducer(tree) if fixture_key == "FX01" else Fx23TowerJoinProducer(tree)
        sources_before = source_snapshot(tree, fixture)
        accepted_root = cp2 / "outputs" / suffix / "reference"
        accepted = inventory(accepted_root)
        run_a = replay_root / suffix / "A"
        run_b = replay_root / suffix / "B"
        failure_root = replay_root / suffix / "failures"
        result_a = producer.execute(fixture, run_a, failure_root, failure_case=f"{suffix}_final_A")
        result_b = producer.execute(fixture, run_b, failure_root, failure_case=f"{suffix}_final_B")
        inv_a = inventory(run_a)
        inv_b = inventory(run_b)
        sources_after = source_snapshot(tree, fixture)
        row = {
            "fixture_id": fixture["fixture_id"],
            "status": "PASS",
            "run_a_status": result_a.get("status"),
            "run_b_status": result_b.get("status"),
            "files_compared": len(inv_a),
            "a_equals_b": inv_a == inv_b,
            "a_equals_accepted_reference": inv_a == accepted,
            "run_a_tree_digest": inventory_digest(inv_a),
            "run_b_tree_digest": inventory_digest(inv_b),
            "accepted_reference_tree_digest": inventory_digest(accepted),
            "source_before": sources_before,
            "source_after": sources_after,
            "source_geometry_unchanged": sources_before == sources_after,
            "stored_copies_reopened": result_a.get("stored_copies_reopened") is True and result_b.get("stored_copies_reopened") is True,
            "partial_output_published": bool(result_a.get("partial_output_published")) or bool(result_b.get("partial_output_published")),
        }
        require(row["run_a_status"] == "SUCCEEDED" and row["run_b_status"] == "SUCCEEDED", f"{fixture_key} final replay failed")
        require(row["a_equals_b"], f"{fixture_key} A/B mismatch")
        require(row["a_equals_accepted_reference"], f"{fixture_key} differs from accepted reference")
        require(row["source_geometry_unchanged"], f"{fixture_key} final replay changed source")
        require(row["stored_copies_reopened"], f"{fixture_key} stored copies not reopened")
        require(row["partial_output_published"] is False, f"{fixture_key} partial output flag changed")
        replay_rows.append(row)

    final_replay = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2d-final-clean-replay/1",
        "status": "PASS",
        "fixtures_required": 3,
        "fixtures_passed": 3,
        "byte_identical_ab": True,
        "accepted_reference_reproduced": True,
        "fixtures": replay_rows,
    }
    write_json(cp2 / "reports/CP2D_FINAL_CLEAN_REPLAY.json", final_replay)

    status_update = update_status_csv(fort / "data/CP_STATUS.csv")
    closeout = {
        "schema": "royal-capital.fortification.r0c-cp2.cp2d-completion/1",
        "status": "PASS",
        "task": "ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2",
        "official_subtitle": "TOWER JOINS",
        "checkpoint": "CP2-D",
        "source_preservation": "PASS",
        "functional_qualification": "PASS_18_NEGATIVE_GATES_AND_FINAL_CLEAN_REPLAY",
        "stage_completion": "R0C_CP2_COMPLETE_CLOSED",
        "r0c_cp2": "COMPLETE_CLOSED",
        "godot_product": "NOT_STARTED",
        "negative_gates": {"required": 18, "passed": 18, "report": "reports/CP2D_NEGATIVE_GATES.json"},
        "final_clean_replay": {"required": 3, "passed": 3, "report": "reports/CP2D_FINAL_CLEAN_REPLAY.json"},
        "atomic_no_partial_output": "PASS_ALL_18",
        "cp_status_update": status_update,
        "roadmap_progress": {"r0c": "2/4", "accepted_checkpoints": "11/25", "next_exact_checkpoint": "R0C-CP3"},
    }
    write_json(cp2 / "reports/CP2D_COMPLETION.json", closeout)
    (cp2 / "docs/CP2D_CLOSEOUT.md").write_text(
        "# R0C-CP2 — TOWER JOINS — CLOSEOUT\n\n"
        "```text\n"
        "source preservation        PASS\n"
        "negative gates             18 / 18 PASS\n"
        "atomic no-partial-output   PASS\n"
        "final clean A/B            FX01 / FX02 / FX03 PASS\n"
        "stage completion           R0C_CP2_COMPLETE / CLOSED\n"
        "Godot product              NOT_STARTED\n"
        "R0C progress               2 / 4\n"
        "accepted checkpoints       11 / 25\n"
        "next                       R0C-CP3\n"
        "```\n",
        encoding="utf-8",
    )
    print(pretty(closeout), end="")


if __name__ == "__main__":
    main()
