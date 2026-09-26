#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("--tree", required=True)
parser.add_argument("--work", required=True)
args = parser.parse_args()
tree = Path(args.tree).resolve()
work = Path(args.work).resolve()
fort = tree / "RC_K0/child_designs/fortification"
cp0 = fort / "r0a_cp0"
cp4 = fort / "r0b_cp4"
for source_root in (
    fort / "r0a_cp1/src",
    fort / "r0b_cp1/src",
    fort / "r0b_cp2/src",
    fort / "r0b_cp3/src",
    fort / "r0b_cp4/src",
):
    sys.path.insert(0, str(source_root))

from rcf_fortification_path_span import PathWallSpanProducer
from rcf_fortification_join import WallJoinProducer
from rcf_fortification_terrain_span import TerrainWallSpanProducer
from rcf_fortification_mixed_span import MixedSpanAssemblyProducer
from rcf_fortification_mixed_span.model import canonical_json_bytes, canonicalize_segmentation, sha256_ref


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_map(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in root.rglob("*") if path.is_file()
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
fixture_path = cp4 / "fixtures/mixed_span.fixture.json"
fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
producer = MixedSpanAssemblyProducer(fort)

# Exact source replay: regenerate every R0B source family in the admitted runtime and compare bytes.
replay_root = work / "source-replay"
source_cases = [
    ("straight", PathWallSpanProducer(cp0), fort / "r0b_cp1/fixtures/straight_path_span.fixture.json", fort / "r0b_cp1/outputs/straight/reference"),
    ("curved", PathWallSpanProducer(cp0), fort / "r0b_cp1/fixtures/curved_path_span.fixture.json", fort / "r0b_cp1/outputs/curved/reference"),
    ("miter", WallJoinProducer(cp0), fort / "r0b_cp2/fixtures/miter_join.fixture.json", fort / "r0b_cp2/outputs/miter/reference"),
    ("bevel", WallJoinProducer(cp0), fort / "r0b_cp2/fixtures/bevel_join.fixture.json", fort / "r0b_cp2/outputs/bevel/reference"),
    ("transition", WallJoinProducer(cp0), fort / "r0b_cp2/fixtures/profile_transition_join.fixture.json", fort / "r0b_cp2/outputs/transition/reference"),
    ("stepped", TerrainWallSpanProducer(cp0), fort / "r0b_cp3/fixtures/terrain_stepped_span.fixture.json", fort / "r0b_cp3/outputs/stepped/reference"),
    ("retaining", TerrainWallSpanProducer(cp0), fort / "r0b_cp3/fixtures/retaining_span.fixture.json", fort / "r0b_cp3/outputs/retaining/reference"),
]
source_replay_rows = []
for name, source_producer, source_fixture_path, accepted in source_cases:
    generated = replay_root / name
    source_fixture = json.loads(source_fixture_path.read_text(encoding="utf-8"))
    result = source_producer.execute(source_fixture, generated)
    generated_files, accepted_files = file_map(generated), file_map(accepted)
    same = generated_files == accepted_files
    row = {
        "case": name,
        "status": result.get("status"),
        "generated_file_count": len(generated_files),
        "accepted_file_count": len(accepted_files),
        "file_set_equal": set(generated_files) == set(accepted_files),
        "byte_identical": same,
        "different": sorted(key for key in set(generated_files) | set(accepted_files) if generated_files.get(key) != accepted_files.get(key)),
    }
    source_replay_rows.append(row)
    if result.get("status") != "SUCCEEDED" or not same:
        raise SystemExit(f"source replay failed for {name}: {row}")
source_replay_report = {
    "schema": "royal-capital.fortification.r0b-source-replay/1",
    "status": "PASS",
    "case_count": len(source_replay_rows),
    "cases": source_replay_rows,
}
write_json(cp4 / "reports/source_replay.json", source_replay_report)

# Clean A/B uses different source-array and connection-array order.
output_root = work / "assembly"
a_output = output_root / "A"
b_output = output_root / "B"
result_a = producer.execute(fixture, a_output)
shuffled = copy.deepcopy(fixture)
shuffled["modules"] = list(reversed(shuffled["modules"]))
shuffled["connections"] = list(reversed(shuffled["connections"]))
result_b = producer.execute(shuffled, b_output)
files_a, files_b = file_map(a_output), file_map(b_output)
clean_replay = {
    "schema": "royal-capital.fortification.r0b-cp4-clean-replay/1",
    "status": "PASS" if files_a == files_b else "FAIL",
    "files_compared": len(files_a),
    "file_set_equal": set(files_a) == set(files_b),
    "byte_identical": files_a == files_b,
    "different": sorted(key for key in set(files_a) | set(files_b) if files_a.get(key) != files_b.get(key)),
    "fixture_a_raw_digest": sha256_ref(canonical_json_bytes(fixture)),
    "fixture_b_raw_digest": sha256_ref(canonical_json_bytes(shuffled)),
    "canonical_segmentation_digest_a": canonicalize_segmentation(fixture)["segmentation_digest"],
    "canonical_segmentation_digest_b": canonicalize_segmentation(shuffled)["segmentation_digest"],
}
write_json(cp4 / "reports/clean_replay.json", clean_replay)
if result_a["status"] != "SUCCEEDED" or result_b["status"] != "SUCCEEDED" or clean_replay["status"] != "PASS":
    raise SystemExit("mixed-span clean replay failed")

reference = cp4 / "outputs/reference"
shutil.rmtree(reference, ignore_errors=True)
shutil.copytree(a_output, reference)

# Expected rejections. No negative target may become an accepted directory.
negative_root = work / "negative"
negative_root.mkdir()
negative_rows = []

def run_negative(name: str, mutate) -> None:
    candidate = copy.deepcopy(fixture)
    mutate(candidate)
    target = negative_root / name
    status = "UNEXPECTED_SUCCESS"
    message = ""
    try:
        producer.execute(candidate, target)
    except Exception as exc:
        status = "PASS_EXPECTED_REJECTION"
        message = f"{type(exc).__name__}: {exc}"
    failed_dirs = sorted(path.name for path in negative_root.glob(name + ".failed-*") if path.is_dir())
    row = {
        "case": name,
        "status": status,
        "message": message,
        "accepted_target_exists": target.exists(),
        "preserved_failed_directories": failed_dirs,
    }
    negative_rows.append(row)
    if status != "PASS_EXPECTED_REJECTION" or target.exists():
        raise SystemExit(f"negative gate failed: {row}")

run_negative("duplicate_module_id", lambda value: value["modules"][-1].__setitem__("module_id", value["modules"][0]["module_id"]))
run_negative("span_interval_gap", lambda value: next(row for row in value["modules"] if row["module_id"] == "span/curved")["station"].__setitem__("start_m", 25.0))
run_negative("span_interval_overlap", lambda value: next(row for row in value["modules"] if row["module_id"] == "span/curved")["station"].__setitem__("start_m", 23.0))
run_negative("join_anchor_off_boundary", lambda value: next(row for row in value["modules"] if row["module_id"] == "join/miter")["station"].__setitem__("anchor_m", 24.5))
run_negative("runtime_mismatch", lambda value: value["runtime"].__setitem__("ocp_version", "0.0.0"))
run_negative("module_budget", lambda value: value["budget"].__setitem__("max_modules", 2))
run_negative("source_result_digest", lambda value: next(row for row in value["modules"] if row["module_id"] == "span/straight").__setitem__("source_result_sha256", "sha256:" + "0" * 64))
run_negative("source_mesh_digest", lambda value: next(row for row in value["modules"] if row["module_id"] == "span/curved").__setitem__("source_mesh_sha256", "sha256:" + "0" * 64))
run_negative("connection_socket_mismatch", lambda value: value["connections"][0].__setitem__("from_socket", "BAD_SOCKET"))
run_negative("source_path_escape", lambda value: next(row for row in value["modules"] if row["module_id"] == "span/straight").__setitem__("source_dir", "../outside"))
negative_report = {
    "schema": "royal-capital.fortification.r0b-cp4-negative-gates/1",
    "status": "PASS",
    "case_count": len(negative_rows),
    "cases": negative_rows,
    "all_rejected": all(row["status"] == "PASS_EXPECTED_REJECTION" for row in negative_rows),
    "no_accepted_negative_output": all(not row["accepted_target_exists"] for row in negative_rows),
}
write_json(cp4 / "reports/negative_gates.json", negative_report)

coverage = json.loads((reference / "source-coverage-map.json").read_text(encoding="utf-8"))
segmentation = json.loads((reference / "segmentation-plan.json").read_text(encoding="utf-8"))
alignment = json.loads((reference / "socket-alignment.json").read_text(encoding="utf-8"))
qualification = {
    "schema": "royal-capital.fortification.r0b-cp4-qualification/1",
    "status": "PASS",
    "source_replay": source_replay_report,
    "clean_replay": clean_replay,
    "negative_gates": negative_report,
    "mixed_result": result_a,
    "segmentation": {
        "digest": segmentation["segmentation_digest"],
        "module_count": segmentation["module_count"],
        "span_segment_count": segmentation["span_segment_count"],
        "join_count": segmentation["join_count"],
        "total_station_length_m": segmentation["total_station_length_m"],
    },
    "source_coverage": coverage["summary"],
    "socket_alignment": {
        "status": alignment["status"],
        "alignment_count": alignment["alignment_count"],
    },
    "accepted_reference_digest": "sha256:" + hashlib.sha256("".join(f"{key}:{value}\n" for key, value in sorted(file_map(reference).items())).encode()).hexdigest(),
}
write_json(cp4 / "reports/qualification.json", qualification)
print(json.dumps(qualification, indent=2, sort_keys=True))
