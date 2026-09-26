#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("--a", required=True)
parser.add_argument("--b", required=True)
parser.add_argument("--cp4-root", required=True)
parser.add_argument("--fortification-root", required=True)
parser.add_argument("--report", required=True)
args = parser.parse_args()
ra, rb = Path(args.a), Path(args.b)
cp4 = Path(args.cp4_root)
fort = Path(args.fortification_root)
checks: list[dict[str, Any]] = []


def ck(name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def load(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def file_map(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}

required = {
    "segmentation-plan.json", "mixed-span-plan.json", "instance-transforms.json",
    "socket-alignment.json", "module-index.json", "module-sockets.json",
    "neutral-mesh.json", "source-coverage-map.json", "cad-provider-receipt.json", "result.json",
}
ck("required_files_A", required <= {path.name for path in ra.iterdir()}, sorted(required - {path.name for path in ra.iterdir()}))
ck("required_files_B", required <= {path.name for path in rb.iterdir()}, sorted(required - {path.name for path in rb.iterdir()}))
fa, fb = file_map(ra), file_map(rb)
ck("clean_file_set", set(fa) == set(fb), {"missing": sorted(set(fa)-set(fb)), "extra": sorted(set(fb)-set(fa))})
ck("clean_byte_identical", fa == fb, sorted(key for key in set(fa)|set(fb) if fa.get(key) != fb.get(key)))

result = load(ra, "result.json")
seg = load(ra, "segmentation-plan.json")
plan = load(ra, "mixed-span-plan.json")
transforms = load(ra, "instance-transforms.json")
alignment = load(ra, "socket-alignment.json")
index = load(ra, "module-index.json")
sockets = load(ra, "module-sockets.json")
mesh = load(ra, "neutral-mesh.json")
coverage = load(ra, "source-coverage-map.json")
receipt = load(ra, "cad-provider-receipt.json")
qualification = json.loads((cp4 / "reports/qualification.json").read_text(encoding="utf-8"))
negative = json.loads((cp4 / "reports/negative_gates.json").read_text(encoding="utf-8"))
source_replay = json.loads((cp4 / "reports/source_replay.json").read_text(encoding="utf-8"))
clean_replay = json.loads((cp4 / "reports/clean_replay.json").read_text(encoding="utf-8"))

ck("result_status", result["status"] == "SUCCEEDED", result["status"])
ck("module_count", result["module_count"] == 7, result["module_count"])
ck("span_segment_count", result["span_segment_count"] == 4, result["span_segment_count"])
ck("join_count", result["join_count"] == 3, result["join_count"])
ck("connection_count", result["connection_count"] == 6, result["connection_count"])
ck("vertex_count", result["vertex_count"] == 544, result["vertex_count"])
ck("triangle_count", result["triangle_count"] == 908, result["triangle_count"])
ck("positive_volume", result["volume_m3"] > 0, result["volume_m3"])
ck("finite_bounds", all(math.isfinite(float(x)) for values in result["bounds_m"].values() for x in values), result["bounds_m"])

expected_chain = ["span/straight", "join/miter", "span/curved", "join/bevel", "span/stepped", "span/retaining", "join/transition"]
ck("chain_order", seg["chain_order"] == expected_chain, seg["chain_order"])
ck("array_order_not_identity", seg["input_array_order_is_identity"] is False)
ck("segmentation_digest_match", result["segmentation_digest"] == seg["segmentation_digest"] == plan["segmentation_digest"] == receipt["segmentation_digest"])
ck("span_rows", len(seg["span_segments"]) == 4)
ck("join_rows", len(seg["join_anchors"]) == 3)
ck("boundary_count", len(seg["boundaries_m"]) == 5, seg["boundaries_m"])
ck("span_interval_continuity", all(abs(float(a["station"]["end_m"])-float(b["station"]["start_m"])) <= 1e-9 for a,b in zip(seg["span_segments"],seg["span_segments"][1:])))
ck("stable_segment_keys_unique", len({row["stable_segment_key"] for row in seg["span_segments"]}) == 4)
ck("stable_join_keys_unique", len({row["stable_join_key"] for row in seg["join_anchors"]}) == 3)

ck("transform_count", len(transforms["transforms"]) == 7)
ck("transform_order", [row["module_id"] for row in transforms["transforms"]] == expected_chain)
ck("root_transform", transforms["transforms"][0]["source"] == "ROOT_TRANSFORM")
ck("socket_alignment_status", alignment["status"] == "PASS", alignment["status"])
ck("socket_alignment_count", alignment["alignment_count"] == 6, alignment["alignment_count"])
ck("socket_position_errors", all(float(row["errors"]["position_m"]) <= 1e-6 for row in alignment["alignments"]))
ck("socket_angular_errors", all(max(float(row["errors"][key]) for key in ("tangent_rad","up_rad","inside_rad","outside_rad")) <= 1e-6 for row in alignment["alignments"]))
ck("module_socket_count", sockets["socket_count"] == len(sockets["sockets"]) and sockets["socket_count"] > 0, sockets["socket_count"])
ck("namespaced_socket_ids", len({row["socket_id"] for row in sockets["sockets"]}) == sockets["socket_count"] and all("::" in row["socket_id"] for row in sockets["sockets"]))

ck("module_index_count", index["module_count"] == 7)
ck("module_index_order", index["module_order"] == expected_chain)
ck("module_ranges_count", len(mesh["module_ranges"]) == 7)
ck("module_range_order", [row["module_id"] for row in mesh["module_ranges"]] == expected_chain)
ck("module_range_vertices", sum(int(row["vertex_count"]) for row in mesh["module_ranges"]) == len(mesh["vertices_m"]) == 544)
ck("module_range_triangles", sum(int(row["triangle_count"]) for row in mesh["module_ranges"]) == len(mesh["triangles"]) == 908)
ck("mesh_indices", all(0 <= int(index_value) < len(mesh["vertices_m"]) for tri in mesh["triangles"] for index_value in tri))
ck("mesh_finite", all(math.isfinite(float(value)) for vertex in mesh["vertices_m"] for value in vertex))

summary = coverage["summary"]
ck("coverage_triangle_total", summary["mesh_triangle_count"] == summary["covered_triangle_count"] == summary["unique_triangle_count"] == 908, summary)
ck("coverage_no_gap", summary["gap_count"] == 0 and not summary["gaps"], summary)
ck("coverage_no_overlap", summary["overlap_count"] == 0 and not summary["overlaps"], summary)
all_global = [int(index_value) for row in coverage["surfaces"] for index_value in row["global_triangle_indices"]]
ck("coverage_exclusive", len(all_global) == len(set(all_global)) == 908)
ck("coverage_complete_set", set(all_global) == set(range(908)))
ck("coverage_modules", {row["module_id"] for row in coverage["module_summaries"]} == set(expected_chain))
ck("coverage_source_refs", all(row["source_refs"]["result"]["sha256"].startswith("sha256:") and row["source_refs"]["neutral_mesh"]["sha256"].startswith("sha256:") for row in coverage["surfaces"]))
ck("provider_face_identity_false", coverage["provider_face_identity_used"] is False and receipt["provider_face_identity_used"] is False)

source_ref_failures = []
for module in index["modules"]:
    for key in ("result", "neutral_mesh", "provider_receipt", "stored_copies", "sockets"):
        ref = module["source_refs"][key]
        path = fort / ref["path"]
        if not path.is_file() or digest(path) != ref["sha256"]:
            source_ref_failures.append([module["module_id"], key, ref])
ck("exact_source_refs", not source_ref_failures, source_ref_failures)

ck("qualification_pass", qualification["status"] == "PASS")
ck("source_replay_pass", source_replay["status"] == "PASS" and source_replay["case_count"] == 7 and all(row["byte_identical"] for row in source_replay["cases"]))
ck("clean_replay_pass", clean_replay["status"] == "PASS" and clean_replay["byte_identical"])
ck("shuffle_digest_same", clean_replay["canonical_segmentation_digest_a"] == clean_replay["canonical_segmentation_digest_b"])
ck("negative_gate_count", negative["case_count"] == 10, negative["case_count"])
ck("negative_all_rejected", negative["all_rejected"] and negative["no_accepted_negative_output"])
ck("global_boolean_false", receipt["global_boolean_used"] is False)
ck("terrain_mutation_false", plan["terrain_mutation"] is False)
ck("product_cook_deferred", plan["product_cook"] == "DEFERRED_TO_R0E")

bad_cache = [path.as_posix() for path in cp4.rglob("*") if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts or ".godot" in path.parts)]
ck("no_generated_cache", not bad_cache, bad_cache)

report = {
    "schema": "royal-capital.fortification.r0b-cp4-validation/1",
    "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
    "checks": checks,
    "summary": {"passed": sum(row["pass"] for row in checks), "failed": sum(not row["pass"] for row in checks), "files_compared": len(fa)},
    "canonical_result_sha256": hashlib.sha256((ra / "result.json").read_bytes()).hexdigest(),
}
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["status"] == "PASS" else 1)
