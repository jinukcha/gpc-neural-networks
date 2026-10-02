#!/usr/bin/env python3
"""Bounded CP5 validation for products, contracts, evidence, and source budgets."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_measurement(build: Path) -> dict:
    payload = load_json(build / "arm_measurements.json")
    values = payload["measurements"]
    assert payload["contract"] == "ArmMeasurementSet/1"
    assert payload["mesh_scaling"] == "FORBIDDEN"
    assert abs(values["sleeve_length_m"] - values["shoulder_to_elbow_m"] - values["elbow_to_wrist_m"]) < 1.0e-9
    assert 1.01 <= values["cap_ease_ratio"] <= 1.12
    return values


def _validate_product(root: Path, build: Path, owner: str) -> dict:
    payload = load_json(build / owner / "rigged_garment_product.json")
    assert (root / payload["path"]).is_file()
    assert payload["bone_count"] == 23
    assert payload["zero_weight_vertex_count"] == 0
    assert payload["maximum_weight_sum_error"] <= 1.0e-6
    assert payload["corrective_target_names"] == ["SHOULDER_RAISE", "UNDERARM_REACH", "ELBOW_BEND"]
    assert all(value > 0 for value in payload["corrective_affected_vertices"].values())
    assert payload["secondary_motion"] == "NOT_EXECUTED"
    assert payload["rig_aware_lod"] == "NOT_EXECUTED"
    construction = load_json(build / owner / "sleeve_construction_receipt.json")
    bind = load_json(build / owner / "rig_bind_plan.json")
    assert construction["sleeve_cap_and_armhole_admitted"] is True
    assert construction["underarm_seam_published"] is True
    assert bind["upper_arm_forearm_transfer"] is True
    return payload


def _validate_registry(build: Path) -> dict:
    registry = load_json(build / "garment_library_registry.json")
    ids = {item["garment_id"] for item in registry["entries"]}
    assert "SLEEVED_TUNIC_RIGGED_R1B" in ids
    assert "STRAIGHT_SLEEVE_ROBE_RIGGED_R1B" in ids
    accepted = [
        load_json(build / "outfits/sleeved_tunic_two_piece.json"),
        load_json(build / "outfits/straight_robe_two_piece.json"),
    ]
    rejected = load_json(build / "outfits/incompatible_double_upper.json")
    assert all(item["status"] == "ACCEPTED" for item in accepted)
    assert rejected["status"] == "REJECTED_ATOMIC"
    assert rejected["rejection_reasons"]
    return registry


def _validate_runtime(build: Path) -> dict:
    receipt = load_json(build / "cp5_receipt.json")
    runtime = load_json(build / "godot_product/godot_sleeve_runtime_receipt.json")
    assert receipt["cp5_acceptance"] is True
    assert receipt["terminal_decision"] == "CP5_COMPLETE_SLEEVE_RIG_GENERALIZATION"
    assert runtime["consumer_pass"] is True
    assert runtime["sleeved_tunic"]["max_blend_shape_count"] >= 3
    assert runtime["straight_robe"]["max_blend_shape_count"] >= 3
    assert runtime["sleeved_tunic"]["skeleton_count"] >= 1
    assert runtime["straight_robe"]["skeleton_count"] >= 1
    return receipt


def _source_paths(root: Path) -> list[Path]:
    kernel = root / "source/wuxia_garment_oss/rig/sleeve_generalization"
    paths = sorted(kernel.glob("*.py"))
    paths.extend(sorted(path for path in (root / "scripts").glob("*r1b_cp5*.py")))
    paths.extend(sorted(path for path in (root / "tests").glob("*r1b_cp5*.py")))
    return paths


def _source_budget(root: Path) -> dict:
    files_over, functions_over = [], []
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    for path in _source_paths(root):
        source = path.read_text(encoding="utf-8")
        line_count = len(source.splitlines())
        relative = path.relative_to(root).as_posix()
        if line_count > maximum_file[1]:
            maximum_file = (relative, line_count)
        if line_count > 500:
            files_over.append((relative, line_count))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
                if length > maximum_function[2]:
                    maximum_function = (relative, node.name, length)
                if length > 80:
                    functions_over.append((relative, node.name, length))
    assert not files_over, files_over
    assert not functions_over, functions_over
    return {
        "files_checked": len(_source_paths(root)),
        "files_over_500": files_over,
        "functions_over_80": functions_over,
        "maximum_file": maximum_file,
        "maximum_function": maximum_function,
    }


def main() -> int:
    root = parse_args().root.resolve()
    build = root / "build/rig_cp5"
    measurements = _validate_measurement(build)
    tunic = _validate_product(root, build, "sleeved_tunic")
    robe = _validate_product(root, build, "straight_sleeve_robe")
    registry = _validate_registry(build)
    receipt = _validate_runtime(build)
    evidence = build / "cp5_sleeve_generalization_evidence.png"
    with Image.open(evidence) as image:
        width, height = image.size
    assert width >= 2000 and height >= 1400
    summary = {
        "checkpoint": "GARMENT_CAD_PRO_R1B_CP5",
        "cp5_validation": "PASS",
        "products": [tunic["product_id"], robe["product_id"]],
        "registry_entries": len(registry["entries"]),
        "sleeve_length_m": measurements["sleeve_length_m"],
        "corrective_driver_count": 3,
        "godot_version": receipt["godot_version"],
        "evidence": {"path": evidence.relative_to(root).as_posix(), "width": width, "height": height},
        "secondary_motion_executed": False,
        "rig_aware_lod_executed": False,
        "source_budget": _source_budget(root),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
