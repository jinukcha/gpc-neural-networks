#!/usr/bin/env python3
"""Validate CP4 registry, outfit, occlusion, runtime, and source budgets."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_products(root: Path) -> dict:
    build = root / "build/rig_cp4"
    registry = load_json(build / "garment_library_registry.json")
    reference = load_json(build / "outfits/reference_two_piece.json")
    rejected = load_json(build / "outfits/incompatible_duplicate_tunic.json")
    mask = load_json(build / "body_occlusion/body_hide_mask.json")
    runtime = load_json(build / "godot_product/godot_outfit_runtime_receipt.json")
    receipt = load_json(build / "cp4_receipt.json")
    assert registry["contract"] == "GarmentLibraryRegistry/1"
    assert len(registry["entries"]) == 3
    assert reference["status"] == "ACCEPTED"
    assert len(reference["garment_ids"]) == 2
    assert rejected["status"] == "REJECTED_ATOMIC"
    assert rejected["rejection_reasons"]
    assert mask["mask_pass"] is True
    assert runtime["consumer_pass"] is True
    assert receipt["cp4_acceptance"] is True
    assert receipt["terminal_decision"] == "CP4_COMPLETE_MULTI_GARMENT_OUTFIT"
    with np.load(build / "body_occlusion/body_hide_mask.npz", allow_pickle=False) as data:
        hide = np.asarray(data["hide_triangle_mask"], dtype=np.bool_)
        triangles = np.asarray(data["triangles"], dtype=np.int32)
    assert len(hide) == len(triangles)
    assert int(np.count_nonzero(hide)) == mask["hidden_triangle_count"]
    return {
        "registry_entries": len(registry["entries"]),
        "compatible_garments": len(reference["garment_ids"]),
        "rejection_reason_count": len(rejected["rejection_reasons"]),
        "hidden_triangles": mask["hidden_triangle_count"],
        "visible_triangles": mask["visible_triangle_count"],
        "godot_equipped_garments": runtime["equipped_garment_count"],
    }


def validate_evidence(root: Path) -> dict:
    path = root / "build/rig_cp4/cp4_outfit_layering_evidence.png"
    with Image.open(path) as image:
        width, height = image.size
    assert width >= 2400 and height >= 1600
    return {"path": path.relative_to(root).as_posix(), "width": width, "height": height}


def source_budget(root: Path) -> dict:
    files = sorted((root / "source").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        loc = len(text.splitlines())
        if loc > maximum_file[1]:
            maximum_file = (relative, loc)
        if loc > 500:
            violations.append(("file", relative, loc))
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
            if size > maximum_function[2]:
                maximum_function = (relative, node.name, size)
            if size > 80:
                violations.append(("function", relative, node.name, size))
    assert not violations, violations
    return {
        "files_checked": len(files),
        "maximum_file": list(maximum_file),
        "maximum_function": list(maximum_function),
        "files_over_500": [],
        "functions_over_80": [],
    }


def main() -> int:
    root = parse_args().root.resolve()
    schema_paths = sorted((root / "contracts/rig/outfit").glob("*.schema.json"))
    assert len(schema_paths) == 5
    checks = {
        "checkpoint": "GARMENT_CAD_PRO_R1B_CP4",
        "cp4_validation": "PASS",
        "products": validate_products(root),
        "evidence": validate_evidence(root),
        "schema_count": len(schema_paths),
        "source_budget": source_budget(root),
        "secondary_motion_executed": False,
        "rig_aware_lod_executed": False,
    }
    (root / "R1B_CHECKS.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
