#!/usr/bin/env python3
"""Validate Blender-free CP4-R2-R1-REV1 products and source budgets."""
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


def source_budget(root: Path) -> dict:
    paths = sorted((root / "source/wuxia_garment_oss/metric_fidelity").glob("*.py"))
    paths += sorted((root / "scripts").glob("*cp4_r2_rev1*.py"))
    paths += sorted((root / "tests").glob("*cp4_r2_rev1*.py"))
    files_over, functions_over = [], []
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    for path in paths:
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        lines = len(source.splitlines())
        if lines > maximum_file[1]:
            maximum_file = (relative, lines)
        if lines > 500:
            files_over.append((relative, lines))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
                if length > maximum_function[2]:
                    maximum_function = (relative, node.name, length)
                if length > 80:
                    functions_over.append((relative, node.name, length))
    if files_over or functions_over:
        raise AssertionError({"files": files_over, "functions": functions_over})
    return {
        "files_checked": len(paths),
        "files_over_500": files_over,
        "functions_over_80": functions_over,
        "maximum_file": maximum_file,
        "maximum_function": maximum_function,
    }


def main() -> int:
    root = parse_args().root.resolve()
    build = root / "build/r1c_cp4_r2_rev1"
    receipt = load_json(build / "cp4_r2_rev1_receipt.json")
    technical = load_json(build / "technical_qualification_receipt.json")
    glb = load_json(build / "direct_glb_package.json")
    validator = load_json(build / "khronos_gltf_validation_report.json")
    godot = load_json(build / "godot_consumer_receipt.json")
    evidence = load_json(build / "godot_visual_evidence_receipt.json")
    assert technical["technical_pass"] is True
    assert glb["fresh_reopen"]["pass"] is True
    assert int(validator["issues"]["numErrors"]) == 0
    assert godot["consumer_pass"] is True
    assert evidence["all_views_reviewable"] is True
    assert receipt["blender_executed"] is False
    assert receipt["blend_generated"] is False
    assert receipt["cp4_r1_predecessor_mutated"] is False
    assert receipt["post_settle_vertex_repair_count"] == 0
    contact = root / evidence["contact_sheet"]["path"]
    with Image.open(contact) as image:
        assert image.size == (3072, 2168)
    result = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP4_R2_R1_REV1",
        "validation": "PASS" if receipt["direct_visual_art_review"] != "PENDING" else "PASS_PENDING_DIRECT_ART_AUDIT",
        "product_acceptance": receipt["product_acceptance"],
        "technical_pass": technical["technical_pass"],
        "validator_errors": int(validator["issues"]["numErrors"]),
        "godot_consumer_pass": godot["consumer_pass"],
        "all_views_reviewable": evidence["all_views_reviewable"],
        "source_budget": source_budget(root),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
