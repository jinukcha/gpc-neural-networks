#!/usr/bin/env python3
"""Validate R1C CP0 outputs, contracts, evidence, and source budgets."""
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


def _source_paths(root: Path) -> list[Path]:
    paths = []
    for relative in (
        "source/wuxia_garment_oss/pattern_components",
        "source/wuxia_garment_oss/pattern_assembly",
        "source/wuxia_garment_oss/visual_acceptance",
        "source/wuxia_garment_oss/r1c_cp0_fixtures",
    ):
        paths.extend(sorted((root / relative).glob("*.py")))
    paths.extend(sorted((root / "source/wuxia_garment_oss").glob("r1c_cp0_*.py")))
    paths.extend(sorted((root / "scripts").glob("*r1c_cp0*.py")))
    paths.extend(sorted((root / "tests").glob("*r1c_cp0*.py")))
    return paths


def _source_budget(root: Path) -> dict:
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    files_over = []
    functions_over = []
    for path in _source_paths(root):
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
    build = root / "build/r1c_cp0"
    receipt = load_json(build / "cp0_receipt.json")
    registry = load_json(build / "component_registry.json")
    assembly = load_json(build / "receipts/assembly_admission.json")
    broken = load_json(build / "receipts/incomplete_recipe_rejection.json")
    profile = load_json(build / "visual_acceptance_profile.json")
    clean = load_json(build / "receipts/clean_visual_review.json")
    rejected = load_json(build / "receipts/rejected_visual_review.json")
    assert receipt["cp0_acceptance"] is True
    assert receipt["terminal_decision"] == "CP0_COMPLETE_VISUAL_AND_MODULAR_FOUNDATION"
    assert registry["component_count"] == 3
    assert all(item["geometry_authority"] == "EXACT_2D_PATTERN" for item in registry["components"])
    assert assembly["accepted"] is True
    assert broken["accepted"] is False
    assert clean["product_acceptance"] is True
    assert rejected["product_acceptance"] is False
    assert profile["acceptance_expression"] == "TECHNICAL_PASS_AND_VISUAL_PASS"
    assert len(profile["required_views"]) >= 30
    assert len(profile["gates"]) >= 16
    schema_paths = sorted((root / "contracts/r1c_cp0").glob("*.schema.json"))
    assert len(schema_paths) == 6
    evidence = build / "cp0_modular_visual_authority_evidence.png"
    with Image.open(evidence) as image:
        width, height = image.size
    assert width >= 2400 and height >= 1600
    summary = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP0",
        "validation": "PASS",
        "component_count": registry["component_count"],
        "interface_count": assembly["interface_count"],
        "required_view_count": len(profile["required_views"]),
        "visual_gate_count": len(profile["gates"]),
        "schema_count": len(schema_paths),
        "evidence": {"width": width, "height": height, "path": evidence.relative_to(root).as_posix()},
        "geometry_executed": False,
        "simulation_executed": False,
        "godot_executed": False,
        "source_budget": _source_budget(root),
    }
    (root / "R1C_CHECKS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
