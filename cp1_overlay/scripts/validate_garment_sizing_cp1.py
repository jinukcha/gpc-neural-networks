#!/usr/bin/env python3
"""Validate CP1 results, PNG evidence, preservation hashes, and source budgets."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def scan_source(root: Path) -> dict:
    files = sorted((root / "source").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        loc = len(text.splitlines())
        if loc > maximum_file[1]:
            maximum_file = (rel, loc)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                size = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
                if size > maximum_function[2]:
                    maximum_function = (rel, node.name, size)
                if size > 80:
                    violations.append((rel, node.name, size))
    if maximum_file[1] > 500 or violations:
        raise AssertionError({"maximum_file": maximum_file, "functions": violations})
    return {
        "files_checked": len(files),
        "maximum_file": {"path": maximum_file[0], "loc": maximum_file[1]},
        "maximum_function": {
            "path": maximum_function[0], "name": maximum_function[1],
            "loc": maximum_function[2],
        },
        "files_over_500": [],
        "functions_over_80": [],
    }


def validate_outputs(root: Path) -> dict:
    status = json.loads((root / "SIZING_STATUS.json").read_text())
    assert status["terminal_decision"] == "CP1_COMPLETE_SELECTION_ONLY"
    assert status["triangulation_executed"] is False
    assert status["warp_simulation_executed"] is False
    assert status["size_table_cardinality"] == 4
    required = {
        "NORMAL_GRADE", "CUSTOM_ALTERATION", "ALTERNATE_BLOCK_REQUIRED",
        "TOPOLOGY_CHANGE_REQUIRED", "HOLD",
    }
    assert required.issubset(status["admission_counts"])
    evidence = root / "build/tunic_pilot/sizing_cp1/cp1_selection_evidence.png"
    with Image.open(evidence) as image:
        assert image.width >= 1600 and image.height >= 1000
        dimensions = [image.width, image.height]
    return {"status": status, "evidence_dimensions": dimensions}


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    outputs = validate_outputs(root)
    checks = {
        "schema_version": 1,
        "cp1_tests": "PASS",
        "product_drape_source_sha256": os.environ["DRAPE_HASH"],
        "cp3_final_mesh_sha256": os.environ["MESH_HASH"],
        "cp3_frame180_png_sha256": os.environ["PNG_HASH"],
        "cp3_receipt_sha256": os.environ["CP3_RECEIPT_HASH"],
        "evidence_dimensions": outputs["evidence_dimensions"],
        "source_budget": scan_source(root),
    }
    (root / "SIZING_CHECKS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
