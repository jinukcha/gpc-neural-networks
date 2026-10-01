#!/usr/bin/env python3
"""Validate CP3 construction outputs, predecessor preservation, and source budget."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image


EXPECTED = {
    "seam_spec_count": 12,
    "edge_finish_count": 8,
    "notch_pair_count": 12,
    "closure_count": 1,
    "facing_count": 2,
    "layer_piece_count": 7,
    "turn_of_cloth_count": 3,
    "assembly_operation_count": 21,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def validate_package(root: Path) -> dict:
    build = root / "build/construction_cp3"
    receipt = load_json(build / "cp3_receipt.json")
    package = load_json(build / "construction_package.json")
    failures = load_json(build / "failure_atomicity_receipt.json")
    assert receipt["terminal_decision"] == "CP3_COMPLETE_CONSTRUCTION_GRAPH"
    for key, expected in EXPECTED.items():
        assert receipt[key] == expected, (key, receipt[key], expected)
    assert receipt["assembly_cycle_free"] is True
    assert receipt["failure_atomicity_pass"] is True
    assert receipt["triangulation_executed"] is False
    assert receipt["warp_simulation_executed"] is False
    assert package["source_document_mutated"] is False
    assert failures["all_rejected"] is True and failures["all_atomic"] is True
    gathered = [row for row in package["seam_specs"] if row["seam_type"] == "GATHERED_SEAM"]
    assert len(gathered) == 1 and gathered[0]["gather_ratio"] == 1.12
    return {"receipt": receipt, "package": package, "failures": failures}


def validate_line_offsets(package: dict) -> dict:
    errors = []
    for row in package["seam_lines"]:
        for side in (row["side_a"], row["side_b"]):
            distances = [math.dist(a, b) for a, b in zip(side["stitch_line"], side["cut_line"])]
            errors.extend(abs(value - side["allowance_m"]) for value in distances)
    return {
        "maximum_allowance_offset_error_m": max(errors, default=0.0),
        "line_count": len(package["seam_lines"]) * 2 + len(package["edge_finishes"]),
    }


def source_budget(root: Path) -> dict:
    files = sorted((root / "source").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    long_functions = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        loc = len(text.splitlines())
        maximum_file = max(maximum_file, (rel, loc), key=lambda item: item[1])
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
            maximum_function = max(maximum_function, (rel, node.name, size), key=lambda item: item[2])
            if size > 80:
                long_functions.append((rel, node.name, size))
    assert maximum_file[1] <= 500, maximum_file
    assert not long_functions, long_functions
    return {
        "files_checked": len(files),
        "maximum_file": {"path": maximum_file[0], "loc": maximum_file[1]},
        "maximum_function": {"path": maximum_function[0], "name": maximum_function[1], "loc": maximum_function[2]},
        "files_over_500": [],
        "functions_over_80": [],
    }


def predecessor_hashes(root: Path) -> dict:
    targets = {
        "pattern_cad_cp1": root / "build/pattern_cad_cp1",
        "pattern_cad_cp2": root / "build/pattern_cad_cp2",
        "tunic_build": root / "build/tunic_pilot",
        "pattern_cad_source": root / "source/wuxia_garment_oss/pattern_cad",
        "tunic_pattern_cad_source": root / "source/wuxia_garment_oss/garments/sleeveless_tunic/pattern_cad",
        "sizing_source": root / "source/wuxia_garment_oss/sizing",
        "drape_source": root / "source/wuxia_garment_oss/drape",
    }
    result = {}
    for name, path in targets.items():
        actual = hash_tree(path)
        expected = os.environ[f"EXPECTED_{name.upper()}_HASH"]
        assert actual == expected, (name, actual, expected)
        result[name] = actual
    return result


def main() -> int:
    root = parse_args().root.resolve()
    validated = validate_package(root)
    evidence = root / "build/construction_cp3/cp3_construction_evidence.png"
    with Image.open(evidence) as image:
        dimensions = [image.width, image.height]
        assert image.width >= 2400 and image.height >= 1600
    schema_paths = sorted((root / "contracts/construction").glob("*.schema.json"))
    assert len(schema_paths) == 4
    line_metrics = validate_line_offsets(validated["package"])
    assert line_metrics["maximum_allowance_offset_error_m"] <= 1.0e-9
    checks = {
        "schema_version": 1,
        "cp3_tests": "PASS",
        "construction_counts": EXPECTED,
        "line_metrics": line_metrics,
        "evidence_dimensions": dimensions,
        "construction_schema_count": len(schema_paths),
        "predecessor_hashes": predecessor_hashes(root),
        "source_budget": source_budget(root),
    }
    (root / "PROFESSIONAL_CHECKS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
