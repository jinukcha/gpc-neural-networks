#!/usr/bin/env python3
"""Validate CP2 parameter packages, PNG evidence, and preserved CP1 authority."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


EXPECTED_PACKAGES = {
    "STANDARD_S", "STANDARD_M", "STANDARD_L",
    "AUTO_REFERENCE", "AUTO_MILD_CUSTOM", "AUTO_BROAD_SHOULDER",
    "AUTO_FULL_CHEST", "AUTO_FULL_ABDOMEN", "AUTO_TALL", "AUTO_SHORT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*.json")):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def load_packages(root: Path) -> dict[str, dict]:
    directory = root / "build/tunic_pilot/sizing_cp2/pattern_parameter_packages"
    packages = {path.stem: json.loads(path.read_text()) for path in directory.glob("*.json")}
    assert set(packages) == EXPECTED_PACKAGES, sorted(packages)
    return packages


def validate_packages(packages: dict[str, dict]) -> dict:
    for package in packages.values():
        assert package["contract"] == "GarmentPatternParameterPackage/1"
        assert package["triangulation_executed"] is False
        assert package["warp_simulation_executed"] is False
        assert package["mesh_scaling"] == "FORBIDDEN"
        assert len(package["panels"]) == 4
        assert len(package["seam_pairs"]) == 8
    standard = [packages[f"STANDARD_{size}"]["poms"] for size in ("S", "M", "L")]
    for key in (
        "finished_chest_circumference_m", "finished_waist_circumference_m",
        "shoulder_half_m", "skirt_length_m", "hem_half_m",
    ):
        assert standard[0][key] < standard[1][key] < standard[2][key], key
    reference = packages["AUTO_REFERENCE"]["poms"]
    full_chest = packages["AUTO_FULL_CHEST"]["poms"]
    full_abdomen = packages["AUTO_FULL_ABDOMEN"]["poms"]
    broad = packages["AUTO_BROAD_SHOULDER"]["poms"]
    tall = packages["AUTO_TALL"]["poms"]
    short = packages["AUTO_SHORT"]["poms"]
    assert full_chest["front_chest_half_m"] - reference["front_chest_half_m"] > 0.015
    assert abs(full_chest["back_chest_half_m"] - reference["back_chest_half_m"]) < 1.0e-9
    assert full_abdomen["front_waist_half_m"] - reference["front_waist_half_m"] > 0.020
    assert abs(full_abdomen["back_waist_half_m"] - reference["back_waist_half_m"]) < 1.0e-9
    assert broad["shoulder_half_m"] > reference["shoulder_half_m"]
    assert tall["skirt_length_m"] > reference["skirt_length_m"]
    assert short["skirt_length_m"] < reference["skirt_length_m"]
    return {"package_count": len(packages), "standard_sizes": ["S", "M", "L"]}


def source_budget(root: Path) -> dict:
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
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                size = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
                if size > maximum_function[2]:
                    maximum_function = (rel, node.name, size)
                if size > 80:
                    violations.append((rel, node.name, size))
    assert maximum_file[1] <= 500, maximum_file
    assert not violations, violations
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


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    status = json.loads((root / "SIZING_STATUS.json").read_text())
    assert status["terminal_decision"] == "CP2_COMPLETE_PARAMETERS_ONLY"
    assert status["reference_parity_max_error"] <= 1.0e-12
    assert status["triangulation_executed"] is False
    assert status["warp_simulation_executed"] is False
    packages = load_packages(root)
    package_checks = validate_packages(packages)
    blocked = json.loads(
        (root / "build/tunic_pilot/sizing_cp2/blocked_admissions.json").read_text()
    )["blocked"]
    assert {row["admission"] for row in blocked} == {
        "ALTERNATE_BLOCK_REQUIRED", "TOPOLOGY_CHANGE_REQUIRED", "HOLD"
    }
    evidence = root / "build/tunic_pilot/sizing_cp2/cp2_pattern_parameter_evidence.png"
    with Image.open(evidence) as image:
        dimensions = [image.width, image.height]
        assert image.width >= 2000 and image.height >= 1200
    receipt_tree = hash_tree(root / "build/tunic_pilot/sizing_cp1/selection_receipts")
    expected_tree = os.environ["CP1_RECEIPT_TREE_HASH"]
    assert receipt_tree == expected_tree
    checks = {
        "schema_version": 1,
        "cp2_tests": "PASS",
        "package_checks": package_checks,
        "blocked_count": len(blocked),
        "cp1_selection_receipt_tree_sha256": receipt_tree,
        "cp1_receipts_mutated": False,
        "evidence_dimensions": dimensions,
        "product_drape_source_sha256": os.environ["DRAPE_HASH"],
        "cp3_final_mesh_sha256": os.environ["MESH_HASH"],
        "cp3_frame180_png_sha256": os.environ["PNG_HASH"],
        "cp3_receipt_sha256": os.environ["CP3_RECEIPT_HASH"],
        "source_budget": source_budget(root),
    }
    (root / "SIZING_CHECKS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
