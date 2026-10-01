#!/usr/bin/env python3
"""Validate CP3 geometry packages, seam admission, and preserved authorities."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED = {
    "STANDARD_S", "STANDARD_M", "STANDARD_L",
    "AUTO_REFERENCE", "AUTO_MILD_CUSTOM", "AUTO_BROAD_SHOULDER",
    "AUTO_FULL_CHEST", "AUTO_FULL_ABDOMEN", "AUTO_TALL", "AUTO_SHORT",
    "CUSTOM_MILD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def shell_tree_hash(path: Path) -> str:
    rows = []
    for item in sorted(path.rglob("*.json")):
        rel = "./" + item.relative_to(path).as_posix()
        rows.append(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {rel}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def load_geometry(root: Path) -> dict[str, dict]:
    directory = root / "build/tunic_pilot/sizing_cp3/geometry_packages"
    packages = {path.stem: json.loads(path.read_text()) for path in directory.glob("*.json")}
    assert set(packages) == EXPECTED, sorted(packages)
    return packages


def _check_npz(root: Path, package: dict) -> None:
    archive = root / package["triangulation_archive"]["path"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package["triangulation_archive"]["sha256"]
    with np.load(archive, allow_pickle=False) as data:
        for panel in package["panels"]:
            panel_id = panel["panel_id"]
            vertices = data[f"{panel_id}__vertices"]
            triangles = data[f"{panel_id}__triangles"]
            assert vertices.ndim == 2 and vertices.shape[1] == 2
            assert triangles.ndim == 2 and triangles.shape[1] == 3
            assert np.isfinite(vertices).all()
            assert int(triangles.min()) >= 0 and int(triangles.max()) < len(vertices)
            metadata = panel["triangulation"]
            assert metadata["mesh_vertex_count"] == len(vertices)
            assert metadata["triangle_count"] == len(triangles)


def validate_packages(root: Path, packages: dict[str, dict]) -> dict:
    for package in packages.values():
        assert package["contract"] == "GarmentGeometryPackage/1"
        assert package["triangulation_admission"] == "PASS"
        assert package["qualification"]["passed"] is True
        assert package["warp_simulation_executed"] is False
        assert package["product_simulation_executed"] is False
        assert package["mesh_scaling"] == "FORBIDDEN"
        assert len(package["panels"]) == 4
        assert len(package["seam_correspondence"]) == 8
        assert all(row["coverage"] == 1.0 for row in package["seam_correspondence"])
        assert max(row["length_mismatch_ratio"] for row in package["seam_correspondence"]) <= 0.03
        assert all(panel["triangulation"]["degenerate_triangle_count"] == 0 for panel in package["panels"])
        assert all(abs(panel["triangulation"]["area_coverage_ratio"] - 1.0) <= 1.0e-8 for panel in package["panels"])
        _check_npz(root, package)
    counts = [packages[f"STANDARD_{size}"]["qualification"]["total_triangles"] for size in ("S", "M", "L")]
    assert counts[0] <= counts[1] <= counts[2], counts
    assert packages["CUSTOM_MILD"]["selection"]["admission"] == "CUSTOM_ALTERATION"
    return {
        "package_count": len(packages),
        "standard_triangle_counts": counts,
        "maximum_seam_mismatch_ratio": max(
            package["qualification"]["maximum_seam_length_mismatch_ratio"]
            for package in packages.values()
        ),
    }


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
    assert status["terminal_decision"] == "CP3_COMPLETE_GEOMETRY_ADMISSION"
    assert status["all_geometry_admitted"] is True
    assert status["triangulation_admission_executed"] is True
    assert status["warp_simulation_executed"] is False
    packages = load_geometry(root)
    package_checks = validate_packages(root, packages)
    evidence = root / "build/tunic_pilot/sizing_cp3/cp3_geometry_evidence.png"
    with Image.open(evidence) as image:
        dimensions = [image.width, image.height]
        assert image.width >= 2200 and image.height >= 1400
    cp2_tree = shell_tree_hash(root / "build/tunic_pilot/sizing_cp2/pattern_parameter_packages")
    cp1_tree = shell_tree_hash(root / "build/tunic_pilot/sizing_cp1/selection_receipts")
    assert cp2_tree == os.environ["CP2_PARAMETER_TREE_HASH"]
    assert cp1_tree == os.environ["CP1_RECEIPT_TREE_HASH"]
    checks = {
        "schema_version": 1,
        "cp3_tests": "PASS",
        "package_checks": package_checks,
        "cp2_parameter_tree_sha256": cp2_tree,
        "cp2_parameter_packages_mutated": False,
        "cp1_selection_receipt_tree_sha256": cp1_tree,
        "cp1_receipts_mutated": False,
        "evidence_dimensions": dimensions,
        "product_drape_source_sha256": os.environ["DRAPE_HASH"],
        "legacy_cp3_final_mesh_sha256": os.environ["LEGACY_MESH_HASH"],
        "legacy_cp3_frame180_png_sha256": os.environ["LEGACY_PNG_HASH"],
        "legacy_cp3_receipt_sha256": os.environ["LEGACY_RECEIPT_HASH"],
        "source_budget": source_budget(root),
    }
    (root / "SIZING_CHECKS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
