#!/usr/bin/env python3
"""Validate R1C CP1 products, evidence, source budgets, and predecessor preservation."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from PIL import Image

from wuxia_garment_oss.proportions.resolution.receipt import verify_embedded_hash


EXPECTED_REJECTIONS = {
    "missing_reference": "MISSING_REFERENCE",
    "quantity_mismatch": "QUANTITY_MISMATCH",
    "expression_quantity_mismatch": "QUANTITY_MISMATCH",
    "unit_mismatch": "UNIT_MISMATCH",
    "dependency_cycle": "CYCLE_DETECTED",
    "non_finite_ratio": "NON_FINITE_RATIO",
    "hard_bound": "BOUND_REJECTED",
    "alternate_component": "ALTERNATE_COMPONENT_REQUIRED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_products(root: Path) -> dict:
    build = root / "build/r1c_cp1"
    receipt = load_json(build / "cp1_receipt.json")
    resolved = load_json(build / "resolved_parameter_set.json")
    reopen = load_json(build / "fresh_process_reopen_receipt.json")
    assert receipt["terminal_decision"] == "CP1_COMPLETE_RATIO_PARAMETER_ENGINE"
    assert receipt["cp1_acceptance"] is True
    assert receipt["parameter_mode_count"] == 3
    assert receipt["reference_scope_count"] == 5
    assert receipt["partial_publication_count"] == 0
    assert receipt["geometry_executed"] is False
    assert receipt["triangulation_executed"] is False
    assert receipt["simulation_executed"] is False
    assert receipt["godot_executed"] is False
    assert verify_embedded_hash(resolved, "resolved_set_sha256")
    assert verify_embedded_hash(reopen, "receipt_sha256")
    assert reopen["fresh_process_reopen_pass"] is True
    assert reopen["deterministic_rerun_pass"] is True
    return {
        "parameter_count": receipt["parameter_count"],
        "clamp_count": receipt["clamp_count"],
        "resolved_set_sha256": receipt["resolved_set_sha256"],
    }


def validate_rejections(root: Path) -> dict:
    target = root / "build/r1c_cp1/rejections"
    result = {}
    for name, expected in EXPECTED_REJECTIONS.items():
        payload = load_json(target / f"{name}.json")
        assert payload["accepted"] is False
        assert payload["status"] == expected
        assert payload["partial_publication_count"] == 0
        assert verify_embedded_hash(payload, "receipt_sha256")
        result[name] = payload["status"]
    return result


def source_budget(root: Path) -> dict:
    paths = sorted((root / "source/wuxia_garment_oss/proportions").rglob("*.py"))
    paths.extend(sorted((root / "source/wuxia_garment_oss/r1c_cp1_fixtures").rglob("*.py")))
    paths.extend(sorted((root / "scripts").glob("*r1c_cp1*.py")))
    paths.extend(sorted((root / "tests").glob("*r1c_cp1*.py")))
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    files_over, functions_over = [], []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        lines = len(source.splitlines())
        if lines > maximum_file[1]:
            maximum_file = (relative, lines)
        if lines > 500:
            files_over.append((relative, lines))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
                if length > maximum_function[2]:
                    maximum_function = (relative, node.name, length)
                if length > 80:
                    functions_over.append((relative, node.name, length))
    assert not files_over, files_over
    assert not functions_over, functions_over
    return {
        "files_checked": len(paths),
        "files_over_500": files_over,
        "functions_over_80": functions_over,
        "maximum_file": maximum_file,
        "maximum_function": maximum_function,
    }


def validate_evidence(root: Path) -> dict:
    path = root / "build/r1c_cp1/cp1_ratio_parameter_evidence.png"
    with Image.open(path) as image:
        width, height = image.size
    assert width >= 2400 and height >= 1600
    return {"path": path.relative_to(root).as_posix(), "width": width, "height": height}


def main() -> int:
    root = parse_args().root.resolve()
    schemas = sorted((root / "contracts/r1c_cp1").glob("*.schema.json"))
    assert len(schemas) == 5
    predecessor = load_json(root / "build/r1c_cp1/predecessor_preservation.json")
    assert predecessor["all_predecessors_unchanged"] is True
    checks = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP1",
        "validation": "PASS",
        "schemas": len(schemas),
        "products": validate_products(root),
        "rejections": validate_rejections(root),
        "evidence": validate_evidence(root),
        "source_budget": source_budget(root),
        "predecessor_preservation": True,
        "geometry_executed": False,
        "simulation_executed": False,
        "godot_executed": False,
    }
    (root / "R1C_CHECKS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
