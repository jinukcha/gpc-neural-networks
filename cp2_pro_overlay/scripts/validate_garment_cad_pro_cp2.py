#!/usr/bin/env python3
"""Validate CP2 grading, feature graph, evidence, and predecessor preservation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


EXPECTED_SIZES = ("XXS", "XS", "S", "M", "L", "XL", "XXL")
EXPECTED_FEATURES = {"DART", "PLEAT", "GATHER", "GUSSET"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_grading(root: Path) -> dict:
    build = root / "build/pattern_cad_cp2/grading"
    rule_set = load_json(build / "grade_rule_set.json")
    receipt = load_json(build / "grading_receipt.json")
    variants = {path.stem: load_json(path) for path in (build / "variants").glob("*.json")}
    assert tuple(rule_set["ordered_size_ids"]) == EXPECTED_SIZES
    assert tuple(variants) == EXPECTED_SIZES
    assert receipt["arbitrary_n_supported"] is True
    assert receipt["size_count"] == 7
    assert receipt["maximum_notch_fraction_error"] <= 1.0e-12
    assert receipt["maximum_hard_constraint_residual"] <= 1.0e-10
    assert receipt["all_curve_propagation_pass"] is True
    assert receipt["underarm_width_monotonic"] is True
    assert receipt["shoulder_width_monotonic"] is True
    assert receipt["skirt_length_monotonic"] is True
    for variant in variants.values():
        assert variant["resolved"]["hard_constraints_pass"] is True
        assert variant["curve_propagation"]["passed"] is True
        assert len(variant["notches"]) == receipt["notch_count_per_size"]
    return {
        "size_count": len(variants),
        "grade_point_count": receipt["grade_point_count"],
        "notch_count_per_size": receipt["notch_count_per_size"],
    }


def validate_features(root: Path) -> dict:
    build = root / "build/pattern_cad_cp2/features"
    graph = load_json(build / "feature_graph.json")
    compiled = load_json(build / "compiled_feature_graph.json")
    failures = load_json(build / "failure_atomicity_receipt.json")
    mapping = load_json(build / "stable_id_mapping.json")
    feature_types = {item["feature_type"] for item in graph["features"]}
    assert feature_types == EXPECTED_FEATURES
    assert len(compiled["feature_receipts"]) == 4
    assert compiled["base_document_mutated"] is False
    assert compiled["final_resolved"]["hard_constraints_pass"] is True
    assert "gusset_underarm" in compiled["final_document"]["panel_ids"]
    assert len(compiled["final_document"]["metadata"]["pattern_features"]) == 4
    assert mapping["removed"] == []
    assert mapping["preserved_count"] > 0
    assert mapping["added_count"] > 0
    assert failures["probe_count"] == 4
    assert failures["all_rejected"] is True
    assert failures["all_atomic"] is True
    for probe in failures["probes"]:
        assert probe["transaction"]["accepted"] is False
        assert probe["revision_unchanged"] is True
        assert probe["document_sha256_unchanged"] is True
    variants = list((build / "feature_variants").glob("*.json"))
    assert len(variants) == 4
    return {
        "feature_types": sorted(feature_types),
        "preserved_ids": mapping["preserved_count"],
        "added_ids": mapping["added_count"],
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
        maximum_file = max(maximum_file, (rel, loc), key=lambda item: item[1])
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
            maximum_function = max(maximum_function, (rel, node.name, size), key=lambda item: item[2])
            if size > 80:
                violations.append((rel, node.name, size))
    assert maximum_file[1] <= 500, maximum_file
    assert not violations, violations
    return {
        "files_checked": len(files),
        "maximum_file": {"path": maximum_file[0], "loc": maximum_file[1]},
        "maximum_function": {
            "path": maximum_function[0], "name": maximum_function[1], "loc": maximum_function[2]
        },
        "files_over_500": [],
        "functions_over_80": [],
    }


def validate_predecessors(root: Path) -> dict:
    targets = {
        "pattern_cad_cp1": root / "build/pattern_cad_cp1",
        "anthropometry_cp0": root / "build/anthropometry_cp0",
        "tunic_build": root / "build/tunic_pilot",
        "robe_build": root / "build/robe_pilot",
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
    status = load_json(root / "PROFESSIONAL_STATUS.json")
    assert status["terminal_decision"] == "CP2_COMPLETE_GRADING_FEATURE_GRAPH"
    assert status["cp1_predecessor_mutated"] is False
    assert status["triangulation_executed"] is False
    assert status["warp_simulation_executed"] is False
    assert status["mesh_scaling"] == "FORBIDDEN"
    grading = validate_grading(root)
    features = validate_features(root)
    evidence = root / "build/pattern_cad_cp2/cp2_grading_feature_evidence.png"
    with Image.open(evidence) as image:
        dimensions = [image.width, image.height]
        assert image.width >= 2400 and image.height >= 1600
    schema_paths = sorted((root / "contracts/pattern_cad").glob("*.schema.json"))
    assert len(schema_paths) == 4
    checks = {
        "schema_version": 1,
        "cp2_tests": "PASS",
        "grading": grading,
        "features": features,
        "evidence_dimensions": dimensions,
        "professional_schema_count": len(schema_paths),
        "predecessor_hashes": validate_predecessors(root),
        "source_budget": source_budget(root),
    }
    write_path = root / "PROFESSIONAL_CHECKS.json"
    write_path.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
