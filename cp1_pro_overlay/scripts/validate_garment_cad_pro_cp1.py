#!/usr/bin/env python3
"""Validate CP1 PatternDocument outputs and immutable predecessor owners."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_budget(root: Path) -> dict:
    files = sorted((root / "source").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    file_violations = []
    function_violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        loc = len(text.splitlines())
        if loc > maximum_file[1]:
            maximum_file = (rel, loc)
        if loc > 500:
            file_violations.append((rel, loc))
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                size = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
                if size > maximum_function[2]:
                    maximum_function = (rel, node.name, size)
                if size > 80:
                    function_violations.append((rel, node.name, size))
    assert not file_violations, file_violations
    assert not function_violations, function_violations
    return {
        "files_checked": len(files),
        "maximum_file": {"path": maximum_file[0], "loc": maximum_file[1]},
        "maximum_function": {
            "path": maximum_function[0],
            "name": maximum_function[1],
            "loc": maximum_function[2],
        },
        "files_over_500": [],
        "functions_over_80": [],
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    build = root / "build/pattern_cad_cp1"
    status = json.loads((root / "PROFESSIONAL_STATUS.json").read_text())
    assert status["terminal_decision"] == "CP1_COMPLETE_PATTERN_DOCUMENT"
    assert status["reference_parity_max_error"] <= 1.0e-12
    assert status["accepted_edit"] is True
    assert status["rejected_edit_atomic"] is True
    assert status["undo_pass"] is True and status["redo_pass"] is True
    assert status["fresh_reopen_pass"] is True
    assert status["triangulation_executed"] is False
    assert status["warp_simulation_executed"] is False

    document = json.loads((build / "pattern_document.json").read_text())
    resolved = json.loads((build / "resolved_pattern_document.json").read_text())
    transactions = json.loads((build / "transaction_receipt.json").read_text())["transactions"]
    assert document["contract"] == "PatternDocument/1"
    assert document["revision"] == 1
    assert resolved["hard_constraints_pass"] is True
    assert len(document["panel_ids"]) == 4
    assert len(document["curves"]) == 24
    assert len(transactions) == 4
    rejected = next(row for row in transactions if row["operation"] == "SET_EXPRESSION")
    assert rejected["accepted"] is False
    assert rejected["before_revision"] == rejected["after_revision"]
    assert rejected["before_sha256"] == rejected["after_sha256"]

    evidence = build / "cp1_pattern_document_evidence.png"
    with Image.open(evidence) as image:
        dimensions = [image.width, image.height]
        assert image.width >= 2400 and image.height >= 1500

    schema_paths = sorted((root / "contracts/professional").glob("*.schema.json"))
    assert len(schema_paths) == 3
    assert os.environ["BASE_SIZING_HASH"] == os.environ["AFTER_SIZING_HASH"]
    assert os.environ["BASE_DRAPE_HASH"] == os.environ["AFTER_DRAPE_HASH"]
    assert os.environ["BASE_TUNIC_BUILD_HASH"] == os.environ["AFTER_TUNIC_BUILD_HASH"]
    checks = {
        "schema_version": 1,
        "cp1_tests": "PASS",
        "reference_parity_max_error": status["reference_parity_max_error"],
        "atomic_failure_preserved": True,
        "undo_redo_pass": True,
        "fresh_reopen_pass": True,
        "document_sha256": document["document_sha256"],
        "evidence_sha256": file_sha256(evidence),
        "evidence_dimensions": dimensions,
        "predecessor": {
            "sizing_source_sha256": os.environ["AFTER_SIZING_HASH"],
            "drape_source_sha256": os.environ["AFTER_DRAPE_HASH"],
            "tunic_build_sha256": os.environ["AFTER_TUNIC_BUILD_HASH"],
            "mutation_count": 0,
        },
        "byte_exact_cp0_predecessor": False,
        "cp0_recovery_basis": "CP3 sizing authority",
        "source_budget": source_budget(root),
    }
    (root / "PROFESSIONAL_CHECKS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
