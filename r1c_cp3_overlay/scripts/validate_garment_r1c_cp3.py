#!/usr/bin/env python3
"""Bounded terminal validation for R1C CP3 completion and repair products."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from PIL import Image


BUILD_REL = Path("build/r1c_cp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_paths(root: Path) -> list[Path]:
    paths = sorted((root / "source/wuxia_garment_oss/completion").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("*r1c_cp3.py")))
    paths.extend(sorted((root / "tests").glob("*r1c_cp3.py")))
    return paths


def _source_budget(root: Path) -> dict:
    files_over, functions_over = [], []
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    for path in _source_paths(root):
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        line_count = len(source.splitlines())
        if line_count > maximum_file[1]:
            maximum_file = (relative, line_count)
        if line_count > 500:
            files_over.append((relative, line_count))
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


def _transaction_checks(build: Path) -> dict:
    expected = {
        "safe_commit": "COMMITTED_SAFE_AUTO",
        "guided_wait": "AWAITING_GUIDED_APPROVAL",
        "topology_hold": "REJECTED_HOLD",
        "forced_rollback": "ROLLED_BACK_ATOMIC",
    }
    result = {}
    for name, status in expected.items():
        payload = read_json(build / "transactions" / f"{name}.json")
        assert payload["status"] == status
        assert payload["partial_publication_count"] == 0
        result[name] = status
    return result


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    receipt = read_json(build / "cp3_receipt.json")
    checks = read_json(root / "build/r1c_cp2/cp2_receipt.json")
    safe_report = read_json(build / "diagnosis/safe_auto.json")
    guided_report = read_json(build / "diagnosis/guided.json")
    hold_report = read_json(build / "diagnosis/hold.json")
    repaired = read_json(build / "repaired/repaired_pattern_snapshot.json")
    canonical = read_json(build / "canonical_snapshot.json")
    assert receipt["cp3_acceptance"] is True
    assert receipt["terminal_decision"] == "CP3_COMPLETE_COMPLETION_REPAIR_TRANSACTION"
    assert receipt["partial_publication_count"] == 0
    assert receipt["fresh_process_reopen_pass"] is True
    assert receipt["deterministic_rerun_pass"] is True
    assert repaired["state_sha256"] == canonical["state_sha256"]
    assert safe_report["strongest_disposition"] == "SAFE_AUTO"
    assert guided_report["strongest_disposition"] == "GUIDED"
    assert hold_report["strongest_disposition"] == "HOLD"
    assert checks["cp2_acceptance"] is True
    evidence = build / "cp3_completion_repair_evidence.png"
    with Image.open(evidence) as image:
        width, height = image.size
    assert width >= 2400 and height >= 1680
    summary = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP3",
        "validation": "PASS",
        "issues": {
            "safe": safe_report["issue_count"],
            "guided": guided_report["issue_count"],
            "hold": hold_report["issue_count"],
        },
        "transactions": _transaction_checks(build),
        "safe_repaired_matches_canonical": True,
        "partial_publication_count": 0,
        "triangulation_executed": False,
        "simulation_executed": False,
        "schema_count": len(list((root / "contracts/r1c_cp3").glob("*.schema.json"))),
        "evidence": {
            "path": evidence.relative_to(root).as_posix(),
            "width": width,
            "height": height,
        },
        "predecessor_preservation": read_json(build / "predecessor_preservation.json")["all_predecessors_unchanged"],
        "source_budget": _source_budget(root),
    }
    assert summary["schema_count"] == 5
    assert summary["predecessor_preservation"] is True
    (root / "R1C_CHECKS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
