#!/usr/bin/env python3
"""Validate R1C CP2 exact geometry, interfaces, assembly, and source budgets."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from PIL import Image

from wuxia_garment_oss.pattern_components.model import canonical_sha256


BUILD_REL = Path("build/r1c_cp2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_valid(payload: dict, key: str) -> bool:
    source = dict(payload)
    recorded = source.pop(key, None)
    return isinstance(recorded, str) and recorded == canonical_sha256(source)


def validate_library(build: Path) -> dict:
    registry = load_json(build / "component_registry.json")
    library = load_json(build / "pattern_geometry_library.json")
    assert registry["component_count"] == 6
    assert library["authority_count"] == 9
    assert hash_valid(library, "library_sha256")
    categories = {item["category"] for item in registry["components"]}
    assert {"BODICE", "SLEEVE", "COLLAR", "CUFF", "GORE"}.issubset(categories)
    instance_ids = {item["instance_id"] for item in library["authorities"]}
    expected = {"bodice_front", "bodice_back", "sleeve_left", "sleeve_right", "collar", "cuff_left", "cuff_right", "gore_left", "gore_right"}
    assert instance_ids == expected
    for authority in library["authorities"]:
        assert hash_valid(authority, "geometry_sha256")
        assert authority["segments"] and authority["boundaries"]
        assert authority["triangulation_executed"] is False
        assert authority["simulation_executed"] is False
    return {"definitions": registry["component_count"], "authorities": library["authority_count"], "categories": sorted(categories)}


def validate_interfaces(build: Path) -> dict:
    receipt = load_json(build / "interface_solver_receipt.json")
    assert receipt["accepted"] is True
    assert receipt["interface_count"] == 16
    assert receipt["accepted_interface_count"] == 16
    cap_ratios, notch_deltas = [], []
    for item in receipt["interface_receipts"]:
        assert item["accepted"] is True
        assert item["length_pass"] is True
        assert item["notch_pass"] is True
        if item["semantic_role"] == "SLEEVE_CAP":
            cap_ratios.append(item["directed_or_symmetric_ratio"])
        notch_deltas.extend(row["delta"] for row in item["notch_correspondence"])
    assert len(cap_ratios) == 4
    assert all(1.0 <= value <= 1.08 for value in cap_ratios)
    assert notch_deltas and max(notch_deltas) <= 0.025
    return {"interfaces": receipt["interface_count"], "cap_ratios": cap_ratios, "maximum_notch_delta": max(notch_deltas)}


def validate_assembly(build: Path) -> dict:
    package = load_json(build / "assembled_pattern_package.json")
    compilation = load_json(build / "assembly_compilation_receipt.json")
    rejected_solver = load_json(build / "rejections/notch_mismatch_interface_receipt.json")
    rejected_compile = load_json(build / "rejections/notch_mismatch_compilation_receipt.json")
    assert hash_valid(package, "assembled_package_sha256")
    assert compilation["accepted"] is True and compilation["status"] == "COMPILED"
    assert package["component_instance_count"] == 7
    assert package["seam_count"] == 16
    assert package["notch_pair_count"] >= 8
    assert package["triangulation_executed"] is False
    assert package["simulation_executed"] is False
    assert rejected_solver["accepted"] is False
    assert any("NOTCH_CORRESPONDENCE_MISMATCH" in error for error in rejected_solver["errors"])
    assert rejected_compile["status"] == "REJECTED_ATOMIC"
    assert rejected_compile["partial_publication_count"] == 0
    return {"instances": package["component_instance_count"], "seams": package["seam_count"], "notch_pairs": package["notch_pair_count"]}


def source_paths(root: Path) -> list[Path]:
    paths = sorted((root / "source/wuxia_garment_oss/pattern_geometry").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("*r1c_cp2.py")))
    paths.extend(sorted((root / "tests").glob("*r1c_cp2.py")))
    return paths


def source_budget(root: Path) -> dict:
    files_over, functions_over = [], []
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    for path in source_paths(root):
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        lines = len(source.splitlines())
        maximum_file = max(maximum_file, (relative, lines), key=lambda item: item[1])
        if lines > 500:
            files_over.append((relative, lines))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
                maximum_function = max(maximum_function, (relative, node.name, length), key=lambda item: item[2])
                if length > 80:
                    functions_over.append((relative, node.name, length))
    assert not files_over, files_over
    assert not functions_over, functions_over
    return {
        "files_checked": len(source_paths(root)),
        "files_over_500": files_over,
        "functions_over_80": functions_over,
        "maximum_file": maximum_file,
        "maximum_function": maximum_function,
    }


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    receipt = load_json(build / "cp2_receipt.json")
    reopen = load_json(build / "fresh_process_reopen_receipt.json")
    predecessor = load_json(build / "predecessor_preservation.json")
    assert receipt["cp2_acceptance"] is True
    assert receipt["terminal_decision"] == "CP2_COMPLETE_PATTERN_COMPONENT_ASSEMBLY"
    assert reopen["fresh_process_reopen_pass"] is True
    assert receipt["deterministic_rebuild_pass"] is True
    assert predecessor["all_predecessors_unchanged"] is True
    assert receipt["triangulation_executed"] is False
    assert receipt["simulation_executed"] is False
    evidence = build / "cp2_exact_component_assembly_evidence.png"
    with Image.open(evidence) as image:
        dimensions = [image.width, image.height]
    assert dimensions[0] >= 2400 and dimensions[1] >= 1600
    schemas = sorted((root / "contracts/r1c_cp2").glob("*.schema.json"))
    assert len(schemas) == 5
    checks = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP2",
        "validation": "PASS",
        "library": validate_library(build),
        "interfaces": validate_interfaces(build),
        "assembly": validate_assembly(build),
        "fresh_process_reopen": True,
        "deterministic_rebuild": True,
        "predecessor_preservation": True,
        "schema_count": len(schemas),
        "evidence": {"path": evidence.relative_to(root).as_posix(), "width": dimensions[0], "height": dimensions[1]},
        "triangulation_executed": False,
        "simulation_executed": False,
        "source_budget": source_budget(root),
    }
    (root / "R1C_CHECKS.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
