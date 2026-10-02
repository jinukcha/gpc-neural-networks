#!/usr/bin/env python3
"""Bounded CP6 validation for runtime products, captures, and source budgets."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from PIL import Image


BUILD_REL = Path("build/rig_cp6")
OWNERS = ("sleeved_tunic", "straight_robe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_products(root: Path, build: Path) -> dict:
    results = {}
    for owner in OWNERS:
        base = build / "products" / owner
        package = load_json(base / "runtime_garment_lod_set.json")
        qualification = load_json(base / "lod_qualification.json")
        assert qualification["accepted"] is True
        assert qualification["skin_preserved"] is True
        assert qualification["target_names_preserved"] is True
        assert qualification["active_morph_targets_preserved"] is True
        counts = []
        for lod_id in ("LOD0", "LOD1", "LOD2"):
            path = root / package["variants"][lod_id]
            assert path.is_file() and path.stat().st_size > 1000
            metrics = next(
                item for item in qualification["lod_metrics"]
                if Path(item["path"]).name == f"{lod_id.lower()}.glb"
            )
            counts.append((metrics["vertex_count"], metrics["triangle_count"]))
            assert metrics["bone_count"] == 23
            assert metrics["zero_weight_vertex_count"] == 0
            assert metrics["negative_weight_count"] == 0
            assert metrics["maximum_weight_sum_error"] <= 1.0e-5
        assert counts[0][0] > counts[1][0] > counts[2][0]
        assert counts[0][1] > counts[1][1] > counts[2][1]
        results[owner] = counts
    return results


def _validate_runtime(build: Path) -> dict:
    runtime = load_json(build / "godot_runtime_receipt.json")
    version = runtime["godot_version"]
    assert (version["major"], version["minor"], version["patch"]) == (4, 7, 2)
    assert runtime["runtime_acceptance"] is True
    assert runtime["read_only_consumer"] is True
    assert len(runtime["products"]) == 2
    for product in runtime["products"]:
        assert product["accepted"] is True
        assert product["secondary_motion_bounded"] is True
        assert product["secondary_state_transfer_pass"] is True
        assert product["corrective_and_secondary_names_preserved"] is True
        assert product["skeleton_signature_preserved"] is True
    return runtime


def _validate_captures(root: Path, build: Path) -> dict:
    receipt = load_json(build / "multi_distance_capture_receipt.json")
    assert receipt["capture_count"] == 6
    assert receipt["all_captures_accepted"] is True
    for capture in receipt["captures"]:
        path = root / capture["path"] if not Path(capture["path"]).is_absolute() else Path(capture["path"])
        assert path.is_file() and capture["accepted"] is True
    sheet = build / "cp6_multi_distance_contact_sheet.png"
    with Image.open(sheet) as image:
        assert image.width == 1920 and image.height == 1280
    return receipt


def _python_function_lengths(path: Path) -> list[tuple[str, int]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return [
        (node.name, node.end_lineno - node.lineno + 1)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno
    ]


def _gdscript_function_lengths(path: Path) -> list[tuple[str, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [(index, line.split("func ", 1)[1].split("(", 1)[0]) for index, line in enumerate(lines) if line.startswith("func ")]
    result = []
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        result.append((name, end - start))
    return result


def _source_budget(root: Path) -> dict:
    source_root = root / "source/wuxia_garment_oss/rig/runtime_product"
    paths = sorted(source_root.rglob("*.py"))
    paths += sorted((root / "scripts").glob("*r1b_cp6*.py"))
    paths += sorted((root / "tests").glob("*r1b_cp6*.py"))
    paths += sorted((root / "godot/garment_r1b_cp6").glob("*.gd"))
    file_violations, function_violations = [], []
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        maximum_file = max(maximum_file, (relative, line_count), key=lambda item: item[1])
        if line_count > 500:
            file_violations.append((relative, line_count))
        functions = _gdscript_function_lengths(path) if path.suffix == ".gd" else _python_function_lengths(path)
        for name, length in functions:
            maximum_function = max(maximum_function, (relative, name, length), key=lambda item: item[2])
            if length > 80:
                function_violations.append((relative, name, length))
    assert not file_violations, file_violations
    assert not function_violations, function_violations
    return {
        "files_checked": len(paths),
        "maximum_file": maximum_file,
        "maximum_function": maximum_function,
        "files_over_500": file_violations,
        "functions_over_80": function_violations,
    }


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    products = _validate_products(root, build)
    runtime = _validate_runtime(build)
    captures = _validate_captures(root, build)
    receipt = load_json(build / "cp6_receipt.json")
    assert receipt["cp6_acceptance"] is True
    assert receipt["r1b_complete"] is True
    assert receipt["terminal_decision"] == "GARMENT_CAD_PRO_R1B_COMPLETE"
    assert receipt["cp5_predecessor_mutated"] is False
    summary = {
        "checkpoint": "GARMENT_CAD_PRO_R1B_CP6",
        "validation": "PASS",
        "r1b_complete": True,
        "products": products,
        "godot_product_count": len(runtime["products"]),
        "capture_count": captures["capture_count"],
        "source_budget": _source_budget(root),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
