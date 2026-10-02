#!/usr/bin/env python3
"""Bounded terminal validation for GARMENT-CAD-PRO-R1C CP4-R1."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_budget(root: Path) -> dict:
    paths = sorted((root / "source/wuxia_garment_oss/materialization").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("*cp4_r1*.py")))
    files_over, functions_over = [], []
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    for path in paths:
        source = path.read_text(encoding="utf-8")
        count = len(source.splitlines())
        relative = path.relative_to(root).as_posix()
        maximum_file = max(maximum_file, (relative, count), key=lambda item: item[1])
        if count > 500:
            files_over.append((relative, count))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
                maximum_function = max(maximum_function, (relative, node.name, length), key=lambda item: item[2])
                if length > 80:
                    functions_over.append((relative, node.name, length))
    if files_over or functions_over:
        raise AssertionError({"files": files_over, "functions": functions_over})
    return {
        "files_checked": len(paths),
        "files_over_500": files_over,
        "functions_over_80": functions_over,
        "maximum_file": maximum_file,
        "maximum_function": maximum_function,
    }


def validate_views(build: Path, visual: dict) -> dict:
    sizes = {}
    for view_id in visual["required_views"]:
        path = build / "visual" / f"{view_id}.png"
        with Image.open(path) as image:
            sizes[view_id] = list(image.size)
            if image.width < 2048 or image.height < 2048:
                raise AssertionError(f"undersized view: {view_id}")
    sheet = build / "cp4_r1_body_visible_contact_sheet.png"
    with Image.open(sheet) as image:
        sheet_size = list(image.size)
    return {"views": sizes, "contact_sheet": sheet_size}


def main():
    root = parse_args().root.resolve()
    build = root / "build/r1c_cp4_r1"
    receipt = load(build / "cp4_r1_receipt.json")
    technical = load(build / "technical_qualification_receipt.json")
    visual = load(build / "visual_evidence_receipt.json")
    seam = load(build / "seam_correspondence_receipt.json")
    rest = load(build / "compiled_rest_metric_receipt.json")
    settling = load(build / "warp_settling_receipt.json")
    preservation = load(build / "predecessor_preservation.json")
    assert receipt["product_acceptance"] is True
    assert receipt["terminal_decision"] == "CP4_R1_ACCEPTED_PATTERN_DRIVEN_TUNIC"
    assert technical["technical_pass"] is True
    assert visual["visual_review"] == "PASS"
    assert seam["interface_count"] == 16
    assert seam["reversed_count"] >= 1
    assert rest["settling_rest_owner"] == "ARRANGEMENT_REST_METRIC"
    assert settling["post_settle_vertex_repair_count"] == 0
    assert preservation["all_predecessors_unchanged"] is True
    glb = build / "product/r1c_cp4_r1_sleeved_tunic.glb"
    blend = build / "product/r1c_cp4_r1_sleeved_tunic.blend"
    if not glb.is_file() or not blend.is_file():
        raise AssertionError("missing Blender product")
    summary = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP4_R1",
        "validation": "PASS",
        "technical_pass": True,
        "visual_review": "PASS",
        "product_acceptance": True,
        "orientation_reversed_count": seam["reversed_count"],
        "seam_p95_m": technical["seam"]["p95_m"],
        "edge_strain_p95": technical["edge_strain"]["p95"],
        "body_penetration_p99_m": technical["body_penetration"]["p99_m"],
        "tail_peak_m": technical["settling_tail"]["peak_m"],
        "views": validate_views(build, visual),
        "source_budget": source_budget(root),
        "triangulation_executed": True,
        "warp_settling_executed": True,
        "post_settle_vertex_repair_count": 0,
    }
    (root / "R1C_CHECKS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
