#!/usr/bin/env python3
"""Validate CP4 material calibration, parity, resume, and predecessor preservation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


EXPECTED_MATERIALS = {
    "LINEN_LIGHT_REFERENCE",
    "WOOL_TWILL_MEDIUM_REFERENCE",
    "COTTON_CANVAS_HEAVY_REFERENCE",
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


def validate_material_outputs(root: Path) -> dict:
    build = root / "build/material_calibration_cp4"
    measurement_paths = sorted((build / "measurements").glob("*.json"))
    assert {item.stem for item in measurement_paths} == EXPECTED_MATERIALS
    experiment_count = 0
    for path in measurement_paths:
        payload = load_json(path)
        assert payload["contract"] == "MaterialMeasurementSet/1"
        assert payload["provenance"]["source_kind"] == "PROJECT_REFERENCE_LAB_SERIES"
        assert payload["provenance"]["certification_status"] == "NOT_EXTERNAL_LAB_CERTIFIED"
        assert len(payload["series"]) == 10
        experiment_count += len(payload["series"])
    profile_dirs = sorted(item for item in (build / "profiles").iterdir() if item.is_dir())
    assert {item.name for item in profile_dirs} == EXPECTED_MATERIALS
    calibration_rmse = []
    parity_relative = []
    meshing_edges = {}
    for directory in profile_dirs:
        sizing = load_json(directory / "material_sizing_profile.json")
        warp = load_json(directory / "warp_material_profile.json")
        calibration = load_json(directory / "calibration_receipt.json")
        parity = load_json(directory / "backend_parity_receipt.json")
        assert sizing["contract"] == "MaterialSizingProfile/2"
        assert warp["contract"] == "WarpMaterialProfile/1"
        assert calibration["calibration_pass"] is True
        assert parity["parity_pass"] is True
        assert parity["runtime"]["version"] == "1.17.0"
        assert parity["runtime"]["device"] == "cpu"
        assert parity["input_identity_match"] is True
        assert parity["non_finite_count"] == 0
        assert parity["topology_mutation_count"] == 0
        calibration_rmse.append(calibration["maximum_normalized_rmse"])
        parity_relative.append(parity["maximum_relative_response_error"])
        meshing_edges[directory.name] = sizing["recommended_meshing_edge_m"]
    assert meshing_edges["LINEN_LIGHT_REFERENCE"] < meshing_edges["WOOL_TWILL_MEDIUM_REFERENCE"]
    assert meshing_edges["WOOL_TWILL_MEDIUM_REFERENCE"] < meshing_edges["COTTON_CANVAS_HEAVY_REFERENCE"]
    return {
        "material_count": len(profile_dirs),
        "experiment_series_count": experiment_count,
        "maximum_normalized_rmse": max(calibration_rmse),
        "maximum_relative_parity_error": max(parity_relative),
        "meshing_edges_m": meshing_edges,
    }


def validate_resume_and_evidence(root: Path) -> dict:
    build = root / "build/material_calibration_cp4"
    resume = load_json(build / "fresh_process_resume_receipt.json")
    assert resume["fresh_process_resume_pass"] is True
    assert resume["measurement_identity_pass"] is True
    assert resume["checkpoint_writer_process_id"] != resume["resume_process_id"]
    checkpoint = load_json(build / "calibration_checkpoint.json")
    assert checkpoint["phase"] == "COMPLETE_3_OF_3"
    assert len(checkpoint["completed_material_ids"]) == 3
    evidence = build / "cp4_material_calibration_evidence.png"
    with Image.open(evidence) as image:
        dimensions = [image.width, image.height]
        assert image.width >= 2400 and image.height >= 1600
    return {
        "fresh_process_resume": True,
        "checkpoint_phase": checkpoint["phase"],
        "evidence_dimensions": dimensions,
    }


def validate_predecessors(root: Path) -> dict:
    targets = {
        "pattern_cad_cp1": root / "build/pattern_cad_cp1",
        "pattern_cad_cp2": root / "build/pattern_cad_cp2",
        "construction_cp3": root / "build/construction_cp3",
        "tunic_build": root / "build/tunic_pilot",
        "pattern_cad_source": root / "source/wuxia_garment_oss/pattern_cad",
        "construction_source": root / "source/wuxia_garment_oss/construction",
        "tunic_pattern_cad_source": root / "source/wuxia_garment_oss/garments/sleeveless_tunic/pattern_cad",
        "tunic_construction_source": root / "source/wuxia_garment_oss/garments/sleeveless_tunic/construction",
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


def source_budget(root: Path) -> dict:
    files = sorted((root / "source").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    maximum_file = ("", 0)
    maximum_function = ("", "", 0)
    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        loc = len(text.splitlines())
        maximum_file = max(maximum_file, (relative, loc), key=lambda item: item[1])
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
            maximum_function = max(maximum_function, (relative, node.name, size), key=lambda item: item[2])
            if size > 80:
                violations.append((relative, node.name, size))
    assert maximum_file[1] <= 500, maximum_file
    assert not violations, violations
    return {
        "files_checked": len(files),
        "maximum_file": {"path": maximum_file[0], "loc": maximum_file[1]},
        "maximum_function": {"path": maximum_function[0], "name": maximum_function[1], "loc": maximum_function[2]},
        "files_over_500": [],
        "functions_over_80": [],
    }


def main() -> int:
    root = parse_args().root.resolve()
    status = load_json(root / "PROFESSIONAL_STATUS.json")
    assert status["terminal_decision"] == "CP4_COMPLETE_MATERIAL_CALIBRATION_PARITY"
    assert status["all_calibration_pass"] is True
    assert status["all_backend_parity_pass"] is True
    assert status["fresh_process_resume_pass"] is True
    assert status["cp3_predecessor_mutated"] is False
    assert status["triangulation_executed"] is False
    assert status["motion_fit_executed"] is False
    assert status["mesh_scaling"] == "FORBIDDEN"
    schema_paths = sorted((root / "contracts/materials").glob("*.schema.json"))
    assert len(schema_paths) == 5
    checks = {
        "schema_version": 1,
        "cp4_tests": "PASS",
        "materials": validate_material_outputs(root),
        "resume": validate_resume_and_evidence(root),
        "material_schema_count": len(schema_paths),
        "predecessor_hashes": validate_predecessors(root),
        "source_budget": source_budget(root),
    }
    (root / "PROFESSIONAL_CHECKS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
