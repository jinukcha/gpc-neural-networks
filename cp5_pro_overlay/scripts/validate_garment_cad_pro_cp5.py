#!/usr/bin/env python3
"""Validate CP5 motion-fit products, canonical maps, and immutable predecessors."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


MATERIAL_IDS = {
    "LINEN_LIGHT_REFERENCE",
    "WOOL_TWILL_MEDIUM_REFERENCE",
    "COTTON_CANVAS_HEAVY_REFERENCE",
}
POSE_IDS = {
    "NEUTRAL_A",
    "ARMS_FORWARD",
    "ARMS_OVERHEAD",
    "CROSS_BODY_REACH",
    "DEEP_ELBOW_BEND",
    "TORSO_TWIST",
    "FORWARD_BEND",
    "SEATED",
    "SQUAT",
    "WALK_STRIDE",
}
MAP_KEYS = {
    "positions",
    "target_positions",
    "drive_weights",
    "frame_max_displacement_m",
    "stress_n_m",
    "strain_ratio",
    "pressure_pa",
    "clearance_m",
    "seam_tension_n_m",
    "contact_persistence",
    "mobility_restriction",
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


def validate_scenarios(root: Path, vertex_count: int) -> dict:
    base = root / "build/motion_fit_cp5"
    receipts = sorted((base / "pose_receipts").glob("*/*.json"))
    archives = sorted((base / "scenarios").glob("*/*.npz"))
    packages = sorted((base / "scenarios").glob("*/*.json"))
    assert len(receipts) == len(archives) == len(packages) == 30
    identities = set()
    pass_count = 0
    nonneutral_motion = 0
    maximum_strain = 0.0
    maximum_pressure = 0.0
    for receipt_path, archive_path, package_path in zip(receipts, archives, packages):
        receipt = load_json(receipt_path)
        package = load_json(package_path)
        identity = (receipt["material_id"], receipt["pose_id"])
        identities.add(identity)
        pass_count += bool(receipt["pose_pass"])
        assert package["contract"] == "MotionFitMapPackage/1"
        assert package["material_id"] == identity[0]
        assert package["pose_id"] == identity[1]
        with np.load(archive_path, allow_pickle=False) as data:
            assert set(data.files) == MAP_KEYS
            assert data["positions"].shape == (vertex_count, 3)
            assert data["target_positions"].shape == (vertex_count, 3)
            for name in MAP_KEYS - {"positions", "target_positions", "frame_max_displacement_m"}:
                assert data[name].shape == (vertex_count,)
                assert np.isfinite(data[name]).all(), (identity, name)
            frame_motion = data["frame_max_displacement_m"]
            assert frame_motion.shape == (12,)
            maximum_strain = max(maximum_strain, float(np.max(data["strain_ratio"])))
            maximum_pressure = max(maximum_pressure, float(np.max(data["pressure_pa"])))
            if identity[1] != "NEUTRAL_A":
                assert float(np.max(frame_motion)) > 0.0, identity
                nonneutral_motion += 1
    expected = {(material, pose) for material in MATERIAL_IDS for pose in POSE_IDS}
    assert identities == expected
    assert nonneutral_motion == 27
    return {
        "scenario_count": len(archives),
        "pose_pass_count": pass_count,
        "nonneutral_motion_scenario_count": nonneutral_motion,
        "maximum_vertex_strain": maximum_strain,
        "maximum_pressure_pa": maximum_pressure,
    }


def validate_suite_and_evidence(root: Path) -> dict:
    base = root / "build/motion_fit_cp5"
    suite = load_json(base / "motion_fit_suite.json")
    receipt = load_json(base / "fit_qualification_receipt.json")
    assert suite["contract"] == "MotionFitSuite/1"
    assert len(suite["poses"]) == 10
    assert {item["pose_id"] for item in suite["poses"]} == POSE_IDS
    assert receipt["contract"] == "FitQualificationReceipt/2"
    assert receipt["scenario_count"] == 30
    assert receipt["scenario_identity_complete"] is True
    dimensions = {}
    for filename, minimum in (
        ("cp5_motion_fit_overview.png", (3000, 1800)),
        ("cp5_fit_maps_evidence.png", (3000, 2100)),
        ("cp5_material_comparison.png", (2880, 1680)),
    ):
        with Image.open(base / filename) as image:
            dimensions[filename] = [image.width, image.height]
            assert image.width >= minimum[0] and image.height >= minimum[1]
    return {
        "motion_fit_pass": receipt["motion_fit_pass"],
        "failed_scenarios": receipt["failed_scenarios"],
        "evidence_dimensions": dimensions,
    }


def validate_predecessors(root: Path) -> dict:
    targets = {
        "pattern_cad_cp1": root / "build/pattern_cad_cp1",
        "pattern_cad_cp2": root / "build/pattern_cad_cp2",
        "construction_cp3": root / "build/construction_cp3",
        "material_cp4": root / "build/material_calibration_cp4",
        "tunic_build": root / "build/tunic_pilot",
        "pattern_cad_source": root / "source/wuxia_garment_oss/pattern_cad",
        "construction_source": root / "source/wuxia_garment_oss/construction",
        "materials_source": root / "source/wuxia_garment_oss/materials",
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
    assert status["checkpoint"] == "GARMENT_CAD_PRO_R1A_CP5"
    assert status["scenario_count"] == 30
    assert status["product_acceptance"] is False
    assert status["cp4_predecessor_mutated"] is False
    assert status["pattern_or_construction_mutated"] is False
    assert status["feature_complete_cp3_topology_triangulated"] is False
    assert status["mesh_scaling"] == "FORBIDDEN"
    schema_paths = sorted((root / "contracts/qualification").glob("*.schema.json"))
    assert len(schema_paths) == 4
    checks = {
        "schema_version": 1,
        "cp5_tests": "PASS",
        "scenario_validation": validate_scenarios(root, int(status["vertex_count"])),
        "suite_and_evidence": validate_suite_and_evidence(root),
        "qualification_schema_count": len(schema_paths),
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
