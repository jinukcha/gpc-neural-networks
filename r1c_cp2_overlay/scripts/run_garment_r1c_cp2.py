#!/usr/bin/env python3
"""Build R1C CP2 exact 2D component library and assembly products."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wuxia_garment_oss.pattern_components.model import canonical_sha256
from wuxia_garment_oss.pattern_geometry import (
    assembly_recipe,
    build_geometry_library,
    compile_assembly,
    component_interfaces,
    component_registry,
    load_geometry_inputs,
    solve_interfaces,
)
from wuxia_garment_oss.pattern_geometry.contracts import cp2_schemas
from wuxia_garment_oss.pattern_geometry.evidence import render_cp2_evidence
from wuxia_garment_oss.pattern_geometry.fixtures import shifted_front_pitch


BUILD_REL = Path("build/r1c_cp2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_schemas(root: Path) -> None:
    for name, schema in cp2_schemas().items():
        write_json(root / "contracts/r1c_cp2" / name, schema)


def publish_library(build: Path, registry, library: dict, geometries: dict) -> None:
    write_json(build / "component_registry.json", registry.to_dict())
    write_json(build / "pattern_geometry_library.json", library)
    for definition in registry.components:
        write_json(build / "definitions" / f"{definition.component_id}.json", definition.to_dict())
    for instance_id, geometry in geometries.items():
        write_json(build / "geometry" / f"{instance_id}.json", geometry.to_dict())


def publish_assembly(build: Path, recipe, interfaces, interface_receipt, package, compilation) -> None:
    write_json(build / "assembly_recipe.json", recipe.to_dict())
    for spec in interfaces:
        write_json(build / "interfaces" / f"{spec.interface_id}.json", spec.to_dict())
    write_json(build / "interface_solver_receipt.json", interface_receipt)
    write_json(build / "assembled_pattern_package.json", package)
    write_json(build / "assembly_compilation_receipt.json", compilation)


def build_rejection(registry, recipe, interfaces, geometries, inputs) -> tuple[dict, dict]:
    broken = dict(geometries)
    broken["sleeve_left"] = shifted_front_pitch(geometries["sleeve_left"])
    interface_receipt = solve_interfaces(registry, recipe, interfaces, broken)
    package, compilation = compile_assembly(
        registry,
        recipe,
        broken,
        interface_receipt,
        inputs.resolved_set_sha256,
        inputs.seam_allowance_m,
        inputs.turn_of_cloth_m,
    )
    if package is not None:
        raise AssertionError("notch mismatch fixture produced an assembly package")
    return interface_receipt, compilation


def preliminary_receipt(registry, library, recipe, interface_receipt, package, compilation, rejected) -> dict:
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP2",
        "phase": "BUILT_PENDING_FRESH_PROCESS_REOPEN",
        "component_definition_count": len(registry.components),
        "geometry_authority_count": library["authority_count"],
        "component_instance_count": package["component_instance_count"],
        "interface_count": interface_receipt["interface_count"],
        "accepted_interface_count": interface_receipt["accepted_interface_count"],
        "seam_count": package["seam_count"],
        "notch_pair_count": package["notch_pair_count"],
        "interface_solver_pass": interface_receipt["accepted"],
        "assembly_compilation_pass": compilation["accepted"],
        "gore_geometry_registered": any(item["component_id"] == "SIDE_GORE_BASIC" for item in library["authorities"]),
        "notch_mismatch_fixture_rejected": not rejected[0]["accepted"],
        "rejection_partial_publication_count": rejected[1]["partial_publication_count"],
        "pattern_geometry_library_sha256": library["library_sha256"],
        "assembled_pattern_package_sha256": package["assembled_package_sha256"],
        "writer_process_id": os.getpid(),
        "triangulation_executed": False,
        "simulation_executed": False,
        "godot_executed": False,
        "r1c_cp1_predecessor_mutated": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    publish_schemas(root)
    inputs = load_geometry_inputs(root)
    registry = component_registry()
    library, geometries = build_geometry_library(inputs)
    recipe = assembly_recipe()
    interfaces = component_interfaces()
    interface_receipt = solve_interfaces(registry, recipe, interfaces, geometries)
    package, compilation = compile_assembly(
        registry,
        recipe,
        geometries,
        interface_receipt,
        inputs.resolved_set_sha256,
        inputs.seam_allowance_m,
        inputs.turn_of_cloth_m,
    )
    if package is None:
        raise RuntimeError(
            "canonical CP2 assembly rejected: "
            f"interface_errors={interface_receipt['errors']} "
            f"compilation_errors={compilation['errors']}"
        )
    rejected = build_rejection(registry, recipe, interfaces, geometries, inputs)
    publish_library(build, registry, library, geometries)
    publish_assembly(build, recipe, interfaces, interface_receipt, package, compilation)
    write_json(build / "geometry_inputs.json", inputs.to_dict())
    write_json(build / "rejections/notch_mismatch_interface_receipt.json", rejected[0])
    write_json(build / "rejections/notch_mismatch_compilation_receipt.json", rejected[1])
    receipt = preliminary_receipt(registry, library, recipe, interface_receipt, package, compilation, rejected)
    write_json(build / "cp2_preliminary_receipt.json", receipt)
    render_cp2_evidence(
        build / "cp2_exact_component_assembly_evidence.png",
        geometries,
        library,
        interface_receipt,
        package,
        rejected[1],
        receipt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
