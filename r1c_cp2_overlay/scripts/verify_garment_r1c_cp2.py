#!/usr/bin/env python3
"""Fresh-process reopen, deterministic rebuild, and CP2 terminal closeout."""
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
from wuxia_garment_oss.pattern_geometry.evidence import render_cp2_evidence


BUILD_REL = Path("build/r1c_cp2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_hash(payload: dict, key: str) -> bool:
    recorded = payload.get(key)
    if not isinstance(recorded, str):
        return False
    source = dict(payload)
    source.pop(key, None)
    return recorded == canonical_sha256(source)


def rebuild(root: Path):
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
        raise RuntimeError(f"deterministic CP2 rebuild rejected: {compilation['errors']}")
    return inputs, registry, library, geometries, recipe, interfaces, interface_receipt, package, compilation


def reopen_receipt(build: Path, preliminary: dict, rebuilt: tuple) -> dict:
    _, registry, library, _, recipe, _, interface_receipt, package, compilation = rebuilt
    stored_library = load_json(build / "pattern_geometry_library.json")
    stored_package = load_json(build / "assembled_pattern_package.json")
    stored_interface = load_json(build / "interface_solver_receipt.json")
    stored_compilation = load_json(build / "assembly_compilation_receipt.json")
    payload = {
        "contract": "CP2FreshProcessReopenReceipt/1",
        "writer_process_id": preliminary["writer_process_id"],
        "reopen_process_id": os.getpid(),
        "fresh_process_reopen_pass": preliminary["writer_process_id"] != os.getpid(),
        "stored_library_hash_pass": verify_hash(stored_library, "library_sha256"),
        "stored_package_hash_pass": verify_hash(stored_package, "assembled_package_sha256"),
        "stored_interface_hash_pass": verify_hash(stored_interface, "receipt_sha256"),
        "stored_compilation_hash_pass": verify_hash(stored_compilation, "receipt_sha256"),
        "deterministic_library_pass": library["library_sha256"] == stored_library["library_sha256"],
        "deterministic_package_pass": package["assembled_package_sha256"] == stored_package["assembled_pattern_package_sha256"] if "assembled_pattern_package_sha256" in stored_package else package["assembled_package_sha256"] == stored_package["assembled_package_sha256"],
        "deterministic_interface_pass": interface_receipt["receipt_sha256"] == stored_interface["receipt_sha256"],
        "deterministic_compilation_pass": compilation["receipt_sha256"] == stored_compilation["receipt_sha256"],
        "registry_hash": registry.to_dict()["registry_sha256"],
        "recipe_hash": recipe.to_dict()["recipe_sha256"],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def terminal_receipt(preliminary: dict, reopen: dict) -> dict:
    gate_names = (
        "fresh_process_reopen_pass",
        "stored_library_hash_pass",
        "stored_package_hash_pass",
        "stored_interface_hash_pass",
        "stored_compilation_hash_pass",
        "deterministic_library_pass",
        "deterministic_package_pass",
        "deterministic_interface_pass",
        "deterministic_compilation_pass",
    )
    accepted = all(bool(reopen[name]) for name in gate_names)
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP2",
        "terminal_decision": "CP2_COMPLETE_PATTERN_COMPONENT_ASSEMBLY" if accepted else "HOLD_CP2",
        "cp2_acceptance": accepted,
        "component_definition_count": preliminary["component_definition_count"],
        "geometry_authority_count": preliminary["geometry_authority_count"],
        "component_instance_count": preliminary["component_instance_count"],
        "interface_count": preliminary["interface_count"],
        "accepted_interface_count": preliminary["accepted_interface_count"],
        "seam_count": preliminary["seam_count"],
        "notch_pair_count": preliminary["notch_pair_count"],
        "interface_solver_pass": preliminary["interface_solver_pass"],
        "assembly_compilation_pass": preliminary["assembly_compilation_pass"],
        "gore_geometry_registered": preliminary["gore_geometry_registered"],
        "notch_mismatch_fixture_rejected": preliminary["notch_mismatch_fixture_rejected"],
        "rejection_partial_publication_count": preliminary["rejection_partial_publication_count"],
        "fresh_process_reopen_pass": reopen["fresh_process_reopen_pass"],
        "deterministic_rebuild_pass": all(reopen[name] for name in gate_names[5:]),
        "pattern_geometry_library_sha256": preliminary["pattern_geometry_library_sha256"],
        "assembled_pattern_package_sha256": preliminary["assembled_pattern_package_sha256"],
        "triangulation_executed": False,
        "simulation_executed": False,
        "godot_executed": False,
        "r1c_cp1_predecessor_mutated": False,
        "next_checkpoint": "GARMENT_CAD_PRO_R1C_CP3",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_docs(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1C / CP2 실행 보고서

```text
terminal decision            {receipt['terminal_decision']}
CP2 acceptance               {receipt['cp2_acceptance']}
component definitions        {receipt['component_definition_count']}
geometry authorities         {receipt['geometry_authority_count']}
compiled instances           {receipt['component_instance_count']}
interfaces                   {receipt['interface_count']}
seams                        {receipt['seam_count']}
notch pairs                  {receipt['notch_pair_count']}
fresh-process reopen         {receipt['fresh_process_reopen_pass']}
deterministic rebuild        {receipt['deterministic_rebuild_pass']}
triangulation / simulation   NOT EXECUTED
```

CP0 modular contracts and CP1 resolved parameters are immutable inputs. CP2 publishes exact curve geometry,
physical boundary compatibility, normalized notch correspondence, and an assembled 2D pattern package.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1C_CP2_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    next_task = (
        "# Next task\n\n"
        "**`GARMENT-CAD-PRO-R1C / CP3 — COMPLETION DIAGNOSIS / BOUNDED REPAIR / ATOMIC TRANSACTION`**\n\n"
        "Use the accepted CP2 geometry library and assembly package as immutable inputs. Implement missing-component "
        "diagnosis, safe completion, bounded source-pattern repair, topology-change HOLD, and atomic commit/rollback receipts.\n"
    )
    (root / "NEXT_TASK.md").write_text(next_task, encoding="utf-8")


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    preliminary = load_json(build / "cp2_preliminary_receipt.json")
    rebuilt = rebuild(root)
    reopen = reopen_receipt(build, preliminary, rebuilt)
    receipt = terminal_receipt(preliminary, reopen)
    write_json(build / "fresh_process_reopen_receipt.json", reopen)
    write_json(build / "cp2_receipt.json", receipt)
    write_json(root / "R1C_STATUS.json", receipt)
    _, _, library, geometries, _, _, interfaces, package, _ = rebuilt
    rejected = load_json(build / "rejections/notch_mismatch_compilation_receipt.json")
    render_cp2_evidence(build / "cp2_exact_component_assembly_evidence.png", geometries, library, interfaces, package, rejected, receipt)
    write_docs(root, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
