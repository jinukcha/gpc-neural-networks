#!/usr/bin/env python3
"""Build R1C CP0 modular contracts and visual acceptance authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.pattern_assembly.compiler import evaluate_recipe
from wuxia_garment_oss.pattern_components.model import canonical_sha256
from wuxia_garment_oss.r1c_cp0_contracts import r1c_cp0_schemas
from wuxia_garment_oss.r1c_cp0_fixtures.components import assembly_recipe, component_interfaces, component_registry
from wuxia_garment_oss.r1c_cp0_fixtures.visual import clean_observations, rejected_observations, visual_profile
from wuxia_garment_oss.visual_acceptance.evaluator import evaluate_visual_review
from wuxia_garment_oss.visual_acceptance.evidence import render_cp0_evidence


BUILD_REL = Path("build/r1c_cp0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_schemas(root: Path) -> None:
    for name, schema in r1c_cp0_schemas().items():
        write_json(root / "contracts/r1c_cp0" / name, schema)


def _evidence_refs(profile, garment_family: str) -> dict[str, list[str]]:
    views = [
        item.view_id
        for item in profile.required_views
        if "ALL" in item.applicable_families or garment_family in item.applicable_families
    ]
    return {gate.gate_id: views for gate in profile.gates}


def _terminal_receipt(assembly, broken, clean, rejected, registry, profile) -> dict:
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP0",
        "terminal_decision": "CP0_COMPLETE_VISUAL_AND_MODULAR_FOUNDATION",
        "cp0_acceptance": bool(
            assembly["accepted"]
            and not broken["accepted"]
            and clean["product_acceptance"]
            and not rejected["product_acceptance"]
        ),
        "component_definition_count": len(registry.components),
        "component_instance_count": assembly["component_instance_count"],
        "interface_count": assembly["interface_count"],
        "all_sewn_boundaries_owned": not assembly["unbound_sewn_boundaries"],
        "incomplete_recipe_rejected": not broken["accepted"],
        "visual_profile_id": profile.profile_id,
        "visual_gate_count": len(profile.gates),
        "required_view_count": len(profile.required_views),
        "clean_visual_fixture_pass": clean["product_acceptance"],
        "defective_visual_fixture_rejected": not rejected["product_acceptance"],
        "geometry_executed": False,
        "triangulation_executed": False,
        "simulation_executed": False,
        "godot_executed": False,
        "r1b_predecessor_mutated": False,
        "next_checkpoint": "GARMENT_CAD_PRO_R1C_CP1",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write_execution_report(root: Path, receipt: dict) -> None:
    lines = [
        "# GARMENT-CAD-PRO-R1C / CP0 실행 보고서",
        "",
        "## Terminal decision",
        "",
        "```text",
        f"terminal decision              {receipt['terminal_decision']}",
        f"CP0 acceptance                 {receipt['cp0_acceptance']}",
        f"component definitions          {receipt['component_definition_count']}",
        f"component instances            {receipt['component_instance_count']}",
        f"interfaces                     {receipt['interface_count']}",
        f"visual gates                   {receipt['visual_gate_count']}",
        f"required views                 {receipt['required_view_count']}",
        "geometry executed              false",
        "simulation executed            false",
        "Godot executed                 false",
        "```",
        "",
        "CP0는 exact 2D pattern component, boundary interface, garment recipe와 visual acceptance authority만 발행한다.",
        "R1B CP6의 rigged products와 runtime build는 immutable predecessor로 유지한다.",
        "",
    ]
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1C_CP0_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_status_docs(root: Path, receipt: dict) -> None:
    status = {
        **receipt,
        "program": "GARMENT_CAD_PRO_R1C",
        "implementation_status": "IN_PROGRESS",
        "design_authority": "MODULAR_PATTERN_COMPONENT_RATIO_COMPLETION_REPAIR",
    }
    write_json(root / "R1C_STATUS.json", status)
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "**`GARMENT-CAD-PRO-R1C / CP1 — ABSOLUTE / RELATIVE / AUTO-DERIVED RATIO PARAMETER ENGINE`**\n\n"
        "Implement typed parameter references, dependency DAG evaluation, bounds, cycle rejection, "
        "and immutable resolution receipts. Do not triangulate, simulate, or rebind products in CP1.\n",
        encoding="utf-8",
    )
    _write_execution_report(root, receipt)


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    publish_schemas(root)
    registry = component_registry()
    interfaces = component_interfaces()
    recipe = assembly_recipe(True)
    broken_recipe = assembly_recipe(False)
    assembly = evaluate_recipe(registry, recipe, interfaces)
    broken = evaluate_recipe(registry, broken_recipe, interfaces)
    profile = visual_profile()
    evidence_refs = _evidence_refs(profile, recipe.garment_family)
    clean = evaluate_visual_review(
        profile, recipe.garment_family, True, clean_observations(), evidence_refs
    )
    rejected = evaluate_visual_review(
        profile, recipe.garment_family, True, rejected_observations(), evidence_refs
    )
    write_json(build / "component_registry.json", registry.to_dict())
    for component in registry.components:
        write_json(build / "components" / f"{component.component_id}.json", component.to_dict())
    for interface in interfaces:
        write_json(build / "interfaces" / f"{interface.interface_id}.json", interface.to_dict())
    write_json(build / "recipes/sleeved_tunic_reference.json", recipe.to_dict())
    write_json(build / "recipes/incomplete_sleeved_tunic_fixture.json", broken_recipe.to_dict())
    write_json(build / "receipts/assembly_admission.json", assembly)
    write_json(build / "receipts/incomplete_recipe_rejection.json", broken)
    write_json(build / "visual_acceptance_profile.json", profile.to_dict())
    write_json(build / "receipts/clean_visual_review.json", clean)
    write_json(build / "receipts/rejected_visual_review.json", rejected)
    receipt = _terminal_receipt(assembly, broken, clean, rejected, registry, profile)
    write_json(build / "cp0_receipt.json", receipt)
    render_cp0_evidence(
        build / "cp0_modular_visual_authority_evidence.png",
        registry.to_dict(),
        recipe.to_dict(),
        [item.to_dict() for item in interfaces],
        profile.to_dict(),
        clean,
        rejected,
        assembly,
        broken,
    )
    _write_status_docs(root, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
