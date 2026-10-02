#!/usr/bin/env python3
"""Build the canonical R1C CP1 parameter products and rejection receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wuxia_garment_oss.proportions.contracts import parameter_schemas
from wuxia_garment_oss.proportions.resolution.receipt import canonical_sha256, write_json
from wuxia_garment_oss.proportions.resolution.resolver import resolve_parameter_set
from wuxia_garment_oss.r1c_cp1_fixtures import canonical_context, canonical_definitions, rejection_fixtures


BUILD_REL = Path("build/r1c_cp1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def definition_set(definitions) -> dict:
    payload = {
        "contract": "ParameterDefinitionSet/1",
        "definitions": [item.to_dict() for item in sorted(definitions, key=lambda value: value.parameter_id)],
    }
    payload["definition_set_sha256"] = canonical_sha256(payload)
    return payload


def publish_schemas(root: Path) -> None:
    for name, schema in parameter_schemas().items():
        write_json(root / "contracts/r1c_cp1" / name, schema)


def publish_rejections(build: Path, context) -> dict[str, dict]:
    receipts = {}
    for name, definitions in rejection_fixtures().items():
        resolved, receipt = resolve_parameter_set(definitions, context, f"R1C_CP1_REJECT_{name.upper()}")
        if resolved is not None or receipt["accepted"]:
            raise RuntimeError(f"rejection fixture unexpectedly resolved: {name}")
        write_json(build / "rejections" / f"{name}.json", receipt)
        receipts[name] = receipt
    return receipts


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    context = canonical_context()
    definitions = canonical_definitions()
    publish_schemas(root)
    definitions_payload = definition_set(definitions)
    context_payload = context.to_dict()
    resolved, receipt = resolve_parameter_set(definitions, context, "R1C_CP1_CANONICAL_RESOLUTION")
    if resolved is None or not receipt["accepted"]:
        raise RuntimeError(receipt)
    write_json(build / "parameter_definitions.json", definitions_payload)
    write_json(build / "resolution_context.json", context_payload)
    write_json(build / "resolved_parameter_set.json", resolved)
    write_json(build / "parameter_resolution_receipt.json", receipt)
    rejections = publish_rejections(build, context)
    second, second_receipt = resolve_parameter_set(definitions, context, "R1C_CP1_CANONICAL_RESOLUTION")
    same_process_pass = bool(
        second is not None
        and second["resolved_set_sha256"] == resolved["resolved_set_sha256"]
        and second_receipt["receipt_sha256"] == receipt["receipt_sha256"]
    )
    preliminary = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP1",
        "phase": "BUILT_PENDING_FRESH_PROCESS_REOPEN",
        "writer_process_id": os.getpid(),
        "parameter_count": len(resolved["parameters"]),
        "mode_count": len({item["mode"] for item in resolved["parameters"]}),
        "reference_scope_count": len({path.split(".", 1)[0] for item in resolved["parameters"] for path in item["reference_paths"] if not path.startswith("param.")}),
        "clamp_count": receipt["clamp_count"],
        "rejection_fixture_count": len(rejections),
        "same_process_determinism_pass": same_process_pass,
        "resolved_set_sha256": resolved["resolved_set_sha256"],
        "geometry_executed": False,
        "triangulation_executed": False,
        "simulation_executed": False,
        "godot_executed": False,
        "r1c_cp0_predecessor_mutated": False,
    }
    preliminary["receipt_sha256"] = canonical_sha256(preliminary)
    write_json(build / "cp1_preliminary_receipt.json", preliminary)
    write_json(root / "R1C_STATUS.json", preliminary)
    print(json.dumps(preliminary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
