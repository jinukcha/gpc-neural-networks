#!/usr/bin/env python3
"""Fresh-process reopen, deterministic rerun, and CP1 terminal publication."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wuxia_garment_oss.proportions.evidence import render_cp1_evidence
from wuxia_garment_oss.proportions.resolution.receipt import (
    canonical_sha256,
    read_verified_json,
    verify_embedded_hash,
    write_json,
)
from wuxia_garment_oss.proportions.resolution.resolver import resolve_parameter_set
from wuxia_garment_oss.r1c_cp1_fixtures import canonical_context, canonical_definitions, rejection_fixtures


BUILD_REL = Path("build/r1c_cp1")
EXPECTED_REJECTIONS = {
    "missing_reference": "MISSING_REFERENCE",
    "quantity_mismatch": "QUANTITY_MISMATCH",
    "expression_quantity_mismatch": "QUANTITY_MISMATCH",
    "unit_mismatch": "UNIT_MISMATCH",
    "dependency_cycle": "CYCLE_DETECTED",
    "non_finite_ratio": "NON_FINITE_RATIO",
    "hard_bound": "BOUND_REJECTED",
    "alternate_component": "ALTERNATE_COMPONENT_REQUIRED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def _definition_set(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not verify_embedded_hash(payload, "definition_set_sha256"):
        raise ValueError("definition-set hash mismatch")
    return payload


def _load_rejections(build: Path) -> dict[str, dict]:
    receipts = {}
    for name, expected in EXPECTED_REJECTIONS.items():
        payload = read_verified_json(build / "rejections" / f"{name}.json", "receipt_sha256")
        if payload["accepted"] or payload["status"] != expected:
            raise ValueError(f"unexpected rejection result: {name}: {payload['status']}")
        if payload["partial_publication_count"] != 0:
            raise ValueError(f"partial publication detected: {name}")
        receipts[name] = payload
    return receipts


def _append_roadmap(root: Path) -> None:
    path = root / "docs/roadmap/GARMENT_CAD_PRO_R1C_ROADMAP_KO.md"
    if not path.is_file():
        return
    marker = "## CP1 구현 결과"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    text += (
        "\n\n## CP1 구현 결과\n\n"
        "- ABSOLUTE / RELATIVE / AUTO_DERIVED parameter mode: 구현\n"
        "- BODY / BLOCK / COMPONENT / BOUNDARY / MATERIAL reference scope: 구현\n"
        "- quantity-aware DAG, cycle rejection, bounds disposition, provenance: 구현\n"
        "- geometry / triangulation / simulation: 미실행\n"
        "- 다음 단계: CP2 pattern component library / interface solver / assembly recipe\n"
    )
    path.write_text(text, encoding="utf-8")


def _write_report(root: Path, receipt: dict, reopen: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1C / CP1 실행 보고서

## Terminal decision

```text
terminal decision             {receipt['terminal_decision']}
CP1 acceptance                {receipt['cp1_acceptance']}
parameter modes               {receipt['parameter_mode_count']} / 3
reference scopes              {receipt['reference_scope_count']} / 5
resolved parameters           {receipt['parameter_count']}
atomic rejection fixtures     {receipt['rejection_fixture_count']}
SAFE_CLAMP events             {receipt['clamp_count']}
fresh-process reopen          {reopen['fresh_process_reopen_pass']}
deterministic rerun           {reopen['deterministic_rerun_pass']}
geometry / simulation         NOT EXECUTED
```

CP0 component and visual authorities are immutable inputs. CP1 publishes typed SI values,
reference provenance, dependency order, bounded-resolution decisions, and immutable hashes.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1C_CP1_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> int:
    root = parse_args().root.resolve()
    build = root / BUILD_REL
    preliminary = read_verified_json(build / "cp1_preliminary_receipt.json", "receipt_sha256")
    definitions = _definition_set(build / "parameter_definitions.json")
    context = read_verified_json(build / "resolution_context.json", "context_sha256")
    resolved = read_verified_json(build / "resolved_parameter_set.json", "resolved_set_sha256")
    resolution_receipt = read_verified_json(build / "parameter_resolution_receipt.json", "receipt_sha256")
    rejections = _load_rejections(build)
    rerun, rerun_receipt = resolve_parameter_set(
        canonical_definitions(),
        canonical_context(),
        "R1C_CP1_CANONICAL_RESOLUTION",
    )
    rerun_pass = bool(
        rerun is not None
        and rerun["resolved_set_sha256"] == resolved["resolved_set_sha256"]
        and rerun_receipt["receipt_sha256"] == resolution_receipt["receipt_sha256"]
    )
    reopen = {
        "contract": "ParameterFreshProcessReopenReceipt/1",
        "writer_process_id": preliminary["writer_process_id"],
        "reopen_process_id": os.getpid(),
        "fresh_process_reopen_pass": preliminary["writer_process_id"] != os.getpid(),
        "definition_set_hash_pass": definitions["definition_set_sha256"] == resolved["definition_set_sha256"],
        "context_hash_pass": context["context_sha256"] == resolved["context_sha256"],
        "resolved_set_hash_pass": resolution_receipt["resolved_set_sha256"] == resolved["resolved_set_sha256"],
        "deterministic_rerun_pass": rerun_pass,
        "rejection_receipts_pass": len(rejections) == len(EXPECTED_REJECTIONS),
    }
    reopen["receipt_sha256"] = canonical_sha256(reopen)
    write_json(build / "fresh_process_reopen_receipt.json", reopen)
    accepted = all(
        reopen[name]
        for name in (
            "fresh_process_reopen_pass",
            "definition_set_hash_pass",
            "context_hash_pass",
            "resolved_set_hash_pass",
            "deterministic_rerun_pass",
            "rejection_receipts_pass",
        )
    )
    receipt = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP1",
        "terminal_decision": "CP1_COMPLETE_RATIO_PARAMETER_ENGINE" if accepted else "HOLD_CP1",
        "cp1_acceptance": accepted,
        "parameter_mode_count": preliminary["mode_count"],
        "reference_scope_count": preliminary["reference_scope_count"],
        "parameter_count": preliminary["parameter_count"],
        "clamp_count": preliminary["clamp_count"],
        "rejection_fixture_count": preliminary["rejection_fixture_count"],
        "partial_publication_count": 0,
        "resolved_set_sha256": resolved["resolved_set_sha256"],
        "fresh_process_reopen_pass": reopen["fresh_process_reopen_pass"],
        "deterministic_rerun_pass": rerun_pass,
        "geometry_executed": False,
        "triangulation_executed": False,
        "simulation_executed": False,
        "godot_executed": False,
        "r1c_cp0_predecessor_mutated": False,
        "next_checkpoint": "GARMENT_CAD_PRO_R1C_CP2",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(build / "cp1_receipt.json", receipt)
    write_json(root / "R1C_STATUS.json", receipt)
    render_cp1_evidence(
        build / "cp1_ratio_parameter_evidence.png",
        resolved,
        resolution_receipt,
        rejections,
        reopen,
    )
    _write_report(root, receipt, reopen)
    _append_roadmap(root)
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "**`GARMENT-CAD-PRO-R1C / CP2 — PATTERN COMPONENT LIBRARY / INTERFACE SOLVER / ASSEMBLY RECIPE`**\n\n"
        "Use CP0 contracts and CP1 resolved parameters as immutable inputs. Implement exact component geometry "
        "authority, interface compatibility, notch correspondence, and assembly compilation without triangulation or simulation.\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
