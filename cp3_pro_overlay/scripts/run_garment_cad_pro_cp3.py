#!/usr/bin/env python3
"""Execute professional construction authority without meshing or simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.construction.compiler import (
    compile_construction_package,
    construction_failure_probes,
)
from wuxia_garment_oss.construction.contracts import construction_schemas
from wuxia_garment_oss.construction.render import render_construction_evidence
from wuxia_garment_oss.garments.sleeveless_tunic.construction.professional import (
    tunic_construction_authority,
)
from wuxia_garment_oss.pattern_cad.document.model import PatternDocument, canonical_sha256
from wuxia_garment_oss.pattern_cad.document.resolver import require_resolved


BUILD_REL = Path("build/construction_cp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_cp2_authority(root: Path) -> tuple[PatternDocument, dict, set[str]]:
    base = root / "build/pattern_cad_cp2"
    document_payload = json.loads(
        (base / "features/composite_pattern_document.json").read_text(encoding="utf-8")
    )
    document = PatternDocument.from_dict(document_payload)
    if document.to_dict()["document_sha256"] != document_payload["document_sha256"]:
        raise ValueError("CP2 composite PatternDocument hash mismatch")
    resolved = require_resolved(document)
    notch_payload = json.loads((base / "grading/notch_set.json").read_text(encoding="utf-8"))
    notch_ids = {str(item["notch_id"]) for item in notch_payload["notches"]}
    return document, resolved, notch_ids


def publish_schemas(root: Path) -> None:
    target = root / "contracts/construction"
    for filename, schema in construction_schemas().items():
        write_json(target / filename, schema)


def publish_subpackages(root: Path, package: dict, failures: dict) -> None:
    build = root / BUILD_REL
    write_json(build / "construction_package.json", package)
    for seam in package["seam_specs"]:
        write_json(build / "seam_specs" / f"{seam['seam_id']}.json", seam)
    write_json(build / "stitch_cut_lines.json", {
        "contract": "ConstructionLineCollection/1",
        "seam_lines": package["seam_lines"],
        "edge_finishes": package["edge_finishes"],
    })
    write_json(build / "notch_correspondence.json", {
        "contract": "NotchCorrespondenceSet/1",
        "pairs": package["notch_correspondence"],
    })
    write_json(build / "closure_facing_layers.json", {
        "contract": "ConstructionComponentSet/1",
        "closures": package["closures"],
        "facings": package["facings"],
        "layer_pieces": package["layer_pieces"],
        "turn_of_cloth": package["turn_of_cloth"],
    })
    write_json(build / "assembly_graph.json", package["assembly_plan"]["graph"])
    write_json(build / "assembly_plan.json", package["assembly_plan"])
    write_json(build / "bill_of_materials.json", {
        "contract": "BillOfMaterials/1",
        "items": package["bill_of_materials"],
    })
    write_json(build / "failure_atomicity_receipt.json", failures)


def build_receipt(package: dict, failures: dict) -> dict:
    gathered = [item for item in package["seam_specs"] if item["seam_type"] == "GATHERED_SEAM"]
    receipt = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP3",
        "terminal_decision": "CP3_COMPLETE_CONSTRUCTION_GRAPH",
        "source_pattern_document_sha256": package["source_pattern_document_sha256"],
        "source_document_mutated": package["source_document_mutated"],
        "seam_spec_count": len(package["seam_specs"]),
        "edge_finish_count": len(package["edge_finishes"]),
        "notch_pair_count": len(package["notch_correspondence"]),
        "closure_count": len(package["closures"]),
        "facing_count": len(package["facings"]),
        "layer_piece_count": len(package["layer_pieces"]),
        "turn_of_cloth_count": len(package["turn_of_cloth"]),
        "assembly_operation_count": package["assembly_plan"]["operation_count"],
        "assembly_cycle_free": package["assembly_plan"]["cycle_free"],
        "gather_ratio": gathered[0]["gather_ratio"] if gathered else 1.0,
        "failure_probe_count": failures["probe_count"],
        "failure_atomicity_pass": failures["all_atomic"],
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "next_checkpoint": "GARMENT_CAD_PRO_R1A_CP4",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def write_report(root: Path, receipt: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1A / CP3 실행 보고서

## 판정

```text
terminal decision       {receipt['terminal_decision']}
SeamSpec/2              {receipt['seam_spec_count']}
edge finishes           {receipt['edge_finish_count']}
notch pairs             {receipt['notch_pair_count']}
assembly operations     {receipt['assembly_operation_count']}
assembly cycle-free     {str(receipt['assembly_cycle_free']).lower()}
triangulation           false
Warp simulation         false
```

CP2의 graded·feature-complete PatternDocument를 변경하지 않고 stitch line, cut line,
seam allowance, notch correspondence, gather/ease, closure, facing, lining, interfacing,
turn-of-cloth와 cycle-free assembly DAG를 발행했다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1A_CP3_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`GARMENT-CAD-PRO-R1A / CP4 — MATERIAL MEASUREMENT SET / CALIBRATION / CPU–WARP PARITY`\n\n"
        "Use the CP3 PatternDocument and ConstructionPackage as immutable predecessors. "
        "Implement three measured material fixtures, solver-parameter calibration, residual receipts, "
        "and CPU/Warp metric parity without motion-fit product acceptance.\n",
        encoding="utf-8",
    )


def main() -> int:
    root = parse_args().root.resolve()
    publish_schemas(root)
    document, resolved, source_notch_ids = load_cp2_authority(root)
    authority = tunic_construction_authority()
    package = compile_construction_package(
        document, resolved, authority["seams"], authority["finishes"], authority["notches"],
        source_notch_ids, authority["closures"], authority["facings"], authority["layers"],
        authority["turns"], authority["graph"], authority["bill_of_materials"],
    )
    failures = construction_failure_probes(document, authority["seams"][0], authority["graph"])
    publish_subpackages(root, package, failures)
    evidence = root / BUILD_REL / "cp3_construction_evidence.png"
    render_construction_evidence(evidence, package, failures)
    receipt = build_receipt(package, failures)
    write_json(root / BUILD_REL / "cp3_receipt.json", receipt)
    write_json(root / "PROFESSIONAL_STATUS.json", receipt)
    write_report(root, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
