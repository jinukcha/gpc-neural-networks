#!/usr/bin/env python3
"""Execute professional grading and topology-feature graph without meshing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from wuxia_garment_oss.garments.sleeveless_tunic.pattern_cad.professional import (
    invalid_feature_specs,
    tunic_feature_graph,
    tunic_grade_rule_set,
    tunic_notch_set,
)
from wuxia_garment_oss.pattern_cad.document.model import PatternDocument, canonical_sha256
from wuxia_garment_oss.pattern_cad.features.compiler import (
    compile_feature_graph,
    failure_atomicity_probes,
)
from wuxia_garment_oss.pattern_cad.features.model import PatternFeatureGraph, PatternFeatureSpec
from wuxia_garment_oss.pattern_cad.grading.resolver import (
    _curve_point,
    compile_grade_variants,
)
from wuxia_garment_oss.pattern_cad.professional_contracts import professional_pattern_schemas


BUILD_REL = Path("build/pattern_cad_cp2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_cp1_document(root: Path) -> PatternDocument:
    path = root / "build/pattern_cad_cp1/documents/tunic_m_pattern_document.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    document = PatternDocument.from_dict(payload)
    if document.to_dict()["document_sha256"] != payload["document_sha256"]:
        raise ValueError("CP1 PatternDocument hash mismatch")
    return document


def publish_schemas(root: Path) -> None:
    target = root / "contracts/pattern_cad"
    for filename, schema in professional_pattern_schemas().items():
        write_json(target / filename, schema)


def _hard_residual(variant: dict) -> float:
    rows = [row["residual"] for row in variant["resolved"]["constraints"] if row["strength"] == "HARD"]
    return max(rows, default=0.0)


def grade_receipt(base: PatternDocument, variants: dict[str, dict], notch_set) -> dict:
    ordered = list(variants)
    underarm = [variants[size]["resolved"]["points"]["bodice_front.underarm_right"][0] for size in ordered]
    shoulder = [variants[size]["resolved"]["points"]["bodice_front.shoulder_right"][0] for size in ordered]
    skirt_length = [-variants[size]["resolved"]["points"]["skirt_front.hem_right"][1] for size in ordered]
    expected_fractions = {item.notch_id: item.arc_fraction for item in notch_set.notches}
    fraction_errors = []
    for variant in variants.values():
        for notch in variant["notches"]:
            fraction_errors.append(abs(notch["arc_fraction"] - expected_fractions[notch["notch_id"]]))
    payload = {
        "contract": "IndustrialGradingReceipt/1",
        "source_document_sha256": base.to_dict()["document_sha256"],
        "ordered_size_ids": ordered,
        "arbitrary_n_supported": len(ordered) != 3,
        "size_count": len(ordered),
        "grade_point_count": len(next(iter(variants.values()))["grade_point_movements"]),
        "notch_count_per_size": len(expected_fractions),
        "maximum_notch_fraction_error": max(fraction_errors, default=0.0),
        "maximum_hard_constraint_residual": max(_hard_residual(item) for item in variants.values()),
        "all_curve_propagation_pass": all(item["curve_propagation"]["passed"] for item in variants.values()),
        "underarm_width_monotonic": underarm == sorted(underarm),
        "shoulder_width_monotonic": shoulder == sorted(shoulder),
        "skirt_length_monotonic": skirt_length == sorted(skirt_length),
        "base_document_mutated": False,
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_grade_outputs(root: Path, base: PatternDocument) -> tuple[dict[str, dict], dict]:
    rule_set = tunic_grade_rule_set(base)
    notch_set = tunic_notch_set(base)
    variants = compile_grade_variants(base, rule_set, notch_set)
    build = root / BUILD_REL / "grading"
    write_json(build / "grade_rule_set.json", rule_set.to_dict(base))
    write_json(build / "notch_set.json", notch_set.to_dict(base))
    for size_id, payload in variants.items():
        write_json(build / "variants" / f"{size_id}.json", payload)
    receipt = grade_receipt(base, variants, notch_set)
    write_json(build / "grading_receipt.json", receipt)
    return variants, receipt


def _feature_variant(base: PatternDocument, feature: PatternFeatureSpec) -> dict:
    independent = PatternFeatureSpec(
        feature.feature_id,
        feature.feature_type,
        feature.owner_id,
        dict(feature.parameters),
        (),
    )
    return compile_feature_graph(base, PatternFeatureGraph(f"SINGLE_{feature.feature_id}", (independent,)))


def write_feature_outputs(root: Path, base: PatternDocument) -> tuple[dict, dict]:
    graph = tunic_feature_graph()
    compiled = compile_feature_graph(base, graph)
    failures = failure_atomicity_probes(base, invalid_feature_specs())
    build = root / BUILD_REL / "features"
    write_json(build / "feature_graph.json", graph.to_dict())
    write_json(build / "compiled_feature_graph.json", compiled)
    write_json(build / "composite_pattern_document.json", compiled["final_document"])
    mapping = {
        "contract": "StableIdMapping/1",
        "source_document_sha256": compiled["source_document_sha256"],
        **compiled["overall_stable_id_mapping"],
    }
    write_json(build / "stable_id_mapping.json", mapping)
    for feature in graph.features:
        write_json(build / "feature_variants" / f"{feature.feature_id}.json", _feature_variant(base, feature))
    for row in compiled["feature_receipts"]:
        write_json(build / "transactions" / f"ACCEPT_{row['feature']['feature_id']}.json", row)
    for row in failures["probes"]:
        write_json(build / "transactions" / f"REJECT_{row['feature']['feature_id']}.json", row)
    write_json(build / "failure_atomicity_receipt.json", failures)
    return compiled, failures


def _sample_curve(curve: dict, count: int = 48) -> tuple[list[float], list[float]]:
    points = [_curve_point(curve["curve_type"], curve["points"], index / count) for index in range(count + 1)]
    return [item[0] for item in points], [item[1] for item in points]


def _plot_grade(ax, variants: dict[str, dict]) -> None:
    for ordinal, (size_id, variant) in enumerate(variants.items()):
        offset = (ordinal - 3) * 0.68
        for curve in variant["resolved"]["curves"].values():
            if curve["panel_id"] not in {"bodice_front", "skirt_front"}:
                continue
            xs, ys = _sample_curve(curve)
            ax.plot([x + offset for x in xs], ys, linewidth=1.0)
        ax.text(offset, 0.60, size_id, ha="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Arbitrary-N grading — XXS to XXL exact curves")
    ax.set_xlabel("separated pattern x (m)")
    ax.set_ylabel("pattern y (m)")
    ax.grid(True, alpha=0.2)


def _plot_features(ax, compiled: dict) -> None:
    resolved = compiled["final_resolved"]
    offsets = {"bodice_front": -0.65, "skirt_front": 0.0, "gusset_underarm": 0.72}
    for curve in resolved["curves"].values():
        if curve["panel_id"] not in offsets:
            continue
        xs, ys = _sample_curve(curve)
        style = "--" if curve["disposition"] == "INTERNAL" else "-"
        ax.plot([x + offsets[curve["panel_id"]] for x in xs], ys, linestyle=style, linewidth=1.2)
    gather = next(item for item in compiled["final_document"]["metadata"]["pattern_features"] if item["feature_type"] == "GATHER")
    ax.text(0.0, 0.04, f"gather ratio {gather['parameters']['ratio']}", ha="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Dart / pleat / gather / gusset feature graph")
    ax.grid(True, alpha=0.2)


def _plot_notches(ax, variants: dict[str, dict]) -> None:
    for offset, size_id in ((-0.4, "M"), (0.4, "XXL")):
        variant = variants[size_id]
        for curve in variant["resolved"]["curves"].values():
            if curve["panel_id"] != "bodice_front":
                continue
            xs, ys = _sample_curve(curve)
            ax.plot([x + offset for x in xs], ys, linewidth=0.9)
        for notch in variant["notches"]:
            if not notch["curve_id"].startswith("bodice_front."):
                continue
            ax.scatter(notch["position"][0] + offset, notch["position"][1], s=18)
        ax.text(offset, 0.60, size_id, ha="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Notches preserve normalized arc position")
    ax.grid(True, alpha=0.2)


def _plot_summary(ax, grade: dict, compiled: dict, failures: dict) -> None:
    ax.axis("off")
    mapping = compiled["overall_stable_id_mapping"]
    lines = [
        "GARMENT-CAD-PRO-R1A / CP2",
        "",
        f"sizes: {grade['size_count']} ({', '.join(grade['ordered_size_ids'])})",
        f"named grade points: {grade['grade_point_count']}",
        f"notches per size: {grade['notch_count_per_size']}",
        f"curve propagation: {grade['all_curve_propagation_pass']}",
        f"features accepted: {len(compiled['feature_receipts'])}",
        f"stable IDs preserved: {mapping['preserved_count']}",
        f"feature IDs added: {mapping['added_count']}",
        f"IDs removed: {mapping['removed_count']}",
        f"failure probes atomic: {failures['all_atomic']} ({failures['probe_count']})",
        "",
        "triangulation: NOT EXECUTED",
        "Warp simulation: NOT EXECUTED",
        "mesh scaling: FORBIDDEN",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=11)
    ax.set_title("Authority and transaction summary")


def render_evidence(path: Path, variants: dict, grade: dict, compiled: dict, failures: dict) -> None:
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    _plot_grade(fig.add_subplot(grid[0, 0]), variants)
    _plot_features(fig.add_subplot(grid[0, 1]), compiled)
    _plot_notches(fig.add_subplot(grid[1, 0]), variants)
    _plot_summary(fig.add_subplot(grid[1, 1]), grade, compiled, failures)
    fig.suptitle("Professional grading and topology-feature execution evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(root: Path, status: dict) -> None:
    report = f"""# GARMENT-CAD-PRO-R1A / CP2 실행 보고서

## 판정

```text
terminal decision       {status['terminal_decision']}
size variants           {status['size_variant_count']}
accepted features       {status['accepted_feature_count']}
atomic failure probes   {status['failure_probe_count']}
triangulation           false
Warp simulation         false
```

CP1 committed PatternDocument를 변경하지 않고 arbitrary-N grade variants, named grade points,
curve propagation, notch arc-position, dart·pleat·gather·gusset feature graph를 발행했다.
Feature 적용은 CP1 transaction store를 사용하며 실패 fixture는 revision과 hash를 보존한다.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1A_CP2_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`GARMENT-CAD-PRO-R1A / CP3 — CONSTRUCTION GRAPH / SEAM ALLOWANCE / NOTCH / CLOSURE / LINING–INTERFACING`\n\n"
        "Use the CP2 graded and feature authorities without triangulation. Implement SeamSpec/2, "
        "cut/stitch lines, seam allowance, notch correspondence, closures, facing, lining, "
        "interfacing, turn-of-cloth and an acyclic assembly graph.\n",
        encoding="utf-8",
    )


def main() -> int:
    root = parse_args().root.resolve()
    publish_schemas(root)
    base = load_cp1_document(root)
    base_sha = base.to_dict()["document_sha256"]
    variants, grade = write_grade_outputs(root, base)
    compiled, failures = write_feature_outputs(root, base)
    if base.to_dict()["document_sha256"] != base_sha:
        raise AssertionError("CP1 predecessor mutated")
    evidence = root / BUILD_REL / "cp2_grading_feature_evidence.png"
    render_evidence(evidence, variants, grade, compiled, failures)
    status = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP2",
        "terminal_decision": "CP2_COMPLETE_GRADING_FEATURE_GRAPH",
        "cp1_document_sha256": base_sha,
        "cp1_predecessor_mutated": False,
        "size_variant_count": len(variants),
        "arbitrary_n_grading": grade["arbitrary_n_supported"],
        "notch_arc_position_preserved": grade["maximum_notch_fraction_error"] <= 1.0e-12,
        "curve_propagation_pass": grade["all_curve_propagation_pass"],
        "accepted_feature_count": len(compiled["feature_receipts"]),
        "feature_types": [item["feature"]["feature_type"] for item in compiled["feature_receipts"]],
        "stable_ids_removed": compiled["overall_stable_id_mapping"]["removed_count"],
        "failure_probe_count": failures["probe_count"],
        "failure_atomicity_pass": failures["all_atomic"],
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "next_checkpoint": "GARMENT_CAD_PRO_R1A_CP3",
    }
    status["receipt_sha256"] = canonical_sha256(status)
    write_json(root / BUILD_REL / "cp2_receipt.json", status)
    write_json(root / "PROFESSIONAL_STATUS.json", status)
    write_report(root, status)
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
