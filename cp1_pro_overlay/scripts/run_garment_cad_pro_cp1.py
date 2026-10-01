#!/usr/bin/env python3
"""Execute the recovered CP1 PatternDocument migration and transaction pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wuxia_garment_oss.anthropometry.v2.migration import migrate_v1_profile
from wuxia_garment_oss.garments.sleeveless_tunic.pattern_cad.migrate import migrate_reference_tunic
from wuxia_garment_oss.pattern_cad.document.resolver import require_resolved
from wuxia_garment_oss.pattern_cad.persistence.json_store import load_pattern_document, save_pattern_document
from wuxia_garment_oss.pattern_cad.transaction.commands import SetExpressionCommand, SetInputCommand
from wuxia_garment_oss.pattern_cad.transaction.store import PatternDocumentStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_reference_v1(root: Path) -> dict:
    return json.loads((root / "fixtures/sizing/bodies/REFERENCE.json").read_text(encoding="utf-8"))


def _load_parameter_package(root: Path) -> dict:
    path = root / "build/tunic_pilot/sizing_cp2/pattern_parameter_packages/STANDARD_M.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _source_landmarks(package: dict) -> dict[str, list[float]]:
    result = {}
    for panel in package["panels"]:
        panel_id = str(panel["panel_id"])
        for name, point in panel["landmarks"].items():
            result[f"{panel_id}.{name}"] = [float(point[0]), float(point[1])]
    return result


def _parity(source: dict, resolved: dict) -> dict:
    source_points = _source_landmarks(source)
    errors = {}
    for point_id, expected in source_points.items():
        actual = resolved["points"][point_id]
        errors[point_id] = max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1]))
    return {
        "source_point_count": len(source_points),
        "maximum_absolute_error": max(errors.values()),
        "errors": errors,
    }


def _sample_curve(curve: dict, count: int = 80) -> np.ndarray:
    points = np.asarray(curve["points"], dtype=np.float64)
    t = np.linspace(0.0, 1.0, count)[:, None]
    if curve["curve_type"] == "LINE":
        return (1.0 - t) * points[0] + t * points[1]
    if curve["curve_type"] == "QUADRATIC_BEZIER":
        return (1.0 - t) ** 2 * points[0] + 2.0 * (1.0 - t) * t * points[1] + t**2 * points[2]
    return (
        (1.0 - t) ** 3 * points[0]
        + 3.0 * (1.0 - t) ** 2 * t * points[1]
        + 3.0 * (1.0 - t) * t**2 * points[2]
        + t**3 * points[3]
    )


def _draw_document(ax, resolved: dict, title: str, offset: float = 0.0) -> None:
    for curve in resolved["curves"].values():
        if curve["panel_id"] not in {"bodice_front", "skirt_front"}:
            continue
        sampled = _sample_curve(curve)
        ax.plot(sampled[:, 0] + offset, sampled[:, 1], linewidth=1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("pattern x (m)")
    ax.set_ylabel("pattern y (m)")
    ax.grid(True, alpha=0.25)


def _plot_constraints(ax, resolved: dict) -> None:
    rows = resolved["constraints"]
    labels = [row["constraint_id"].split(".")[-1] for row in rows]
    residual = [max(float(row["residual"]), 1.0e-14) for row in rows]
    tolerance = [max(float(row["tolerance"]), 1.0e-14) for row in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.18, residual, 0.36, label="residual")
    ax.bar(x + 0.18, tolerance, 0.36, label="tolerance")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    ax.set_title("Hard/soft constraint residuals")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)


def _plot_transactions(ax, receipts: list[dict], parity: dict) -> None:
    ax.axis("off")
    lines = [
        "operation | accepted | before→after revision | hash transition",
        "-" * 92,
    ]
    for receipt in receipts:
        lines.append(
            f"{receipt['operation']:<15} {str(receipt['accepted']):<8} "
            f"{receipt['before_revision']}→{receipt['after_revision']:<4} "
            f"{receipt['before_sha256'][:10]}→{receipt['after_sha256'][:10]}"
        )
        if receipt.get("error"):
            lines.append(f"  rejected: {receipt['error']}")
    lines.extend([
        "",
        f"source landmark parity max error: {parity['maximum_absolute_error']:.3e} m",
        "triangulation: NOT EXECUTED",
        "Warp simulation: NOT EXECUTED",
        "mesh scaling: FORBIDDEN",
    ])
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("Atomic transaction / undo / redo receipt")


def render_evidence(path: Path, original: dict, edited: dict, receipts: list[dict], parity: dict) -> None:
    figure = plt.figure(figsize=(20, 13.333), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    _draw_document(figure.add_subplot(grid[0, 0]), original, "Revision 0 — migrated tunic M exact curves")
    _draw_document(figure.add_subplot(grid[0, 1]), original, "Original vs accepted chest-ease edit")
    _draw_document(figure.axes[-1], edited, "", offset=0.72)
    _plot_constraints(figure.add_subplot(grid[1, 0]), edited)
    _plot_transactions(figure.add_subplot(grid[1, 1]), receipts, parity)
    figure.suptitle("GARMENT-CAD-PRO-R1A / CP1 — PatternDocument, constraints, atomic edits")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def publish_schemas(root: Path) -> None:
    schemas = {
        "body_measurement_profile_v2.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "BodyMeasurementProfile/2",
            "type": "object",
            "required": ["contract", "profile_id", "records", "profile_sha256"],
            "properties": {"contract": {"const": "BodyMeasurementProfile/2"}},
        },
        "pattern_document.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "PatternDocument/1",
            "type": "object",
            "required": ["contract", "document_id", "revision", "points", "curves", "constraints", "document_sha256"],
            "properties": {"contract": {"const": "PatternDocument/1"}},
        },
        "resolved_pattern_document.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "ResolvedPatternDocument/1",
            "type": "object",
            "required": ["contract", "document_id", "points", "curves", "hard_constraints_pass"],
            "properties": {"contract": {"const": "ResolvedPatternDocument/1"}},
        },
    }
    for name, schema in schemas.items():
        write_json(root / "contracts/professional" / name, schema)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    build = root / "build/pattern_cad_cp1"
    profile, migration_report = migrate_v1_profile(_load_reference_v1(root))
    parameter_package = _load_parameter_package(root)
    document = migrate_reference_tunic(parameter_package, profile)
    original_resolved = require_resolved(document)
    parity = _parity(parameter_package, original_resolved)
    if parity["maximum_absolute_error"] > 1.0e-12:
        raise ValueError(f"reference pattern parity failed: {parity['maximum_absolute_error']}")

    store = PatternDocumentStore(document)
    accepted = store.apply(SetInputCommand("front_chest_ease", 0.056))
    if not accepted.accepted:
        raise ValueError(accepted.error)
    rejected = store.apply(SetExpressionCommand("neck_half", "-0.01"))
    if rejected.accepted:
        raise ValueError("invalid neck edit unexpectedly committed")
    edited_sha = store.document.to_dict()["document_sha256"]
    undo = store.undo()
    if store.document.to_dict()["document_sha256"] != document.to_dict()["document_sha256"]:
        raise ValueError("undo did not restore revision 0")
    redo = store.redo()
    if store.document.to_dict()["document_sha256"] != edited_sha:
        raise ValueError("redo did not restore accepted revision")

    final_document = store.document
    edited_resolved = require_resolved(final_document)
    save_path = build / "pattern_document.json"
    saved_sha = save_pattern_document(save_path, final_document)
    reopened = load_pattern_document(save_path)
    reopen_sha = reopened.to_dict()["document_sha256"]
    if saved_sha != reopen_sha:
        raise ValueError("fresh-process document hash mismatch")

    publish_schemas(root)
    write_json(build / "anthropometry_v2_profile.json", profile.to_dict())
    write_json(build / "anthropometry_v2_migration_report.json", migration_report)
    write_json(build / "resolved_pattern_document.json", edited_resolved)
    write_json(build / "reference_parity.json", parity)
    write_json(build / "transaction_receipt.json", {"transactions": store.receipts})
    evidence = build / "cp1_pattern_document_evidence.png"
    render_evidence(evidence, original_resolved, edited_resolved, store.receipts, parity)

    status = {
        "checkpoint": "GARMENT_CAD_PRO_R1A_CP1",
        "terminal_decision": "CP1_COMPLETE_PATTERN_DOCUMENT",
        "byte_exact_cp0_predecessor": False,
        "cp0_recovery_basis": "CP3 sizing authority after two failed acquisition paths",
        "pattern_document_revision": final_document.revision,
        "reference_parity_max_error": parity["maximum_absolute_error"],
        "accepted_edit": accepted.accepted,
        "rejected_edit_atomic": not rejected.accepted,
        "undo_pass": undo.accepted,
        "redo_pass": redo.accepted,
        "fresh_reopen_pass": saved_sha == reopen_sha,
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "next_checkpoint": "GARMENT_CAD_PRO_R1A_CP2",
    }
    write_json(build / "cp1_receipt.json", status)
    write_json(root / "PROFESSIONAL_STATUS.json", status)
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`GARMENT-CAD-PRO-R1A / CP2 — INDUSTRIAL GRADING / DART–PLEAT–GATHER–GUSSET FEATURE GRAPH`\n\n"
        "Use PatternDocument/1 and the transaction kernel as immutable inputs. Implement named grade points, "
        "size-rule propagation and topology-aware dart, pleat, gather and gusset features. Do not triangulate "
        "or run Warp product simulation yet.\n",
        encoding="utf-8",
    )
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
