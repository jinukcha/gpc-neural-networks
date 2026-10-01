"""Compile professional construction authority without meshing or simulation."""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..pattern_cad.document.model import PatternDocument, canonical_sha256
from .assembly import compile_assembly_plan, cycle_failure_probe
from .lines import closure_record, curve_point, facing_record, line_record
from .model import (
    ClosureSpec,
    ConstructionGraph,
    EdgeFinishSpec,
    FacingSpec,
    LayerPieceSpec,
    NotchMatch,
    SeamSpecV2,
    TurnOfClothSpec,
)


def _unique_ids(rows: Sequence[object], attribute: str, label: str) -> None:
    values = [str(getattr(item, attribute)) for item in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} IDs")


def _validate_inputs(
    document: PatternDocument,
    seams: Sequence[SeamSpecV2],
    finishes: Sequence[EdgeFinishSpec],
    notches: Sequence[NotchMatch],
    closures: Sequence[ClosureSpec],
    facings: Sequence[FacingSpec],
    layers: Sequence[LayerPieceSpec],
    turns: Sequence[TurnOfClothSpec],
    graph: ConstructionGraph,
) -> None:
    groups = (
        (seams, "seam_id", "seam"),
        (finishes, "finish_id", "finish"),
        (notches, "pair_id", "notch pair"),
        (closures, "closure_id", "closure"),
        (facings, "facing_id", "facing"),
        (layers, "piece_id", "layer piece"),
        (turns, "turn_id", "turn-of-cloth"),
    )
    for rows, attribute, label in groups:
        _unique_ids(rows, attribute, label)
    for seam in seams:
        seam.validate(document)
    for finish in finishes:
        finish.validate(document)
    seam_ids = {item.seam_id for item in seams}
    for notch in notches:
        notch.validate()
        if notch.seam_id not in seam_ids:
            raise ValueError(f"notch references unknown seam: {notch.pair_id}")
    for closure in closures:
        closure.validate(document)
    for facing in facings:
        facing.validate(document)
    for layer in layers:
        layer.validate(document)
    for turn in turns:
        turn.validate()
    graph.validate()


def _notch_position(resolved: Mapping[str, object], curve_id: str, fraction: float) -> list[float]:
    curve = resolved["curves"][curve_id]
    point = curve_point(str(curve["curve_type"]), curve["points"], fraction)
    return [point[0], point[1]]


def _compile_notches(
    resolved: Mapping[str, object],
    seams: Sequence[SeamSpecV2],
    notches: Sequence[NotchMatch],
    source_notch_ids: set[str],
) -> list[dict]:
    by_seam = {item.seam_id: item for item in seams}
    rows = []
    for notch in notches:
        seam = by_seam[notch.seam_id]
        for notch_id in (notch.side_a_notch_id, notch.side_b_notch_id):
            if not notch_id.startswith("CP3_") and notch_id not in source_notch_ids:
                raise ValueError(f"unknown CP2 notch identity: {notch_id}")
        rows.append({
            **notch.to_dict(),
            "side_a_curve_id": seam.side_a.curve_id,
            "side_b_curve_id": seam.side_b.curve_id,
            "side_a_position": _notch_position(resolved, seam.side_a.curve_id, notch.side_a_fraction),
            "side_b_position": _notch_position(resolved, seam.side_b.curve_id, notch.side_b_fraction),
            "fraction_error": abs(notch.side_a_fraction - notch.side_b_fraction),
        })
    return rows


def _compile_seam_lines(document: PatternDocument, resolved: Mapping[str, object], seams) -> list[dict]:
    rows = []
    for seam in seams:
        rows.append({
            "seam": seam.to_dict(),
            "side_a": line_record(document, resolved, seam.side_a, seam.allowance_a_m),
            "side_b": line_record(document, resolved, seam.side_b, seam.allowance_b_m),
        })
    return rows


def _compile_finish_lines(document: PatternDocument, resolved: Mapping[str, object], finishes) -> list[dict]:
    return [
        {"finish": item.to_dict(), "line": line_record(document, resolved, item.boundary, item.allowance_m)}
        for item in finishes
    ]


def compile_construction_package(
    document: PatternDocument,
    resolved: Mapping[str, object],
    seams: Sequence[SeamSpecV2],
    finishes: Sequence[EdgeFinishSpec],
    notches: Sequence[NotchMatch],
    source_notch_ids: set[str],
    closures: Sequence[ClosureSpec],
    facings: Sequence[FacingSpec],
    layers: Sequence[LayerPieceSpec],
    turns: Sequence[TurnOfClothSpec],
    graph: ConstructionGraph,
    bill_of_materials: Sequence[Mapping[str, object]],
) -> dict:
    document_sha = document.to_dict()["document_sha256"]
    _validate_inputs(document, seams, finishes, notches, closures, facings, layers, turns, graph)
    package = {
        "contract": "ConstructionPackage/1",
        "source_pattern_document_sha256": document_sha,
        "source_pattern_revision": document.revision,
        "seam_specs": [item.to_dict() for item in seams],
        "seam_lines": _compile_seam_lines(document, resolved, seams),
        "edge_finishes": _compile_finish_lines(document, resolved, finishes),
        "notch_correspondence": _compile_notches(resolved, seams, notches, source_notch_ids),
        "closures": [closure_record(document, resolved, item) for item in closures],
        "facings": [facing_record(document, resolved, item) for item in facings],
        "layer_pieces": [item.to_dict() for item in layers],
        "turn_of_cloth": [item.to_dict() for item in turns],
        "assembly_plan": compile_assembly_plan(graph),
        "bill_of_materials": [dict(item) for item in bill_of_materials],
        "source_document_mutated": False,
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
    }
    if document.to_dict()["document_sha256"] != document_sha:
        raise AssertionError("construction compiler mutated PatternDocument")
    package["package_sha256"] = canonical_sha256(package)
    return package


def _invalid_probe(document: PatternDocument, seam: SeamSpecV2, label: str) -> dict:
    before = document.to_dict()
    try:
        seam.validate(document)
    except ValueError as exc:
        after = document.to_dict()
        return {
            "probe": label,
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
            "revision_unchanged": before["revision"] == after["revision"],
            "document_sha256_unchanged": before["document_sha256"] == after["document_sha256"],
            "failure_atomicity": before["document_sha256"] == after["document_sha256"],
        }
    raise AssertionError(f"invalid construction seam unexpectedly accepted: {label}")


def construction_failure_probes(
    document: PatternDocument,
    valid_seam: SeamSpecV2,
    graph: ConstructionGraph,
) -> dict:
    bad_curve = replace(
        valid_seam,
        seam_id="BAD_CURVE",
        side_a=replace(valid_seam.side_a, curve_id="missing.curve"),
    )
    bad_allowance = replace(valid_seam, seam_id="BAD_ALLOWANCE", allowance_a_m=-0.001)
    bad_gather = replace(valid_seam, seam_id="BAD_GATHER", seam_type="GATHERED_SEAM", gather_ratio=1.0)
    rows = [
        _invalid_probe(document, bad_curve, "UNKNOWN_CURVE"),
        _invalid_probe(document, bad_allowance, "NEGATIVE_ALLOWANCE"),
        _invalid_probe(document, bad_gather, "INVALID_GATHER_RATIO"),
        cycle_failure_probe(graph),
    ]
    payload = {
        "contract": "ConstructionFailureAtomicityReceipt/1",
        "source_pattern_document_sha256": document.to_dict()["document_sha256"],
        "probe_count": len(rows),
        "all_rejected": all(not item["accepted"] for item in rows),
        "all_atomic": all(item["failure_atomicity"] for item in rows),
        "probes": rows,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
