"""Compile feature graphs through the CP1 atomic transaction store."""
from __future__ import annotations

from ..document.model import PatternDocument, canonical_sha256
from ..document.resolver import require_resolved
from ..transaction.store import PatternDocumentStore
from .commands import command_for
from .model import PatternFeatureGraph, PatternFeatureSpec


def _owned_ids(document: PatternDocument) -> set[str]:
    return {
        *(f"panel:{item}" for item in document.panel_ids),
        *(f"point:{item}" for item in document.points),
        *(f"curve:{item}" for item in document.curves),
        *(f"constraint:{item}" for item in document.constraints),
    }


def _mapping(before: set[str], after: set[str], feature_id: str) -> dict:
    preserved = sorted(before & after)
    added = sorted(after - before)
    removed = sorted(before - after)
    return {
        "feature_id": feature_id,
        "preserved": {item: item for item in preserved},
        "added": [{"stable_id": item, "origin": f"FEATURE:{feature_id}"} for item in added],
        "removed": removed,
        "preserved_count": len(preserved),
        "added_count": len(added),
        "removed_count": len(removed),
    }


def compile_feature_graph(base: PatternDocument, graph: PatternFeatureGraph) -> dict:
    graph.validate()
    base_sha = base.to_dict()["document_sha256"]
    store = PatternDocumentStore(base)
    feature_receipts = []
    mappings = []
    for feature in graph.features:
        before_ids = _owned_ids(store.document)
        result = store.apply(command_for(feature))
        if not result.accepted:
            raise ValueError(f"feature graph rejected: {feature.feature_id}: {result.error}")
        after_ids = _owned_ids(store.document)
        mappings.append(_mapping(before_ids, after_ids, feature.feature_id))
        feature_receipts.append({**result.to_dict(), "feature": feature.to_dict()})
    final_document = store.document
    final_resolved = require_resolved(final_document)
    if base.to_dict()["document_sha256"] != base_sha:
        raise AssertionError("feature graph mutated its predecessor")
    overall = _mapping(_owned_ids(base), _owned_ids(final_document), graph.feature_graph_id)
    payload = {
        "contract": "CompiledPatternFeatureGraph/1",
        "feature_graph": graph.to_dict(),
        "source_document_sha256": base_sha,
        "final_document": final_document.to_dict(),
        "final_resolved": final_resolved,
        "feature_receipts": feature_receipts,
        "stable_id_mappings": mappings,
        "overall_stable_id_mapping": overall,
        "base_document_mutated": False,
        "triangulation_executed": False,
        "warp_simulation_executed": False,
    }
    payload["compiled_feature_graph_sha256"] = canonical_sha256(payload)
    return payload


def _failure_row(base: PatternDocument, spec: PatternFeatureSpec) -> dict:
    store = PatternDocumentStore(base)
    before = store.document.to_dict()
    result = store.apply(command_for(spec))
    after = store.document.to_dict()
    if result.accepted:
        raise AssertionError(f"invalid fixture unexpectedly accepted: {spec.feature_id}")
    if before["document_sha256"] != after["document_sha256"]:
        raise AssertionError(f"rejected feature mutated document: {spec.feature_id}")
    return {
        "feature": spec.to_dict(),
        "transaction": result.to_dict(),
        "failure_atomicity": True,
        "revision_unchanged": before["revision"] == after["revision"],
        "document_sha256_unchanged": before["document_sha256"] == after["document_sha256"],
    }


def failure_atomicity_probes(
    base: PatternDocument,
    invalid_specs: tuple[PatternFeatureSpec, ...],
) -> dict:
    rows = [_failure_row(base, spec) for spec in invalid_specs]
    payload = {
        "contract": "PatternFeatureFailureAtomicityReceipt/1",
        "source_document_sha256": base.to_dict()["document_sha256"],
        "probe_count": len(rows),
        "all_rejected": all(not row["transaction"]["accepted"] for row in rows),
        "all_atomic": all(row["failure_atomicity"] for row in rows),
        "probes": rows,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
