from __future__ import annotations

import json
from pathlib import Path

import pytest

from wuxia_garment_oss.anthropometry.v2.migration import migrate_v1_profile
from wuxia_garment_oss.garments.sleeveless_tunic.pattern_cad.migrate import migrate_reference_tunic
from wuxia_garment_oss.pattern_cad.document.resolver import require_resolved
from wuxia_garment_oss.pattern_cad.expressions.evaluator import evaluate_expression_dag
from wuxia_garment_oss.pattern_cad.persistence.json_store import load_pattern_document, save_pattern_document
from wuxia_garment_oss.pattern_cad.transaction.commands import SetExpressionCommand, SetInputCommand
from wuxia_garment_oss.pattern_cad.transaction.store import PatternDocumentStore

ROOT = Path(__file__).resolve().parents[1]


def _authority() -> tuple[dict, object, object]:
    v1 = json.loads((ROOT / "fixtures/sizing/bodies/REFERENCE.json").read_text())
    package = json.loads(
        (ROOT / "build/tunic_pilot/sizing_cp2/pattern_parameter_packages/STANDARD_M.json").read_text()
    )
    profile, _report = migrate_v1_profile(v1)
    document = migrate_reference_tunic(package, profile)
    return package, profile, document


def test_expression_dag_is_deterministic() -> None:
    expressions = {"half": "body / 2", "finished": "half + ease"}
    first = evaluate_expression_dag({"body": 1.0, "ease": 0.1}, expressions)
    second = evaluate_expression_dag({"body": 1.0, "ease": 0.1}, expressions)
    assert first == second == {"half": 0.5, "finished": 0.6}


def test_expression_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        evaluate_expression_dag({}, {"a": "b + 1", "b": "a + 1"})


def test_reference_migration_matches_cp2_landmarks() -> None:
    package, _profile, document = _authority()
    resolved = require_resolved(document)
    for panel in package["panels"]:
        for name, expected in panel["landmarks"].items():
            actual = resolved["points"][f"{panel['panel_id']}.{name}"]
            assert actual == pytest.approx(expected, abs=1.0e-12)


def test_open_boundaries_remain_open() -> None:
    _package, _profile, document = _authority()
    roles = {
        curve.boundary_role: curve.disposition
        for curve in document.curves.values()
        if curve.panel_id == "bodice_front"
    }
    assert roles["NECKLINE"] == "OPEN"
    assert roles["ARMHOLE_LEFT"] == "OPEN"
    assert roles["ARMHOLE_RIGHT"] == "OPEN"


def test_atomic_failure_preserves_revision_and_hash() -> None:
    _package, _profile, document = _authority()
    store = PatternDocumentStore(document)
    before = store.document.to_dict()
    result = store.apply(SetExpressionCommand("neck_half", "-0.01"))
    after = store.document.to_dict()
    assert result.accepted is False
    assert before["revision"] == after["revision"]
    assert before["document_sha256"] == after["document_sha256"]


def test_success_undo_redo_roundtrip() -> None:
    _package, _profile, document = _authority()
    store = PatternDocumentStore(document)
    initial = store.document.to_dict()["document_sha256"]
    result = store.apply(SetInputCommand("front_chest_ease", 0.056))
    assert result.accepted is True
    edited = store.document.to_dict()["document_sha256"]
    assert edited != initial
    store.undo()
    assert store.document.to_dict()["document_sha256"] == initial
    store.redo()
    assert store.document.to_dict()["document_sha256"] == edited


def test_save_reopen_preserves_hash(tmp_path: Path) -> None:
    _package, _profile, document = _authority()
    store = PatternDocumentStore(document)
    assert store.apply(SetInputCommand("front_chest_ease", 0.056)).accepted
    path = tmp_path / "pattern_document.json"
    saved = save_pattern_document(path, store.document)
    reopened = load_pattern_document(path)
    assert reopened.to_dict()["document_sha256"] == saved


def test_no_geometry_execution_flags() -> None:
    _package, _profile, document = _authority()
    assert document.metadata["triangulation_executed"] is False
    assert document.metadata["warp_simulation_executed"] is False
    assert document.metadata["mesh_scaling"] == "FORBIDDEN"
