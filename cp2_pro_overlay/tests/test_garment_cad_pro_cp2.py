from __future__ import annotations

import json
from pathlib import Path

from wuxia_garment_oss.garments.sleeveless_tunic.pattern_cad.professional import (
    invalid_feature_specs,
    tunic_feature_graph,
    tunic_grade_rule_set,
    tunic_notch_set,
)
from wuxia_garment_oss.pattern_cad.document.model import PatternDocument
from wuxia_garment_oss.pattern_cad.features.compiler import (
    compile_feature_graph,
    failure_atomicity_probes,
)
from wuxia_garment_oss.pattern_cad.features.model import PatternFeatureGraph, PatternFeatureSpec
from wuxia_garment_oss.pattern_cad.grading.resolver import compile_grade_variants


ROOT = Path(__file__).resolve().parents[1]


def _base() -> PatternDocument:
    payload = json.loads(
        (ROOT / "build/pattern_cad_cp1/pattern_document.json").read_text()
    )
    return PatternDocument.from_dict(payload)


def _variants() -> dict[str, dict]:
    base = _base()
    return compile_grade_variants(base, tunic_grade_rule_set(base), tunic_notch_set(base))


def test_arbitrary_n_grading_and_base_immutability() -> None:
    base = _base()
    before = base.to_dict()["document_sha256"]
    variants = compile_grade_variants(base, tunic_grade_rule_set(base), tunic_notch_set(base))
    assert tuple(variants) == ("XXS", "XS", "S", "M", "L", "XL", "XXL")
    assert base.to_dict()["document_sha256"] == before


def test_named_grade_points_move_monotonically() -> None:
    variants = _variants()
    right_underarm = [
        variants[size]["resolved"]["points"]["bodice_front.underarm_right"][0]
        for size in variants
    ]
    skirt_length = [
        -variants[size]["resolved"]["points"]["skirt_front.hem_right"][1]
        for size in variants
    ]
    assert right_underarm == sorted(right_underarm)
    assert skirt_length == sorted(skirt_length)


def test_curve_propagation_is_exact() -> None:
    for variant in _variants().values():
        assert variant["curve_propagation"]["passed"] is True
        assert variant["curve_propagation"]["maximum_curve_propagation_error"] <= 1.0e-12


def test_notch_arc_positions_survive_all_sizes() -> None:
    variants = _variants()
    base = {
        item["notch_id"]: item["arc_fraction"]
        for item in variants["M"]["notches"]
    }
    for variant in variants.values():
        assert {item["notch_id"]: item["arc_fraction"] for item in variant["notches"]} == base


def test_feature_graph_adds_all_feature_types_and_gusset_panel() -> None:
    base = _base()
    compiled = compile_feature_graph(base, tunic_feature_graph())
    assert {
        item["feature"]["feature_type"] for item in compiled["feature_receipts"]
    } == {"DART", "PLEAT", "GATHER", "GUSSET"}
    assert "gusset_underarm" in compiled["final_document"]["panel_ids"]
    assert compiled["final_resolved"]["hard_constraints_pass"] is True


def test_feature_stable_ids_preserve_all_predecessor_ids() -> None:
    compiled = compile_feature_graph(_base(), tunic_feature_graph())
    mapping = compiled["overall_stable_id_mapping"]
    assert mapping["removed"] == []
    assert mapping["preserved_count"] > 0
    assert mapping["added_count"] > 0


def test_each_feature_compiles_independently() -> None:
    base = _base()
    for feature in tunic_feature_graph().features:
        independent = PatternFeatureSpec(
            feature.feature_id,
            feature.feature_type,
            feature.owner_id,
            dict(feature.parameters),
            (),
        )
        result = compile_feature_graph(base, PatternFeatureGraph(feature.feature_id, (independent,)))
        assert len(result["feature_receipts"]) == 1
        assert result["base_document_mutated"] is False


def test_invalid_features_are_rejected_atomically() -> None:
    result = failure_atomicity_probes(_base(), invalid_feature_specs())
    assert result["probe_count"] == 4
    assert result["all_rejected"] is True
    assert result["all_atomic"] is True
    assert all(item["revision_unchanged"] for item in result["probes"])
    assert all(item["document_sha256_unchanged"] for item in result["probes"])


def test_cp2_has_no_meshing_or_simulation_path() -> None:
    compiled = compile_feature_graph(_base(), tunic_feature_graph())
    assert compiled["triangulation_executed"] is False
    assert compiled["warp_simulation_executed"] is False
    for variant in _variants().values():
        metadata = variant["document"]["metadata"]
        assert metadata["triangulation_executed"] is False
        assert metadata["warp_simulation_executed"] is False
        assert metadata["mesh_scaling"] == "FORBIDDEN"
