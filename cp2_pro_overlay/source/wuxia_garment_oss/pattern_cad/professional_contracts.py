"""JSON Schema publication for CP2 professional pattern authorities."""
from __future__ import annotations


def _schema(title: str, required: list[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "required": ["contract", *required],
        "properties": {"contract": {"const": title}, **properties},
        "additionalProperties": True,
    }


def professional_pattern_schemas() -> dict[str, dict]:
    identity = {"type": "string", "minLength": 1}
    return {
        "grade_rule_set_v2.schema.json": _schema(
            "GradeRuleSet/2",
            ["grade_rule_set_id", "base_size_id", "ordered_size_ids", "sizes", "grade_points"],
            {
                "grade_rule_set_id": identity,
                "base_size_id": identity,
                "ordered_size_ids": {"type": "array", "minItems": 1},
                "sizes": {"type": "array", "minItems": 1},
                "grade_points": {"type": "array", "minItems": 1},
            },
        ),
        "pattern_notch_set.schema.json": _schema(
            "PatternNotchSet/1",
            ["notch_set_id", "notches"],
            {"notch_set_id": identity, "notches": {"type": "array", "minItems": 1}},
        ),
        "pattern_feature_graph.schema.json": _schema(
            "PatternFeatureGraph/1",
            ["feature_graph_id", "features"],
            {"feature_graph_id": identity, "features": {"type": "array", "minItems": 1}},
        ),
        "stable_id_mapping.schema.json": _schema(
            "StableIdMapping/1",
            ["source_document_sha256", "preserved", "added", "removed"],
            {
                "source_document_sha256": identity,
                "preserved": {"type": "object"},
                "added": {"type": "array"},
                "removed": {"type": "array"},
            },
        ),
    }
