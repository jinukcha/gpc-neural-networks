from __future__ import annotations

import json
from pathlib import Path

from wuxia_garment_oss.proportions.model.quantity import convert_to_si
from wuxia_garment_oss.proportions.resolution.receipt import verify_embedded_hash
from wuxia_garment_oss.proportions.resolution.resolver import resolve_parameter_set
from wuxia_garment_oss.r1c_cp1_fixtures import canonical_context, canonical_definitions, rejection_fixtures


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/r1c_cp1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_units_convert_to_si() -> None:
    assert convert_to_si(12.0, "mm", "LENGTH") == 0.012
    assert abs(convert_to_si(12.0, "deg", "ANGLE") - 0.20943951023931953) < 1.0e-12
    assert convert_to_si(265.0, "g/m2", "MASS_PER_AREA") == 0.265


def test_canonical_resolution_supports_all_modes_and_scopes() -> None:
    resolved, receipt = resolve_parameter_set(
        canonical_definitions(), canonical_context(), "R1C_CP1_CANONICAL_RESOLUTION"
    )
    assert resolved is not None and receipt["accepted"]
    assert {item["mode"] for item in resolved["parameters"]} == {"ABSOLUTE", "RELATIVE", "AUTO_DERIVED"}
    scopes = {
        path.split(".", 1)[0]
        for item in resolved["parameters"]
        for path in item["reference_paths"]
        if not path.startswith("param.")
    }
    assert scopes == {"body", "block", "component", "boundary", "material"}


def test_safe_clamp_and_dependency_order_are_published() -> None:
    resolved, _ = resolve_parameter_set(
        canonical_definitions(), canonical_context(), "R1C_CP1_CANONICAL_RESOLUTION"
    )
    assert resolved is not None
    by_id = {item["parameter_id"]: item for item in resolved["parameters"]}
    assert by_id["cap_ease"]["bounds"]["action"] == "CLAMPED"
    assert by_id["cap_ease"]["resolved_value_si"] == 0.028
    order = resolved["resolution_order"]
    assert order.index("pocket_area") > order.index("pocket_width")
    assert order.index("pocket_area") > order.index("pocket_height")


def test_rejection_fixtures_are_atomic() -> None:
    expected = {
        "missing_reference": "MISSING_REFERENCE",
        "quantity_mismatch": "QUANTITY_MISMATCH",
        "expression_quantity_mismatch": "QUANTITY_MISMATCH",
        "unit_mismatch": "UNIT_MISMATCH",
        "dependency_cycle": "CYCLE_DETECTED",
        "non_finite_ratio": "NON_FINITE_RATIO",
        "hard_bound": "BOUND_REJECTED",
        "alternate_component": "ALTERNATE_COMPONENT_REQUIRED",
    }
    for name, definitions in rejection_fixtures().items():
        resolved, receipt = resolve_parameter_set(definitions, canonical_context(), f"TEST_{name}")
        assert resolved is None
        assert receipt["status"] == expected[name]
        assert receipt["partial_publication_count"] == 0


def test_resolution_is_deterministic() -> None:
    first, first_receipt = resolve_parameter_set(
        canonical_definitions(), canonical_context(), "R1C_CP1_CANONICAL_RESOLUTION"
    )
    second, second_receipt = resolve_parameter_set(
        canonical_definitions(), canonical_context(), "R1C_CP1_CANONICAL_RESOLUTION"
    )
    assert first is not None and second is not None
    assert first["resolved_set_sha256"] == second["resolved_set_sha256"]
    assert first_receipt["receipt_sha256"] == second_receipt["receipt_sha256"]


def test_published_terminal_products() -> None:
    receipt = _load(BUILD / "cp1_receipt.json")
    resolved = _load(BUILD / "resolved_parameter_set.json")
    reopen = _load(BUILD / "fresh_process_reopen_receipt.json")
    assert receipt["terminal_decision"] == "CP1_COMPLETE_RATIO_PARAMETER_ENGINE"
    assert receipt["cp1_acceptance"] is True
    assert receipt["parameter_mode_count"] == 3
    assert receipt["reference_scope_count"] == 5
    assert reopen["fresh_process_reopen_pass"] is True
    assert reopen["deterministic_rerun_pass"] is True
    assert verify_embedded_hash(resolved, "resolved_set_sha256")
