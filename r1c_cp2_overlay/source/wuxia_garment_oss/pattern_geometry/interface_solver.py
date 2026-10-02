"""Physical boundary, ease, and notch compatibility solver for CP2."""
from __future__ import annotations

from wuxia_garment_oss.pattern_assembly.compiler import evaluate_recipe
from wuxia_garment_oss.pattern_assembly.interface import ComponentInterfaceSpec
from wuxia_garment_oss.pattern_assembly.recipe import GarmentAssemblyRecipe
from wuxia_garment_oss.pattern_components.model import canonical_sha256
from wuxia_garment_oss.pattern_components.registry import PatternComponentRegistry
from wuxia_garment_oss.pattern_geometry.curve import boundary_length
from wuxia_garment_oss.pattern_geometry.model import ComponentGeometryAuthority


_NOTCH_TOLERANCE = 0.025


def _geometry_endpoint(spec, endpoint, geometries):
    geometry = geometries.get(endpoint.component_instance_id)
    if geometry is None:
        raise KeyError(f"missing geometry: {endpoint.component_instance_id}")
    boundary = geometry.boundary(endpoint.boundary_id)
    return geometry, boundary


def _length_result(spec, length_a: float, length_b: float) -> tuple[bool, float, str | None]:
    if min(length_a, length_b) <= 0.0:
        return False, float("inf"), "ZERO_LENGTH_BOUNDARY"
    directed = length_a / length_b
    ratio = max(length_a, length_b) / min(length_a, length_b)
    if spec.length_policy == "BOUNDED_EASE":
        passed = spec.ratio_min <= directed <= spec.ratio_max
        return passed, directed, None if passed else "EASE_RATIO_OUT_OF_RANGE"
    if spec.length_policy == "GATHERED":
        passed = spec.ratio_min <= directed <= spec.ratio_max
        return passed, directed, None if passed else "GATHER_RATIO_OUT_OF_RANGE"
    passed = ratio <= spec.ratio_max
    return passed, ratio, None if passed else "BOUNDARY_LENGTH_MISMATCH"


def _boundary_notches(geometry, boundary_id: str) -> dict[str, float]:
    return {
        notch.semantic_role: notch.normalized_arc
        for notch in geometry.notches
        if notch.boundary_id == boundary_id
    }


def _notch_result(spec, geometry_a, geometry_b) -> tuple[bool, list[dict], list[str]]:
    if spec.notch_policy == "NONE":
        return True, [], []
    left = _boundary_notches(geometry_a, spec.endpoint_a.boundary_id)
    right = _boundary_notches(geometry_b, spec.endpoint_b.boundary_id)
    common = sorted(set(left) & set(right))
    if not common:
        return False, [], ["MISSING_COMMON_NOTCH_ROLE"]
    rows, errors = [], []
    for role in common:
        first = left[role]
        second = 1.0 - right[role] if spec.orientation_relation == "OPPOSED" else right[role]
        delta = abs(first - second)
        passed = delta <= _NOTCH_TOLERANCE
        rows.append({
            "semantic_role": role,
            "endpoint_a_normalized": first,
            "endpoint_b_aligned_normalized": second,
            "delta": delta,
            "passed": passed,
        })
        if not passed:
            errors.append(f"NOTCH_CORRESPONDENCE_MISMATCH:{role}")
    return not errors, rows, errors


def _interface_receipt(spec, geometries) -> dict:
    geometry_a, boundary_a = _geometry_endpoint(spec, spec.endpoint_a, geometries)
    geometry_b, boundary_b = _geometry_endpoint(spec, spec.endpoint_b, geometries)
    length_a = boundary_length(boundary_a, geometry_a.segment_map())
    length_b = boundary_length(boundary_b, geometry_b.segment_map())
    length_pass, ratio, length_error = _length_result(spec, length_a, length_b)
    notch_pass, notch_rows, notch_errors = _notch_result(spec, geometry_a, geometry_b)
    errors = ([] if length_error is None else [length_error]) + notch_errors
    payload = {
        "contract": "InterfaceCompatibilityReceipt/1",
        "interface_id": spec.interface_id,
        "semantic_role": spec.semantic_role,
        "endpoint_a": spec.endpoint_a.to_dict(),
        "endpoint_b": spec.endpoint_b.to_dict(),
        "length_policy": spec.length_policy,
        "length_a_m": length_a,
        "length_b_m": length_b,
        "directed_or_symmetric_ratio": ratio,
        "ratio_min": spec.ratio_min,
        "ratio_max": spec.ratio_max,
        "length_pass": length_pass,
        "notch_policy": spec.notch_policy,
        "notch_correspondence": notch_rows,
        "notch_pass": notch_pass,
        "accepted": length_pass and notch_pass,
        "errors": errors,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def solve_interfaces(
    registry: PatternComponentRegistry,
    recipe: GarmentAssemblyRecipe,
    interfaces: tuple[ComponentInterfaceSpec, ...],
    geometries: dict[str, ComponentGeometryAuthority],
) -> dict:
    contract_admission = evaluate_recipe(registry, recipe, interfaces)
    selected = {item.interface_id: item for item in interfaces}
    receipts, errors = [], list(contract_admission["errors"])
    if contract_admission["accepted"]:
        for interface_id in recipe.interface_ids:
            receipt = _interface_receipt(selected[interface_id], geometries)
            receipts.append(receipt)
            errors.extend(f"{interface_id}:{item}" for item in receipt["errors"])
    accepted = contract_admission["accepted"] and not errors
    payload = {
        "contract": "InterfaceSolverReceipt/1",
        "recipe_id": recipe.recipe_id,
        "accepted": accepted,
        "contract_admission_receipt_sha256": contract_admission["receipt_sha256"],
        "interface_receipts": receipts,
        "interface_count": len(receipts),
        "accepted_interface_count": sum(item["accepted"] for item in receipts),
        "errors": sorted(set(errors)),
        "partial_publication_count": 0 if accepted else 0,
        "triangulation_executed": False,
        "simulation_executed": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
