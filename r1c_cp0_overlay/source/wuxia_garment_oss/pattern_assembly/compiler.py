"""Contract-only garment recipe admission; no geometry or simulation is executed."""
from __future__ import annotations

from collections import Counter

from wuxia_garment_oss.pattern_components.model import canonical_sha256
from wuxia_garment_oss.pattern_components.registry import PatternComponentRegistry

from .interface import ComponentInterfaceSpec, InterfaceEndpoint
from .recipe import GarmentAssemblyRecipe


def _endpoint_key(endpoint: InterfaceEndpoint) -> str:
    return f"{endpoint.component_instance_id}.{endpoint.boundary_id}"


def _interface_errors(
    registry: PatternComponentRegistry,
    recipe: GarmentAssemblyRecipe,
    spec: ComponentInterfaceSpec,
) -> list[str]:
    errors: list[str] = []
    instances = recipe.instance_map()
    resolved = []
    for endpoint in (spec.endpoint_a, spec.endpoint_b):
        instance = instances.get(endpoint.component_instance_id)
        if instance is None:
            errors.append(f"UNKNOWN_COMPONENT_INSTANCE:{endpoint.component_instance_id}")
            continue
        try:
            boundary = registry.get(instance.component_id).boundary(endpoint.boundary_id)
        except KeyError:
            errors.append(f"UNKNOWN_BOUNDARY:{_endpoint_key(endpoint)}")
            continue
        resolved.append(boundary)
    if len(resolved) != 2:
        return errors
    first, second = resolved
    if first.disposition != "SEWN" or second.disposition != "SEWN":
        errors.append(f"NON_SEWN_BOUNDARY_USED:{spec.interface_id}")
    if first.interface_family != second.interface_family:
        errors.append(f"INTERFACE_FAMILY_MISMATCH:{spec.interface_id}")
    if spec.orientation_relation == "OPPOSED" and first.orientation == second.orientation:
        errors.append(f"ORIENTATION_MISMATCH:{spec.interface_id}")
    if spec.orientation_relation == "SAME_FOR_FOLD" and first.orientation != second.orientation:
        errors.append(f"FOLD_ORIENTATION_MISMATCH:{spec.interface_id}")
    if spec.notch_policy == "EXACT_ID" and set(first.notch_ids) != set(second.notch_ids):
        errors.append(f"NOTCH_ID_MISMATCH:{spec.interface_id}")
    return errors


def _required_sewn_boundaries(
    registry: PatternComponentRegistry,
    recipe: GarmentAssemblyRecipe,
) -> set[str]:
    required: set[str] = set()
    for instance in recipe.component_instances:
        component = registry.get(instance.component_id)
        for boundary in component.boundaries:
            if boundary.disposition == "SEWN":
                required.add(f"{instance.instance_id}.{boundary.boundary_id}")
    return required


def evaluate_recipe(
    registry: PatternComponentRegistry,
    recipe: GarmentAssemblyRecipe,
    interfaces: tuple[ComponentInterfaceSpec, ...],
) -> dict:
    registry.validate()
    recipe.validate()
    interface_map = {item.interface_id: item for item in interfaces}
    errors: list[str] = []
    if len(interface_map) != len(interfaces):
        errors.append("DUPLICATE_INTERFACE_ID")
    selected: list[ComponentInterfaceSpec] = []
    for interface_id in recipe.interface_ids:
        spec = interface_map.get(interface_id)
        if spec is None:
            errors.append(f"UNKNOWN_INTERFACE:{interface_id}")
            continue
        spec.validate()
        selected.append(spec)
        errors.extend(_interface_errors(registry, recipe, spec))
    endpoint_counts = Counter(
        _endpoint_key(endpoint)
        for spec in selected
        for endpoint in (spec.endpoint_a, spec.endpoint_b)
    )
    required = _required_sewn_boundaries(registry, recipe)
    missing = sorted(required - set(endpoint_counts))
    duplicated = sorted(key for key, count in endpoint_counts.items() if count > 1)
    extra = sorted(set(endpoint_counts) - required)
    errors.extend(f"UNBOUND_SEWN_BOUNDARY:{item}" for item in missing)
    errors.extend(f"BOUNDARY_USED_MORE_THAN_ONCE:{item}" for item in duplicated)
    errors.extend(f"UNEXPECTED_INTERFACE_BOUNDARY:{item}" for item in extra)
    accepted = not errors
    payload = {
        "contract": "AssemblyAdmissionReceipt/1",
        "recipe_id": recipe.recipe_id,
        "garment_family": recipe.garment_family,
        "accepted": accepted,
        "component_instance_count": len(recipe.component_instances),
        "interface_count": len(selected),
        "required_sewn_boundary_count": len(required),
        "bound_sewn_boundary_count": len(required & set(endpoint_counts)),
        "unbound_sewn_boundaries": missing,
        "duplicated_boundary_bindings": duplicated,
        "unexpected_interface_boundaries": extra,
        "errors": sorted(set(errors)),
        "geometry_executed": False,
        "simulation_executed": False,
        "topology_change_policy": recipe.topology_change_policy,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
