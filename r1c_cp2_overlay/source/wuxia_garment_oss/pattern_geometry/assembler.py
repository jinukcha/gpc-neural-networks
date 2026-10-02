"""Compile accepted exact 2D components into an immutable pattern assembly package."""
from __future__ import annotations

from wuxia_garment_oss.pattern_assembly.recipe import GarmentAssemblyRecipe
from wuxia_garment_oss.pattern_components.model import canonical_sha256
from wuxia_garment_oss.pattern_components.registry import PatternComponentRegistry
from wuxia_garment_oss.pattern_geometry.model import ComponentGeometryAuthority


def _seam_row(receipt: dict, seam_allowance_m: float, turn_of_cloth_m: float) -> dict:
    return {
        "interface_id": receipt["interface_id"],
        "semantic_role": receipt["semantic_role"],
        "endpoint_a": receipt["endpoint_a"],
        "endpoint_b": receipt["endpoint_b"],
        "length_a_m": receipt["length_a_m"],
        "length_b_m": receipt["length_b_m"],
        "length_ratio": receipt["directed_or_symmetric_ratio"],
        "seam_allowance_a_m": seam_allowance_m,
        "seam_allowance_b_m": seam_allowance_m,
        "turn_of_cloth_m": turn_of_cloth_m if receipt["semantic_role"] in {"SLEEVE_CAP", "COLLAR_ATTACH"} else 0.0,
        "notch_correspondence": receipt["notch_correspondence"],
    }


def _instance_row(recipe: GarmentAssemblyRecipe, geometry: ComponentGeometryAuthority) -> dict:
    instance = recipe.instance_map()[geometry.instance_id]
    payload = geometry.to_dict()
    return {
        "instance_id": geometry.instance_id,
        "component_id": geometry.component_id,
        "component_revision": geometry.component_revision,
        "mirror_source_instance_id": instance.mirror_source_instance_id,
        "geometry_sha256": payload["geometry_sha256"],
        "boundary_count": len(geometry.boundaries),
        "notch_count": len(geometry.notches),
        "internal_line_count": len(geometry.internal_segment_ids),
    }


def compile_assembly(
    registry: PatternComponentRegistry,
    recipe: GarmentAssemblyRecipe,
    geometries: dict[str, ComponentGeometryAuthority],
    interface_receipt: dict,
    parameter_set_sha256: str,
    seam_allowance_m: float,
    turn_of_cloth_m: float,
) -> tuple[dict | None, dict]:
    errors = []
    if not interface_receipt["accepted"]:
        errors.append("INTERFACE_SOLVER_REJECTED")
    missing = [
        instance.instance_id
        for instance in recipe.component_instances
        if instance.instance_id not in geometries
    ]
    errors.extend(f"MISSING_COMPONENT_GEOMETRY:{item}" for item in missing)
    if errors:
        receipt = {
            "contract": "AssemblyCompilationReceipt/1",
            "recipe_id": recipe.recipe_id,
            "accepted": False,
            "status": "REJECTED_ATOMIC",
            "assembled_package_sha256": None,
            "errors": errors,
            "partial_publication_count": 0,
            "triangulation_executed": False,
            "simulation_executed": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        return None, receipt
    selected = [geometries[item.instance_id] for item in recipe.component_instances]
    seams = [
        _seam_row(item, seam_allowance_m, turn_of_cloth_m)
        for item in interface_receipt["interface_receipts"]
    ]
    package = {
        "contract": "AssembledPatternPackage/1",
        "assembly_id": "R1C_CP2_SLEEVED_TUNIC_EXACT_ASSEMBLY",
        "recipe_id": recipe.recipe_id,
        "garment_family": recipe.garment_family,
        "component_registry_sha256": registry.to_dict()["registry_sha256"],
        "recipe_sha256": recipe.to_dict()["recipe_sha256"],
        "resolved_parameter_set_sha256": parameter_set_sha256,
        "interface_solver_receipt_sha256": interface_receipt["receipt_sha256"],
        "component_instances": [_instance_row(recipe, item) for item in selected],
        "seams": seams,
        "component_instance_count": len(selected),
        "seam_count": len(seams),
        "notch_pair_count": sum(len(item["notch_correspondence"]) for item in seams),
        "required_layers": list(recipe.required_layers),
        "completion_modes": list(recipe.allowed_completion_modes),
        "topology_change_policy": recipe.topology_change_policy,
        "geometry_authority": "EXACT_2D_COMPONENTS",
        "triangulation_executed": False,
        "simulation_executed": False,
        "three_dimensional_primitive_fallback": False,
    }
    package["assembled_package_sha256"] = canonical_sha256(package)
    receipt = {
        "contract": "AssemblyCompilationReceipt/1",
        "recipe_id": recipe.recipe_id,
        "accepted": True,
        "status": "COMPILED",
        "assembled_package_sha256": package["assembled_package_sha256"],
        "component_instance_count": len(selected),
        "seam_count": len(seams),
        "notch_pair_count": package["notch_pair_count"],
        "errors": [],
        "partial_publication_count": 0,
        "triangulation_executed": False,
        "simulation_executed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return package, receipt
