"""Public CP5 product compiler and receipt construction."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .correctives import corrective_driver_set
from .measurements import ArmMeasurementSet
from .product_glb import append_sleeve_family


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _affected_vertices(components: list[dict]) -> dict[str, int]:
    result = {item["driver_id"]: 0 for item in corrective_driver_set()["drivers"]}
    for component in components:
        for target in component["corrective_deformation_set"]["targets"]:
            result[target["driver_id"]] += int(target["affected_vertex_count"])
    return result


def compile_sleeved_product(
    base_product: Path,
    target: Path,
    product_id: str,
    product_kind: str,
    skeleton: dict,
    measurements: ArmMeasurementSet,
) -> dict:
    components = append_sleeve_family(base_product, target, product_id, product_kind, skeleton, measurements)
    weight_receipts = [item["skin_weight_field"] for item in components]
    receipt = {
        "contract": "RiggedGarmentProduct/1",
        "product_id": product_id,
        "product_kind": product_kind,
        "path": target.as_posix(),
        "glb_sha256": _sha256(target),
        "canonical_skeleton_sha256": skeleton["skeleton_sha256"],
        "bone_count": int(skeleton["bone_count"]),
        "component_count": len(components),
        "vertex_count": sum(item["vertex_count"] for item in components),
        "triangle_count": sum(item["triangle_count"] for item in components),
        "maximum_weight_sum_error": max(item["maximum_weight_sum_error"] for item in weight_receipts),
        "zero_weight_vertex_count": sum(item["zero_weight_vertex_count"] for item in weight_receipts),
        "corrective_target_names": [item["driver_id"] for item in corrective_driver_set()["drivers"]],
        "corrective_affected_vertices": _affected_vertices(components),
        "components": components,
        "secondary_motion": "NOT_EXECUTED",
        "rig_aware_lod": "NOT_EXECUTED",
        "mesh_scaling": "FORBIDDEN",
    }
    return receipt
