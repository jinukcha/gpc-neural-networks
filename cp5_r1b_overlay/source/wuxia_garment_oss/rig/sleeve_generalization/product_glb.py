"""Append CP5 sleeve-family components to an immutable CP3 rigged GLB."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wuxia_garment_oss.rig.game_product.glb_skin import GLB, _append_accessor, read_glb, write_glb

from .construction import ConstructedPrimitive, build_robe_skirt, build_sleeve
from .correctives import TARGET_NAMES, sleeve_correctives, zero_correctives
from .measurements import ArmMeasurementSet
from .weights import robe_skirt_weight_field, sleeve_weight_field


def _joint_map(skeleton: dict) -> dict[str, np.ndarray]:
    source = skeleton.get("global_joint_positions")
    if not source:
        source = {item["semantic_id"]: item["global_translation"] for item in skeleton["bones"]}
    return {name: np.asarray(value, dtype=np.float64) for name, value in source.items()}


def _append_material(glb: GLB) -> int:
    glb.document.setdefault("materials", []).append({
        "name": "CP5_NEUTRAL_GRAY",
        "doubleSided": True,
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.58, 0.58, 0.58, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.88,
        },
    })
    return len(glb.document["materials"]) - 1


def _primitive_document(glb, geometry, joints, weights, targets, material_index):
    attributes = {
        "POSITION": _append_accessor(glb, geometry.positions, 5126, "VEC3", 34962, True),
        "NORMAL": _append_accessor(glb, geometry.normals, 5126, "VEC3", 34962),
        "TEXCOORD_0": _append_accessor(glb, geometry.texcoords, 5126, "VEC2", 34962),
        "JOINTS_0": _append_accessor(glb, joints, 5123, "VEC4", 34962),
        "WEIGHTS_0": _append_accessor(glb, weights, 5126, "VEC4", 34962),
    }
    target_docs = [
        {"POSITION": _append_accessor(glb, values, 5126, "VEC3", 34962)}
        for values in targets
    ]
    return {
        "attributes": attributes,
        "indices": _append_accessor(glb, geometry.indices.reshape(-1), 5125, "SCALAR", 34963),
        "material": material_index,
        "mode": 4,
        "targets": target_docs,
        "extras": {"componentName": geometry.name, "construction": geometry.metadata},
    }


def _component_receipt(geometry, weight_receipt, corrective_receipt):
    return {
        "component_id": geometry.name,
        "vertex_count": int(len(geometry.positions)),
        "triangle_count": int(len(geometry.indices)),
        "construction": geometry.metadata,
        "skin_weight_field": weight_receipt,
        "corrective_deformation_set": corrective_receipt,
    }


def _compile_component(glb: GLB, geometry: ConstructedPrimitive, material_index: int, skirt: bool):
    if skirt:
        joints, weights, weight_receipt = robe_skirt_weight_field(geometry)
        targets, corrective_receipt = zero_correctives(geometry)
    else:
        joints, weights, weight_receipt = sleeve_weight_field(geometry)
        targets, corrective_receipt = sleeve_correctives(geometry)
    document = _primitive_document(glb, geometry, joints, weights, targets, material_index)
    return document, _component_receipt(geometry, weight_receipt, corrective_receipt)


def _append_mesh_node(glb: GLB, product_id: str, primitives: list[dict]) -> None:
    mesh = {
        "name": f"{product_id}_CP5_COMPONENTS",
        "primitives": primitives,
        "weights": [0.0] * len(TARGET_NAMES),
        "extras": {"targetNames": list(TARGET_NAMES), "ownerKernel": "sleeve_generalization"},
    }
    glb.document.setdefault("meshes", []).append(mesh)
    node = {
        "name": f"{product_id}_CP5_COMPONENT_NODE",
        "mesh": len(glb.document["meshes"]) - 1,
        "skin": 0,
        "extras": {"riggedGarmentProduct": "RiggedGarmentProduct/1", "cp5Product": product_id},
    }
    glb.document.setdefault("nodes", []).append(node)
    scene_index = int(glb.document.get("scene", 0))
    glb.document.setdefault("scenes", [{"nodes": []}])[scene_index].setdefault("nodes", []).append(len(glb.document["nodes"]) - 1)


def append_sleeve_family(
    base_product: Path,
    target: Path,
    product_id: str,
    product_kind: str,
    skeleton: dict,
    measurements: ArmMeasurementSet,
) -> list[dict]:
    if product_kind not in {"SLEEVED_TUNIC", "STRAIGHT_SLEEVE_ROBE"}:
        raise ValueError(f"unsupported product kind: {product_kind}")
    glb = read_glb(base_product)
    if len(glb.document.get("skins", [])) != 1:
        raise ValueError("CP5 requires the immutable CP3 canonical skin")
    joint_map = _joint_map(skeleton)
    material_index = _append_material(glb)
    style = "STRAIGHT_ROBE" if product_kind == "STRAIGHT_SLEEVE_ROBE" else "FITTED_TUNIC"
    geometries = [
        build_sleeve("LEFT", joint_map, measurements, style),
        build_sleeve("RIGHT", joint_map, measurements, style),
    ]
    if product_kind == "STRAIGHT_SLEEVE_ROBE":
        geometries.append(build_robe_skirt(joint_map))
    primitives, receipts = [], []
    for geometry in geometries:
        primitive, receipt = _compile_component(glb, geometry, material_index, geometry.name.endswith("ROBE_SKIRT"))
        primitives.append(primitive)
        receipts.append(receipt)
    _append_mesh_node(glb, product_id, primitives)
    glb.document.setdefault("asset", {})["generator"] = "GARMENT-CAD-PRO-R1B CP5"
    glb.document.setdefault("extras", {})["r1bCp5"] = {
        "productId": product_id,
        "productKind": product_kind,
        "measurementContract": "ArmMeasurementSet/1",
        "correctiveDrivers": list(TARGET_NAMES),
        "secondaryMotion": "NOT_EXECUTED",
        "rigAwareLod": "NOT_EXECUTED",
    }
    write_glb(target, glb)
    return receipts
