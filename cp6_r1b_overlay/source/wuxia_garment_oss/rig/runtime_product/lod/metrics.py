"""Inspect rig, skin, corrective, and secondary-motion continuity across LODs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from wuxia_garment_oss.rig.game_product.glb_skin import _accessor_array, read_glb


CP5_MESH_SUFFIX = "_CP5_COMPONENTS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _primitive_counts(glb, primitive: dict) -> tuple[int, int]:
    position = primitive.get("attributes", {}).get("POSITION")
    vertices = int(glb.document["accessors"][position]["count"]) if position is not None else 0
    indices = primitive.get("indices")
    if indices is not None:
        triangles = int(glb.document["accessors"][indices]["count"]) // 3
    else:
        triangles = vertices // 3
    return vertices, triangles


def _weight_metrics(glb, primitive: dict) -> tuple[float, int, int]:
    accessor = primitive.get("attributes", {}).get("WEIGHTS_0")
    if accessor is None:
        return 0.0, 0, 0
    weights = _accessor_array(glb, accessor).astype(np.float64)
    sums = np.sum(weights, axis=1)
    error = float(np.max(np.abs(sums - 1.0))) if len(sums) else 0.0
    zero = int(np.count_nonzero(sums <= 0.0))
    negative = int(np.count_nonzero(weights < 0.0))
    return error, zero, negative


def _target_names(mesh: dict) -> tuple[str, ...]:
    return tuple(str(item) for item in mesh.get("extras", {}).get("targetNames", []))


def _target_maxima(glb, mesh: dict, names: tuple[str, ...]) -> dict[str, float]:
    maxima = {name: 0.0 for name in names}
    for primitive in mesh.get("primitives", []):
        for index, target in enumerate(primitive.get("targets", [])):
            if index >= len(names) or "POSITION" not in target:
                continue
            values = _accessor_array(glb, target["POSITION"]).astype(np.float64)
            value = float(np.max(np.linalg.norm(values, axis=1))) if len(values) else 0.0
            maxima[names[index]] = max(maxima[names[index]], value)
    return maxima


def inspect_rigged_glb(path: Path) -> dict:
    glb = read_glb(path)
    vertices = triangles = zero_weights = negative_weights = 0
    max_weight_error = 0.0
    primitive_count = 0
    for mesh in glb.document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            primitive_count += 1
            primitive_vertices, primitive_triangles = _primitive_counts(glb, primitive)
            vertices += primitive_vertices
            triangles += primitive_triangles
            error, zero, negative = _weight_metrics(glb, primitive)
            max_weight_error = max(max_weight_error, error)
            zero_weights += zero
            negative_weights += negative
    cp5_meshes = [
        item for item in glb.document.get("meshes", [])
        if str(item.get("name", "")).endswith(CP5_MESH_SUFFIX)
    ]
    if len(cp5_meshes) != 1:
        raise ValueError(f"expected one CP5 component mesh in {path.name}")
    names = _target_names(cp5_meshes[0])
    skins = glb.document.get("skins", [])
    bone_count = len(skins[0].get("joints", [])) if skins else 0
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "file_size_bytes": path.stat().st_size,
        "mesh_count": len(glb.document.get("meshes", [])),
        "primitive_count": primitive_count,
        "vertex_count": vertices,
        "triangle_count": triangles,
        "skin_count": len(skins),
        "bone_count": bone_count,
        "maximum_weight_sum_error": max_weight_error,
        "zero_weight_vertex_count": zero_weights,
        "negative_weight_count": negative_weights,
        "cp5_target_names": list(names),
        "cp5_target_maximum_displacement_m": _target_maxima(glb, cp5_meshes[0], names),
    }


def qualify_lod_set(metrics: tuple[dict, dict, dict]) -> dict:
    lod0, lod1, lod2 = metrics
    target_names = lod0["cp5_target_names"]
    counts_descend = (
        lod0["vertex_count"] > lod1["vertex_count"] > lod2["vertex_count"]
        and lod0["triangle_count"] > lod1["triangle_count"] > lod2["triangle_count"]
    )
    skin_preserved = all(
        item["skin_count"] == 1
        and item["bone_count"] == lod0["bone_count"] == 23
        and item["zero_weight_vertex_count"] == 0
        and item["negative_weight_count"] == 0
        and item["maximum_weight_sum_error"] <= 1.0e-5
        for item in metrics
    )
    target_names_preserved = all(item["cp5_target_names"] == target_names for item in metrics)
    active_targets = [name for name, value in lod0["cp5_target_maximum_displacement_m"].items() if value > 1.0e-8]
    morphs_preserved = all(
        all(item["cp5_target_maximum_displacement_m"].get(name, 0.0) > 1.0e-8 for name in active_targets)
        for item in metrics
    )
    return {
        "contract": "RigAwareLODQualification/1",
        "counts_descend": counts_descend,
        "skin_preserved": skin_preserved,
        "target_names_preserved": target_names_preserved,
        "active_morph_targets_preserved": morphs_preserved,
        "active_target_names": active_targets,
        "lod_metrics": list(metrics),
        "accepted": counts_descend and skin_preserved and target_names_preserved and morphs_preserved,
    }
