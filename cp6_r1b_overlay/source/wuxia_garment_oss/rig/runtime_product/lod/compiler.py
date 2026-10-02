"""Compile deterministic rig-aware LOD variants from a secondary-motion product."""
from __future__ import annotations

from pathlib import Path

from wuxia_garment_oss.rig.game_product.glb_skin import read_glb, write_glb

from .clustering import cluster_primitive
from .metrics import inspect_rigged_glb


LOD_RATIOS = {"LOD0": 1.0, "LOD1": 0.60, "LOD2": 0.30}


def _compile_meshes(glb, ratio: float) -> list[dict]:
    receipts = []
    for mesh_index, mesh in enumerate(glb.document.get("meshes", [])):
        compiled = []
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            result, receipt = cluster_primitive(glb, primitive, ratio)
            compiled.append(result)
            receipt.update({"mesh_index": mesh_index, "primitive_index": primitive_index})
            receipts.append(receipt)
        mesh["primitives"] = compiled
    return receipts


def compile_lod_glb(source: Path, target: Path, lod_id: str) -> dict:
    if lod_id not in LOD_RATIOS:
        raise ValueError(f"unknown LOD identity: {lod_id}")
    glb = read_glb(source)
    ratio = LOD_RATIOS[lod_id]
    primitive_receipts = _compile_meshes(glb, ratio)
    glb.document.setdefault("asset", {})["generator"] = "GARMENT-CAD-PRO-R1B CP6"
    glb.document.setdefault("extras", {})["r1bCp6LOD"] = {
        "contract": "RigAwareLODProduct/1",
        "lodId": lod_id,
        "targetVertexRatio": ratio,
        "skinTransfer": "REPRESENTATIVE_VERTEX_WITH_ALL_ATTRIBUTES",
        "correctiveTransfer": "TARGET_ARRAY_REPRESENTATIVE_CONTINUITY",
        "secondaryMotionTransfer": "TARGET_ARRAY_REPRESENTATIVE_CONTINUITY",
    }
    write_glb(target, glb)
    metrics = inspect_rigged_glb(target)
    changed = [item for item in primitive_receipts if item.get("changed")]
    return {
        "contract": "RigAwareLODProduct/1",
        "lod_id": lod_id,
        "target_ratio": ratio,
        "source_path": source.name,
        "target_path": target.name,
        "changed_primitive_count": len(changed),
        "primitive_receipts": primitive_receipts,
        "metrics": metrics,
    }
