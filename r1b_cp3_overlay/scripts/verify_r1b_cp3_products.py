#!/usr/bin/env python3
"""Fresh-process verification of skinned GLB products."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.rig.game_product.glb_skin import BONES, _accessor_array, read_glb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def verify_product(path: Path, expected_primitives: int) -> dict:
    glb = read_glb(path)
    skins = glb.document.get("skins", [])
    if len(skins) != 1:
        raise AssertionError(f"expected one skin: {path}")
    skin = skins[0]
    if len(skin["joints"]) != len(BONES):
        raise AssertionError("canonical joint count mismatch")
    inverse = _accessor_array(glb, skin["inverseBindMatrices"])
    if inverse.shape != (len(BONES), 16) or not np.all(np.isfinite(inverse)):
        raise AssertionError("invalid inverse bind matrices")
    primitive_count = 0
    vertex_count = 0
    maximum_error = 0.0
    zero_vertices = 0
    morph_counts = []
    for mesh in glb.document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "JOINTS_0" not in attributes or "WEIGHTS_0" not in attributes:
                raise AssertionError("missing skin attributes")
            joints = _accessor_array(glb, attributes["JOINTS_0"]).astype(np.int64)
            weights = _accessor_array(glb, attributes["WEIGHTS_0"]).astype(np.float64)
            if np.any(joints < 0) or np.any(joints >= len(BONES)):
                raise AssertionError("joint index out of range")
            if np.any(weights < -1.0e-8) or not np.all(np.isfinite(weights)):
                raise AssertionError("invalid skin weights")
            sums = np.sum(weights, axis=1)
            maximum_error = max(maximum_error, float(np.max(np.abs(sums - 1.0))))
            zero_vertices += int(np.count_nonzero(sums <= 0.0))
            vertex_count += len(joints)
            primitive_count += 1
            morph_counts.append(len(primitive.get("targets", [])))
    if primitive_count != expected_primitives:
        raise AssertionError((path.name, primitive_count, expected_primitives))
    mesh_nodes = [node for node in glb.document.get("nodes", []) if "mesh" in node]
    if not mesh_nodes or any(node.get("skin") != 0 for node in mesh_nodes):
        raise AssertionError("mesh nodes are not bound to canonical skin")
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "skin_count": len(skins),
        "joint_count": len(skin["joints"]),
        "inverse_bind_matrix_count": len(inverse),
        "primitive_count": primitive_count,
        "vertex_count": vertex_count,
        "mesh_node_count": len(mesh_nodes),
        "morph_target_counts": morph_counts,
        "maximum_weight_sum_error": maximum_error,
        "zero_weight_vertex_count": zero_vertices,
        "fresh_reopen_pass": maximum_error <= 2.0e-6 and zero_vertices == 0,
    }


def main() -> int:
    root = parse_args().root.resolve()
    build = root / "build/rig_cp3"
    products = build / "products"
    receipt = {
        "contract": "RiggedGLBFreshReopenReceipt/1",
        "tunic": verify_product(products / "feature_complete_tunic_rigged.glb", 14),
        "trousers": verify_product(products / "trousers_rigged.glb", 3),
    }
    receipt["fresh_process_pass"] = receipt["tunic"]["fresh_reopen_pass"] and receipt["trousers"]["fresh_reopen_pass"]
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    target = build / "glb_fresh_reopen_receipt.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
