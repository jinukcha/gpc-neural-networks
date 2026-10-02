"""Publish canonical arrays and Blender-readable CP4-R1 product geometry."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .body import BodyProfile, build_body_mesh
from .model import ComponentMesh, SeamMap, canonical_sha256, write_json


def publish_materialized_product(
    build: Path,
    meshes: dict[str, ComponentMesh],
    seam_maps: list[SeamMap],
    arrays: dict,
    final_positions: np.ndarray,
    profile: BodyProfile,
) -> dict:
    product_dir = build / "product"
    product_dir.mkdir(parents=True, exist_ok=True)
    body_vertices, body_triangles, body_regions = build_body_mesh(profile)
    components = []
    for instance_id in arrays["component_order"]:
        offset = arrays["offsets"][instance_id]
        mesh = meshes[instance_id]
        count = len(mesh.vertices_3d)
        positions = final_positions[offset : offset + count]
        components.append({
            "instance_id": instance_id,
            "component_id": mesh.component_id,
            "positions_m": positions.tolist(),
            "triangles": mesh.triangles.tolist(),
            "boundaries": {
                key: value.tolist() for key, value in sorted(mesh.boundary_indices.items())
            },
            "metadata": mesh.metadata,
        })
    seam_lines = _seam_lines(meshes, seam_maps, arrays, final_positions)
    product = {
        "contract": "PatternDrivenMaterializedGarment/1",
        "product_id": "R1C_CP4_R1_SET_IN_SLEEVE_TUNIC",
        "coordinate_frame": "RH_Y_UP_METRE",
        "neutral_gray": True,
        "body_visible_review_fixture": True,
        "components": components,
        "body": {
            "positions_m": body_vertices.tolist(),
            "triangles": body_triangles.tolist(),
            "region_ids": body_regions.tolist(),
        },
        "seam_lines": seam_lines,
        "source_pattern_changed": False,
        "post_settle_vertex_repair_count": 0,
    }
    product["product_sha256"] = canonical_sha256(product)
    write_json(product_dir / "materialized_product.json", product)
    np.savez_compressed(
        product_dir / "materialized_product_arrays.npz",
        positions=final_positions.astype(np.float32),
        positions_rest=arrays["positions_rest"].astype(np.float32),
        triangles=arrays["triangles"],
        edges=arrays["edges"],
        arrangement_rest_lengths=arrays["arrangement_rest_lengths"].astype(np.float32),
        seam_pairs=arrays["seam_pairs"],
        fixed_mask=arrays["fixed_mask"],
        component_ids=arrays["component_ids"],
        body_positions=body_vertices.astype(np.float32),
        body_triangles=body_triangles,
    )
    receipt = {
        "contract": "MaterializedProductReceipt/1",
        "product_id": product["product_id"],
        "product_sha256": product["product_sha256"],
        "component_count": len(components),
        "vertex_count": int(len(final_positions)),
        "triangle_count": int(len(arrays["triangles"])),
        "seam_line_count": len(seam_lines),
        "body_vertex_count": int(len(body_vertices)),
        "body_triangle_count": int(len(body_triangles)),
        "glb_pending_blender_export": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(product_dir / "materialized_product_receipt.json", receipt)
    return receipt


def _seam_lines(meshes, seam_maps, arrays, positions):
    lines = []
    for seam in seam_maps:
        offset_a = arrays["offsets"][seam.component_a]
        local = seam.vertex_pairs[:, 0] + offset_a
        lines.append({
            "interface_id": seam.interface_id,
            "positions_m": positions[local].tolist(),
            "orientation": seam.orientation,
        })
    return lines
