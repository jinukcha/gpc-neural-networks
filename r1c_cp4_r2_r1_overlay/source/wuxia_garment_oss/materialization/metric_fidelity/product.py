"""Publish a repaired product without mutating the accepted CP4-R1 owner."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from ..model import canonical_sha256, write_json


def _replace_components(product: dict, arrays: dict, positions: np.ndarray) -> list[dict]:
    components = []
    for source in product["components"]:
        item = deepcopy(source)
        instance_id = item["instance_id"]
        offset = int(arrays["offsets"][instance_id])
        count = len(item["positions_m"])
        item["positions_m"] = positions[offset : offset + count].tolist()
        components.append(item)
    return components


def _replace_seams(product: dict, arrays: dict, positions: np.ndarray) -> list[dict]:
    result = []
    for source in product["seam_lines"]:
        item = deepcopy(source)
        indices = arrays["seam_line_indices"][item["interface_id"]]
        item["positions_m"] = positions[indices].tolist()
        result.append(item)
    return result


def publish_product(build: Path, baseline: dict, arrays: dict, final_positions: np.ndarray) -> dict:
    target = build / "product"
    target.mkdir(parents=True, exist_ok=True)
    product = deepcopy(baseline)
    product["product_id"] = "R1C_CP4_R2_R1_METRIC_FIDELITY_TUNIC"
    product["components"] = _replace_components(product, arrays, final_positions)
    product["seam_lines"] = _replace_seams(product, arrays, final_positions)
    product["source_cp4_r1_product_sha256"] = baseline["product_sha256"]
    product["source_pattern_changed"] = False
    product["post_settle_vertex_repair_count"] = 0
    product.pop("product_sha256", None)
    product["product_sha256"] = canonical_sha256(product)
    write_json(target / "materialized_product.json", product)
    np.savez_compressed(
        target / "materialized_product_arrays.npz",
        positions=final_positions.astype(np.float32),
        positions_rest=arrays["positions_rest"].astype(np.float32),
        triangles=arrays["triangles"], edges=arrays["edges"],
        pattern_rest_lengths=arrays["pattern_rest_lengths"].astype(np.float32),
        arrangement_rest_lengths=arrays["arrangement_rest_lengths"].astype(np.float32),
        seam_pairs=arrays["seam_pairs"], fixed_mask=arrays["fixed_mask"],
        component_ids=arrays["component_ids"],
    )
    receipt = {
        "contract": "MetricFidelityMaterializedProductReceipt/1",
        "product_id": product["product_id"],
        "product_sha256": product["product_sha256"],
        "source_cp4_r1_product_sha256": baseline["product_sha256"],
        "component_count": len(product["components"]),
        "vertex_count": int(len(final_positions)),
        "triangle_count": int(len(arrays["triangles"])),
        "seam_line_count": len(product["seam_lines"]),
        "glb_pending_blender_export": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(target / "materialized_product_receipt.json", receipt)
    return receipt
