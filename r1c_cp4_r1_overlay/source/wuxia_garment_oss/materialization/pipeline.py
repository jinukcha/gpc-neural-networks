"""End-to-end CP4-R1 materialization pipeline before Blender review."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .arrange import arrange_components
from .body import BodyProfile
from .contracts import cp4_r1_schemas
from .export import publish_materialized_product
from .model import canonical_sha256, write_json
from .qualification import qualify_product
from .rest_metric import compile_rest_metric
from .seams import align_and_relax, compile_seam_maps
from .settle import settle_material
from .triangulation import triangulate_snapshot


def run_materialization(root: Path) -> dict:
    build = root / "build/r1c_cp4_r1"
    build.mkdir(parents=True, exist_ok=True)
    _publish_schemas(root)
    snapshot = _load_snapshot(root)
    meshes, triangulation = triangulate_snapshot(snapshot)
    profile = BodyProfile()
    arrangement = arrange_components(meshes, profile)
    seam_maps, seam_receipt = compile_seam_maps(meshes, snapshot["assembled_package"])
    cap_patch = align_and_relax(meshes, seam_maps)
    arrays, rest_metric = compile_rest_metric(meshes, seam_maps)
    final_positions, settling = settle_material(root, arrays, profile)
    qualification = qualify_product(meshes, seam_maps, arrays, final_positions, settling, profile)
    product = publish_materialized_product(build, meshes, seam_maps, arrays, final_positions, profile)
    _publish_arrays(build, arrays, final_positions)
    receipts = {
        "triangulation_receipt.json": triangulation,
        "avatar_arrangement_receipt.json": arrangement,
        "seam_correspondence_receipt.json": seam_receipt,
        "cap_patch_arrangement_receipt.json": cap_patch,
        "compiled_rest_metric_receipt.json": rest_metric,
        "warp_settling_receipt.json": settling,
        "technical_qualification_receipt.json": qualification,
        "materialized_product_receipt.json": product,
    }
    for name, payload in receipts.items():
        write_json(build / name, payload)
    preliminary = _preliminary_receipt(snapshot, receipts)
    write_json(build / "cp4_r1_preliminary_receipt.json", preliminary)
    write_json(root / "R1C_STATUS.json", preliminary)
    return preliminary


def _publish_schemas(root: Path) -> None:
    for name, schema in cp4_r1_schemas().items():
        write_json(root / "contracts/r1c_cp4_r1" / name, schema)


def _load_snapshot(root: Path) -> dict:
    path = root / "build/r1c_cp3/repaired/repaired_pattern_snapshot.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != "PatternCompletionSnapshot/1":
        raise ValueError("CP4-R1 requires the CP3 committed repaired snapshot")
    return payload


def _publish_arrays(build: Path, arrays: dict, final_positions: np.ndarray) -> None:
    np.savez_compressed(
        build / "compiled_materialization.npz",
        positions_2d=arrays["positions_2d"].astype(np.float32),
        positions_rest=arrays["positions_rest"].astype(np.float32),
        positions_final=final_positions.astype(np.float32),
        triangles=arrays["triangles"],
        edges=arrays["edges"],
        seam_pairs=arrays["seam_pairs"],
        fixed_mask=arrays["fixed_mask"],
        component_ids=arrays["component_ids"],
        arrangement_rest_lengths=arrays["arrangement_rest_lengths"].astype(np.float32),
        pattern_rest_lengths=arrays["pattern_rest_lengths"].astype(np.float32),
    )
    write_json(build / "component_offsets.json", {
        "component_order": arrays["component_order"],
        "offsets": arrays["offsets"],
    })


def _preliminary_receipt(snapshot: dict, receipts: dict) -> dict:
    technical = receipts["technical_qualification_receipt.json"]
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP4_R1",
        "phase": "MATERIALIZED_PENDING_BLENDER_VISUAL_REVIEW",
        "terminal_decision": "PENDING_VISUAL_REVIEW",
        "cp3_snapshot_sha256": snapshot["snapshot_sha256"],
        "component_count": receipts["triangulation_receipt.json"]["component_count"],
        "vertex_count": receipts["triangulation_receipt.json"]["vertex_count"],
        "triangle_count": receipts["triangulation_receipt.json"]["triangle_count"],
        "seam_interface_count": receipts["seam_correspondence_receipt.json"]["interface_count"],
        "orientation_reversed_count": receipts["seam_correspondence_receipt.json"]["reversed_count"],
        "technical_pass": technical["technical_pass"],
        "visual_review": "PENDING",
        "product_acceptance": False,
        "triangulation_executed": True,
        "avatar_arrangement_executed": True,
        "warp_material_settling_executed": True,
        "blender_visual_review_executed": False,
        "post_settle_vertex_repair_count": 0,
        "cp3_predecessor_mutated": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
