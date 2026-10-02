"""CP4-R2-R1 metric decomposition, repair, resettle, and candidate publication."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..body import BodyProfile
from ..model import canonical_sha256, write_json
from ..qualification import _edge_strain, _geometry_metrics, _penetration_metrics
from ..settle import settle_material
from .classify import decompose_metric
from .product import publish_product
from .repair import repair_arrangement


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seam_line_indices(root: Path, seam_pairs: np.ndarray) -> dict[str, np.ndarray]:
    receipt = _load_json(root / "build/r1c_cp4_r1/seam_correspondence_receipt.json")
    result, cursor = {}, 0
    for item in receipt["interfaces"]:
        count = int(item["pair_count"])
        result[item["interface_id"]] = seam_pairs[cursor : cursor + count, 0].copy()
        cursor += count
    if cursor != len(seam_pairs):
        raise ValueError("seam-pair concatenation does not match the correspondence receipt")
    return result


def _load_arrays(root: Path) -> tuple[dict, dict, dict]:
    build = root / "build/r1c_cp4_r1"
    with np.load(build / "compiled_materialization.npz", allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    offsets = _load_json(build / "component_offsets.json")
    arrays["component_order"] = list(offsets["component_order"])
    arrays["offsets"] = {key: int(value) for key, value in offsets["offsets"].items()}
    arrays["seam_line_indices"] = _seam_line_indices(root, arrays["seam_pairs"])
    product = _load_json(build / "product/materialized_product.json")
    return arrays, product, arrays["offsets"]


def _seam_metrics(arrays: dict, positions: np.ndarray) -> dict:
    pairs = arrays["seam_pairs"]
    gaps = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    return {
        "mean_m": float(gaps.mean()),
        "p95_m": float(np.quantile(gaps, 0.95)),
        "maximum_m": float(gaps.max()),
    }


def _qualification(arrays: dict, positions: np.ndarray, settling: dict, profile: BodyProfile) -> dict:
    seam = _seam_metrics(arrays, positions)
    strain = _edge_strain(arrays, positions)
    geometry = _geometry_metrics(arrays, positions)
    penetration = _penetration_metrics(arrays, positions, profile)
    gates = {
        "seam_p95": seam["p95_m"] <= 0.004,
        "edge_strain_p95": strain["p95"] <= 0.080,
        "edge_strain_p99": strain["p99"] <= 0.180,
        "body_penetration_p99": penetration["p99_m"] <= 0.001,
        "non_finite_vertex": geometry["non_finite_vertex_count"] == 0,
        "degenerate_triangle": geometry["degenerate_triangle_count"] == 0,
        "normal_inversion": geometry["normal_inversion_count"] == 0,
        "stable_tail": settling["tail_peak_frame_displacement_m"] <= 0.0015,
        "post_settle_vertex_repair": settling["post_settle_vertex_repair_count"] == 0,
    }
    payload = {
        "contract": "CP4R2R1TechnicalQualificationReceipt/1",
        "seam": seam,
        "edge_strain": strain,
        "geometry": geometry,
        "body_penetration": penetration,
        "settling_tail": {
            "peak_m": settling["tail_peak_frame_displacement_m"],
            "mean_m": settling["tail_mean_frame_displacement_m"],
        },
        "gates": gates,
        "technical_pass": all(gates.values()),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _save_arrays(build: Path, arrays: dict, final_positions: np.ndarray) -> None:
    np.savez_compressed(
        build / "metric_fidelity_materialization.npz",
        positions_2d=arrays["positions_2d"].astype(np.float32),
        positions_rest=arrays["positions_rest"].astype(np.float32),
        positions_final=final_positions.astype(np.float32),
        triangles=arrays["triangles"], edges=arrays["edges"],
        pattern_rest_lengths=arrays["pattern_rest_lengths"].astype(np.float32),
        arrangement_rest_lengths=arrays["arrangement_rest_lengths"].astype(np.float32),
        seam_pairs=arrays["seam_pairs"], fixed_mask=arrays["fixed_mask"],
        component_ids=arrays["component_ids"],
    )


def run_metric_fidelity(root: Path) -> dict:
    build = root / "build/r1c_cp4_r2_r1"
    build.mkdir(parents=True, exist_ok=True)
    arrays, baseline_product, offsets = _load_arrays(root)
    before = decompose_metric(arrays, baseline_product, offsets, "CP4_R1_BEFORE")
    profile = BodyProfile()
    repaired_positions, repair, repaired_lengths = repair_arrangement(arrays, profile)
    arrays["positions_rest"] = repaired_positions
    arrays["arrangement_rest_lengths"] = repaired_lengths
    after = decompose_metric(arrays, baseline_product, offsets, "CP4_R2_R1_AFTER")
    final_positions, settling = settle_material(root, arrays, profile)
    qualification = _qualification(arrays, final_positions, settling, profile)
    product = publish_product(build, baseline_product, arrays, final_positions)
    _save_arrays(build, arrays, final_positions)
    for name, payload in (
        ("metric_decomposition_before.json", before),
        ("isometric_arrangement_repair.json", repair),
        ("metric_decomposition_after.json", after),
        ("warp_settling_receipt.json", settling),
        ("technical_qualification_receipt.json", qualification),
        ("materialized_product_receipt.json", product),
    ):
        write_json(build / name, payload)
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP4_R2_R1",
        "phase": "CANDIDATE_PENDING_DIRECT_ART_AUDIT",
        "terminal_decision": "PENDING_DIRECT_ART_AUDIT",
        "source_cp4_r1_product_sha256": baseline_product["product_sha256"],
        "edge_count": before["edge_count"],
        "metric_improved": repair["global_metric_improved"],
        "before_ratio_p95": before["global"]["arrangement_over_pattern"]["p95"],
        "before_ratio_p99": before["global"]["arrangement_over_pattern"]["p99"],
        "after_ratio_p95": after["global"]["arrangement_over_pattern"]["p95"],
        "after_ratio_p99": after["global"]["arrangement_over_pattern"]["p99"],
        "technical_pass": qualification["technical_pass"],
        "direct_art_audit": "PENDING",
        "product_acceptance": False,
        "cp4_r1_predecessor_mutated": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    write_json(build / "cp4_r2_r1_candidate_receipt.json", payload)
    write_json(root / "R1C_STATUS.json", payload)
    return payload
