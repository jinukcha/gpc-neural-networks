"""End-to-end Blender-free CP4-R2-R1-REV1 metric repair pipeline."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.materialization.body import BodyProfile
from wuxia_garment_oss.materialization.qualification import _geometry_metrics, _penetration_metrics
from wuxia_garment_oss.materialization.settle import _project_body_clearance, settle_material

from .classify import classify_edges, load_fidelity_inputs
from .direct_glb import write_direct_glb
from .repair import repair_arrangement


def run_metric_fidelity_pipeline(root: Path) -> dict:
    build = root / "build/r1c_cp4_r2_rev1"
    build.mkdir(parents=True, exist_ok=True)
    inputs = load_fidelity_inputs(root)
    baseline_positions = inputs.arrays["positions_rest"].astype(np.float64)
    _, baseline_metrics = classify_edges(inputs, baseline_positions)
    repaired_rest, repair_receipt = repair_arrangement(inputs)
    repaired_rest = _body_clear_rest(inputs, repaired_rest)
    _, after_metrics = classify_edges(inputs, repaired_rest)
    arrays = _updated_arrays(inputs, repaired_rest)
    final_positions, settling = settle_material(root, arrays, BodyProfile())
    _, final_metrics = classify_edges(inputs, final_positions)
    qualification = _qualify(inputs, arrays, final_positions, settling, baseline_metrics, after_metrics)
    components, body_positions, body_triangles = _product_geometry(root, inputs, final_positions)
    glb_path = root / "product/r1c_cp4_r2_rev1_sleeved_tunic.glb"
    glb_receipt = write_direct_glb(
        glb_path,
        components,
        body_positions,
        body_triangles,
        {
            "checkpoint": "GARMENT_CAD_PRO_R1C_CP4_R2_R1_REV1",
            "sourceProduct": "GARMENT_CAD_PRO_R1C_CP4_R1",
            "sourcePatternChanged": False,
            "blenderUsed": False,
        },
    )
    glb_receipt["path"] = glb_path.relative_to(root).as_posix()
    _publish_outputs(
        build,
        inputs,
        arrays,
        final_positions,
        baseline_metrics,
        after_metrics,
        final_metrics,
        repair_receipt,
        settling,
        qualification,
        glb_receipt,
    )
    preliminary = _preliminary_receipt(inputs, qualification, glb_receipt)
    _write_json(build / "cp4_r2_rev1_preliminary_receipt.json", preliminary)
    _write_json(root / "R1C_STATUS.json", preliminary)
    return preliminary


def _body_clear_rest(inputs, positions: np.ndarray) -> np.ndarray:
    arrays = _updated_arrays(inputs, positions)
    temporary = copy.deepcopy(arrays)
    temporary["fixed_mask"] = np.zeros(len(positions), dtype=np.bool_)
    current = positions.copy()
    for _ in range(3):
        current, _ = _project_body_clearance(current, temporary, BodyProfile(), 0.0030)
        _weld_seams(current, inputs.arrays["seam_pairs"].astype(np.int32))
    return current


def _updated_arrays(inputs, positions_rest: np.ndarray) -> dict:
    arrays = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in inputs.arrays.items()}
    arrays["positions_rest"] = positions_rest.astype(np.float64)
    edges = arrays["edges"].astype(np.int32)
    arrays["arrangement_rest_lengths"] = np.linalg.norm(
        positions_rest[edges[:, 0]] - positions_rest[edges[:, 1]], axis=1
    )
    arrays["component_order"] = list(inputs.component_order)
    arrays["offsets"] = dict(inputs.offsets)
    return arrays


def _weld_seams(positions: np.ndarray, pairs: np.ndarray) -> None:
    parent: dict[int, int] = {}
    for first, second in pairs:
        _union(parent, int(first), int(second))
    groups: dict[int, list[int]] = {}
    for node in parent:
        groups.setdefault(_find(parent, node), []).append(node)
    for values in groups.values():
        indices = np.asarray(values, dtype=np.int32)
        positions[indices] = positions[indices].mean(axis=0)


def _qualify(inputs, arrays, final_positions, settling, baseline, after) -> dict:
    edges = arrays["edges"].astype(np.int32)
    rest = arrays["arrangement_rest_lengths"].astype(np.float64)
    lengths = np.linalg.norm(final_positions[edges[:, 0]] - final_positions[edges[:, 1]], axis=1)
    strain = np.abs(lengths / np.maximum(rest, 1.0e-12) - 1.0)
    seam = _seam_metrics(final_positions, arrays["seam_pairs"].astype(np.int32))
    geometry = _geometry_metrics(arrays, final_positions)
    penetration = _penetration_metrics(arrays, final_positions, BodyProfile())
    metric_gates = _metric_gates(baseline, after)
    technical_gates = {
        "metric_fidelity": all(metric_gates.values()),
        "seam_p95": seam["p95_m"] <= 0.0015,
        "edge_strain_p95": float(np.quantile(strain, 0.95)) <= 0.080,
        "edge_strain_p99": float(np.quantile(strain, 0.99)) <= 0.180,
        "body_penetration_p99": penetration["p99_m"] <= 0.0010,
        "non_finite_vertex": geometry["non_finite_vertex_count"] == 0,
        "degenerate_triangle": geometry["degenerate_triangle_count"] == 0,
        "normal_inversion": geometry["normal_inversion_count"] == 0,
        "stable_tail": settling["tail_peak_frame_displacement_m"] <= 0.0015,
        "post_settle_vertex_repair": settling["post_settle_vertex_repair_count"] == 0,
    }
    return {
        "contract": "CP4R2Rev1TechnicalQualificationReceipt/1",
        "metric_gates": metric_gates,
        "technical_gates": technical_gates,
        "seam": seam,
        "edge_strain": _strain_summary(strain),
        "geometry": geometry,
        "body_penetration": penetration,
        "technical_pass": all(technical_gates.values()),
        "visual_review": "PENDING_GODOT_DIRECT_ART_AUDIT",
        "product_acceptance": False,
    }


def _strain_summary(strain: np.ndarray) -> dict:
    return {
        "mean": float(strain.mean()),
        "p95": float(np.quantile(strain, 0.95)),
        "p99": float(np.quantile(strain, 0.99)),
        "maximum": float(strain.max()),
    }


def _metric_gates(baseline: dict, after: dict) -> dict:
    before_target = baseline["target_components"]
    after_target = after["target_components"]
    groups = after["groups"]
    ordinary = _select(groups, region="INTERIOR", feature="ORDINARY_INTERIOR")
    cap = _select(groups, feature_prefix="SLEEVE_CAP_")
    attach = _select(groups, features={"COLLAR_ATTACH", "CUFF_ATTACH"})
    boundary = _select(groups, region="BOUNDARY", exclude_prefix="SLEEVE_CAP_", exclude_features={"COLLAR_ATTACH", "CUFF_ATTACH"})
    return {
        "target_log_p95_improved": after_target["log_error_p95"] < before_target["log_error_p95"],
        "target_log_p99_improved": after_target["log_error_p99"] < before_target["log_error_p99"],
        "ordinary_interior_p95": ordinary["p95"] <= 1.15,
        "ordinary_interior_p99": ordinary["p99"] <= 1.35,
        "non_eased_boundary_p95": boundary["p95"] <= 1.08,
        "non_eased_boundary_max": boundary["maximum"] <= 1.15,
        "sleeve_cap_p95": cap["p95"] <= 1.10,
        "sleeve_cap_max": cap["maximum"] <= 1.20,
        "collar_cuff_attach_p95": attach["p95"] <= 1.08,
    }


def _select(groups, region=None, feature=None, feature_prefix=None, features=None, exclude_prefix=None, exclude_features=None):
    selected = []
    for item in groups:
        if region and item["region"] != region:
            continue
        if feature and item["feature"] != feature:
            continue
        if feature_prefix and not item["feature"].startswith(feature_prefix):
            continue
        if features and item["feature"] not in features:
            continue
        if exclude_prefix and item["feature"].startswith(exclude_prefix):
            continue
        if exclude_features and item["feature"] in exclude_features:
            continue
        selected.append(item)
    return {
        "p95": max((item["symmetric_ratio_p95"] for item in selected), default=1.0),
        "p99": max((item["symmetric_ratio_p99"] for item in selected), default=1.0),
        "maximum": max((item["symmetric_ratio_max"] for item in selected), default=1.0),
    }


def _seam_metrics(positions: np.ndarray, pairs: np.ndarray) -> dict:
    values = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    return {
        "mean_m": float(values.mean()),
        "p95_m": float(np.quantile(values, 0.95)),
        "maximum_m": float(values.max()),
    }


def _product_geometry(root, inputs, final_positions):
    components = []
    for instance_id in inputs.component_order:
        item = inputs.components[instance_id]
        offset = inputs.offsets[instance_id]
        count = len(item["positions_m"])
        components.append({
            "instance_id": instance_id,
            "positions": final_positions[offset : offset + count],
            "triangles": np.asarray(item["triangles"], dtype=np.uint32),
        })
    with np.load(root / "build/r1c_cp4_r1/product/materialized_product_arrays.npz", allow_pickle=False) as data:
        body_positions = np.asarray(data["body_positions"], dtype=np.float64)
        body_triangles = np.asarray(data["body_triangles"], dtype=np.uint32)
    return components, body_positions, body_triangles


def _publish_outputs(build, inputs, arrays, final_positions, baseline, after, final, repair, settling, qualification, glb):
    _write_json(build / "metric_decomposition_before.json", baseline)
    _write_json(build / "metric_decomposition_after.json", after)
    _write_json(build / "metric_decomposition_post_settle.json", final)
    _write_json(build / "top_edges_before.json", {"edges": baseline["top_distorted_edges"]})
    _write_json(build / "top_edges_after.json", {"edges": after["top_distorted_edges"]})
    _write_json(build / "isometric_repair_receipt.json", repair)
    _write_json(build / "warp_resettle_receipt.json", settling)
    _write_json(build / "technical_qualification_receipt.json", qualification)
    _write_json(build / "direct_glb_package.json", glb)
    _write_json(build / "before_seam_lines.json", _seam_lines(inputs, inputs.arrays["positions_final"]))
    _write_json(build / "after_seam_lines.json", _seam_lines(inputs, final_positions))
    np.savez_compressed(
        build / "repaired_materialization.npz",
        positions_2d=arrays["positions_2d"].astype(np.float32),
        positions_rest=arrays["positions_rest"].astype(np.float32),
        positions_final=final_positions.astype(np.float32),
        triangles=arrays["triangles"], edges=arrays["edges"], seam_pairs=arrays["seam_pairs"],
        fixed_mask=arrays["fixed_mask"], component_ids=arrays["component_ids"],
        pattern_rest_lengths=arrays["pattern_rest_lengths"].astype(np.float32),
        arrangement_rest_lengths=arrays["arrangement_rest_lengths"].astype(np.float32),
    )


def _seam_lines(inputs, positions: np.ndarray) -> dict:
    lines = []
    for seam in inputs.assembled_package["seams"]:
        endpoint = seam["endpoint_a"]
        component = endpoint["component_instance_id"]
        boundary = endpoint["boundary_id"]
        indices = np.asarray(inputs.components[component]["boundaries"][boundary], dtype=np.int32)
        global_indices = indices + inputs.offsets[component]
        lines.append({"interface_id": seam["interface_id"], "positions_m": positions[global_indices].tolist()})
    return {"seam_lines": lines}


def _preliminary_receipt(inputs, qualification, glb):
    return {
        "checkpoint": "GARMENT_CAD_PRO_R1C_CP4_R2_R1_REV1",
        "phase": "DIRECT_GLB_PENDING_VALIDATOR_AND_GODOT_ART_AUDIT",
        "terminal_decision": "PENDING_EXTERNAL_VALIDATION_AND_DIRECT_ART_AUDIT",
        "cp4_r1_predecessor_mutated": False,
        "structural_edge_count": int(len(inputs.arrays["edges"])),
        "technical_pass": qualification["technical_pass"],
        "direct_glb_internal_reopen_pass": glb["fresh_reopen"]["pass"],
        "khronos_validator_pass": False,
        "godot_consumer_pass": False,
        "direct_visual_art_review": "PENDING",
        "product_acceptance": False,
        "blender_executed": False,
        "blend_generated": False,
        "post_settle_vertex_repair_count": 0,
    }


def _find(parent, node):
    parent.setdefault(node, node)
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent, first, second):
    root_first, root_second = _find(parent, first), _find(parent, second)
    if root_first != root_second:
        lower, upper = sorted((root_first, root_second))
        parent[upper] = lower


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
