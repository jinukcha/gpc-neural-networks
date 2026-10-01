"""Warp-driven sit, squat, and stride qualification for trousers topology."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...pattern_cad.document.model import canonical_sha256
from .mesh import TrousersMesh


POSE_IDS = ("SEATED", "SQUAT", "WALK_STRIDE")


@wp.kernel
def _drive(
    positions: wp.array(dtype=wp.vec3),
    previous: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    gain: float,
    damping: float,
):
    index = wp.tid()
    current = positions[index]
    velocity = current - previous[index]
    previous[index] = current
    positions[index] = current + gain * weights[index] * (targets[index] - current) + damping * velocity


@dataclass(frozen=True)
class TrousersFitResult:
    material_id: str
    pose_id: str
    positions: np.ndarray
    target_positions: np.ndarray
    frame_motion: np.ndarray
    maps: dict[str, np.ndarray]
    metrics: dict[str, float]
    receipt: dict
    runtime: dict


def _rotate_x(points: np.ndarray, angles: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    output = points.copy()
    local = points - pivot
    cosine = np.cos(angles)
    sine = np.sin(angles)
    output[:, 1] = cosine * local[:, 1] - sine * local[:, 2] + pivot[:, 1]
    output[:, 2] = sine * local[:, 1] + cosine * local[:, 2] + pivot[:, 2]
    return output


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - edge0) / max(edge1 - edge0, 1.0e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def pose_target(positions: np.ndarray, pose_id: str, pom: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(positions, dtype=np.float64)
    outseam = pom["finished_outseam"]
    crotch_z = outseam - pom["finished_inseam"]
    knee_z = pom["knee_height"]
    upper_leg = 1.0 - _smoothstep(crotch_z + 0.10, crotch_z + 0.32, base[:, 2])
    lower_leg = 1.0 - _smoothstep(knee_z + 0.05, knee_z + 0.20, base[:, 2])
    side = np.tanh(base[:, 0] / max(pom["finished_hip"] * 0.08, 1.0e-5))
    target = base.copy()
    if pose_id == "SEATED":
        angle = upper_leg * 0.42
        pivot = np.column_stack((np.zeros(len(base)), np.zeros(len(base)), np.full(len(base), crotch_z + 0.08)))
        target = _rotate_x(target, angle, pivot)
        target[:, 2] -= upper_leg * 0.045
        target[:, 1] += upper_leg * 0.035
    elif pose_id == "SQUAT":
        hip_angle = upper_leg * 0.30
        pivot = np.column_stack((np.zeros(len(base)), np.zeros(len(base)), np.full(len(base), crotch_z + 0.09)))
        target = _rotate_x(target, hip_angle, pivot)
        target[:, 2] -= upper_leg * 0.085 + lower_leg * 0.030
        target[:, 1] += upper_leg * 0.020
        target[:, 0] += side * upper_leg * 0.010
    elif pose_id == "WALK_STRIDE":
        angle = side * upper_leg * 0.24
        pivot = np.column_stack((base[:, 0] * 0.20, np.zeros(len(base)), np.full(len(base), crotch_z + 0.07)))
        target = _rotate_x(target, angle, pivot)
        target[:, 1] += side * lower_leg * 0.028
    else:
        raise KeyError(pose_id)
    displacement = np.linalg.norm(target - base, axis=1)
    weights = np.clip(0.20 + 0.72 * displacement / max(float(np.max(displacement)), 1.0e-8), 0.20, 0.96)
    return target, weights


def _unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0)
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int64)


def _project_edges(positions: np.ndarray, edges: np.ndarray, rest: np.ndarray, gain: float) -> np.ndarray:
    first, second = edges[:, 0], edges[:, 1]
    delta = positions[second] - positions[first]
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1.0e-9
    scalar = np.zeros_like(length)
    scalar[valid] = 0.5 * gain * (length[valid] - rest[valid]) / length[valid]
    correction = delta * scalar[:, None]
    accumulated = np.zeros_like(positions)
    count = np.zeros(len(positions), dtype=np.float64)
    np.add.at(accumulated, first, correction)
    np.add.at(accumulated, second, -correction)
    np.add.at(count, first, 1.0)
    np.add.at(count, second, 1.0)
    active = count > 0.0
    positions[active] += accumulated[active] / count[active, None]
    return positions


def _drive_step(
    positions: np.ndarray,
    previous: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    gain: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    current_wp = wp.array(np.asarray(positions, dtype=np.float32), dtype=wp.vec3, device="cpu")
    previous_wp = wp.array(np.asarray(previous, dtype=np.float32), dtype=wp.vec3, device="cpu")
    target_wp = wp.array(np.asarray(target, dtype=np.float32), dtype=wp.vec3, device="cpu")
    weights_wp = wp.array(np.asarray(weights, dtype=np.float32), dtype=float, device="cpu")
    wp.launch(_drive, dim=len(positions), inputs=[current_wp, previous_wp, target_wp, weights_wp, gain, damping], device="cpu")
    wp.synchronize()
    return current_wp.numpy().astype(np.float64), previous_wp.numpy().astype(np.float64)


def _runtime() -> dict:
    wp.init()
    version = str(getattr(wp, "__version__", "UNKNOWN"))
    if version != "1.17.0":
        raise RuntimeError(f"unexpected Warp runtime: {version}")
    try:
        cuda = bool(wp.is_cuda_available())
    except Exception:
        cuda = False
    return {
        "package": "warp-lang",
        "version": version,
        "device": "cpu",
        "cuda_status": "AVAILABLE_NOT_USED" if cuda else "EXPLICIT_NO_CUDA_DEVICE",
        "kernel": "CP6_TROUSERS_MOTION_DRIVE_V1",
    }


def _effective_stiffness(profile: dict) -> float:
    parameters = profile["parameters"]
    return float(max((parameters["warp_tensile_linear"] * parameters["weft_tensile_linear"]) ** 0.5, 100.0))


def _vertex_max(count: int, edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.zeros(count, dtype=np.float64)
    np.maximum.at(output, edges[:, 0], values)
    np.maximum.at(output, edges[:, 1], values)
    return output


def _maps(
    mesh: TrousersMesh,
    edges: np.ndarray,
    rest: np.ndarray,
    positions: np.ndarray,
    target: np.ndarray,
    stiffness: float,
    profile: dict,
) -> dict[str, np.ndarray]:
    length = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
    strain_edge = np.abs(length - rest) / np.maximum(rest, 1.0e-8)
    strain = _vertex_max(len(positions), edges, strain_edge)
    stress = strain * stiffness
    residual = np.linalg.norm(target - positions, axis=1)
    base_clearance = 0.010 + 0.006 * np.clip(np.abs(positions[:, 0]) / max(np.ptp(positions[:, 0]), 1.0e-8), 0.0, 1.0)
    clearance = base_clearance - residual * 0.08
    penetration = np.maximum(-clearance, 0.0)
    parameters = profile["parameters"]
    thickness = max(float(profile.get("thickness_m", 0.0008)), 1.0e-5)
    compression = np.clip(penetration / thickness, 0.0, 0.35)
    pressure = float(parameters["compression_scale"]) * (np.exp(float(parameters["compression_exponent"]) * compression) - 1.0)
    mobility = np.clip(residual / np.maximum(np.linalg.norm(target - mesh.positions, axis=1), 0.003), 0.0, 1.0)
    return {
        "strain_ratio": strain,
        "stress_n_m": stress,
        "clearance_m": clearance,
        "pressure_pa": pressure,
        "mobility_restriction": mobility,
    }


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _metrics(maps: dict[str, np.ndarray], frame_motion: np.ndarray) -> dict[str, float]:
    return {
        "strain_p99_ratio": _percentile(maps["strain_ratio"], 99.0),
        "strain_max_ratio": float(np.max(maps["strain_ratio"])),
        "stress_p99_n_m": _percentile(maps["stress_n_m"], 99.0),
        "pressure_p99_kpa": _percentile(maps["pressure_pa"], 99.0) / 1000.0,
        "clearance_min_m": float(np.min(maps["clearance_m"])),
        "mobility_restriction_p95": _percentile(maps["mobility_restriction"], 95.0),
        "tail_peak_displacement_m": float(np.max(frame_motion[-3:])),
        "non_finite_map_count": float(sum(np.count_nonzero(~np.isfinite(value)) for value in maps.values())),
    }


def _qualification(material_id: str, pose_id: str, metrics: dict[str, float], runtime: dict) -> dict:
    limits = {
        "strain_p99_ratio": 0.34,
        "pressure_p99_kpa": 25.0,
        "mobility_restriction_p95": 0.35,
        "tail_peak_displacement_m": 0.006,
    }
    gates = {
        "finite_maps": metrics["non_finite_map_count"] == 0.0,
        "strain_p99": metrics["strain_p99_ratio"] <= limits["strain_p99_ratio"],
        "pressure_p99": metrics["pressure_p99_kpa"] <= limits["pressure_p99_kpa"],
        "body_clearance": metrics["clearance_min_m"] >= 0.0,
        "mobility_restriction": metrics["mobility_restriction_p95"] <= limits["mobility_restriction_p95"],
        "convergence": metrics["tail_peak_displacement_m"] <= limits["tail_peak_displacement_m"],
    }
    payload = {
        "contract": "TrousersPoseQualificationReceipt/1",
        "material_id": material_id,
        "pose_id": pose_id,
        "metrics": metrics,
        "limits": limits,
        "gates": gates,
        "runtime": runtime,
        "pose_pass": all(gates.values()),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def solve_pose(
    mesh: TrousersMesh,
    pose_id: str,
    material_id: str,
    material_profile: dict,
    frames: int = 28,
) -> TrousersFitResult:
    runtime = _runtime()
    target, weights = pose_target(mesh.positions, pose_id, mesh.authority["points_of_measure"] if "points_of_measure" in mesh.authority else {})
    positions = mesh.positions.copy()
    previous = positions.copy()
    edges = _unique_edges(mesh.triangles)
    rest = np.linalg.norm(mesh.positions[edges[:, 1]] - mesh.positions[edges[:, 0]], axis=1)
    stiffness = _effective_stiffness(material_profile)
    normalized = float(np.clip((np.log10(stiffness) - 3.5), 0.0, 1.0))
    frame_motion = []
    for frame in range(frames):
        phase = min((frame + 1) / 16.0, 1.0)
        eased = phase * phase * (3.0 - 2.0 * phase)
        phase_target = mesh.positions + eased * (target - mesh.positions)
        start = positions.copy()
        for _ in range(2):
            positions, previous = _drive_step(positions, previous, phase_target, weights, 0.18 - 0.04 * normalized, 0.025)
            for _ in range(8):
                positions = _project_edges(positions, edges, rest, 0.38 + 0.12 * normalized)
        frame_motion.append(float(np.max(np.linalg.norm(positions - start, axis=1))))
    maps = _maps(mesh, edges, rest, positions, target, stiffness, material_profile)
    metrics = _metrics(maps, np.asarray(frame_motion, dtype=np.float64))
    receipt = _qualification(material_id, pose_id, metrics, runtime)
    return TrousersFitResult(
        material_id,
        pose_id,
        positions,
        target,
        np.asarray(frame_motion, dtype=np.float64),
        maps,
        metrics,
        receipt,
        runtime,
    )


def run_fit_suite(mesh: TrousersMesh, pom: dict[str, float], profiles: dict[str, dict]) -> tuple[TrousersFitResult, ...]:
    enriched_authority = {**mesh.authority, "points_of_measure": dict(pom)}
    mesh = TrousersMesh(
        mesh.positions, mesh.triangles, mesh.triangle_panel_ids,
        mesh.waistband_positions, mesh.waistband_triangles,
        mesh.gusset_positions, mesh.gusset_triangles,
        enriched_authority,
    )
    return tuple(
        solve_pose(mesh, pose_id, material_id, profiles[material_id])
        for material_id in sorted(profiles)
        for pose_id in POSE_IDS
    )
