"""Bounded Warp-driven quasi-static motion-fit solver for the fitted tunic reference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .body import BodyEnvelope, pose_target, project_outside
from .mesh import MotionMesh
from .poses import MotionFitSuite, MotionPose


@wp.kernel
def _drive_to_pose(
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
class MaterialControls:
    structural_gain: float
    seam_gain: float
    drive_gain: float
    damping: float
    effective_tensile_n_m: float
    compression_scale: float
    compression_exponent: float
    thickness_m: float


@dataclass(frozen=True)
class MotionSolveResult:
    pose_id: str
    material_id: str
    positions: np.ndarray
    target_positions: np.ndarray
    drive_weights: np.ndarray
    frame_max_displacement_m: np.ndarray
    contact_frame_count: np.ndarray
    maximum_contact_correction_m: np.ndarray
    runtime: dict


def material_controls(profile: dict) -> MaterialControls:
    parameters = profile["parameters"]
    warp = float(parameters["warp_tensile_linear"])
    weft = float(parameters["weft_tensile_linear"])
    effective = max((warp * weft) ** 0.5, 100.0)
    normalized = float(np.clip((np.log10(effective) - 3.5) / 1.0, 0.0, 1.0))
    return MaterialControls(
        structural_gain=0.30 + 0.20 * normalized,
        seam_gain=0.48 + 0.22 * normalized,
        drive_gain=0.14 + 0.07 * (1.0 - normalized),
        damping=0.11 - 0.04 * normalized,
        effective_tensile_n_m=effective,
        compression_scale=float(parameters["compression_scale"]),
        compression_exponent=float(parameters["compression_exponent"]),
        thickness_m=float(profile.get("thickness_m", 0.0008)),
    )


def _accumulate_corrections(vertex_count: int, indices: np.ndarray, corrections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    accumulated = np.zeros((vertex_count, 3), dtype=np.float64)
    count = np.zeros(vertex_count, dtype=np.float64)
    np.add.at(accumulated, indices, corrections)
    np.add.at(count, indices, 1.0)
    return accumulated, count


def _project_edges(positions: np.ndarray, mesh: MotionMesh, gain: float) -> np.ndarray:
    first = mesh.edges[:, 0]
    second = mesh.edges[:, 1]
    delta = positions[second] - positions[first]
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1.0e-9
    scalar = np.zeros_like(length)
    scalar[valid] = 0.5 * gain * (length[valid] - mesh.edge_rest_lengths[valid]) / length[valid]
    correction = delta * scalar[:, None]
    accumulated_a, count_a = _accumulate_corrections(len(positions), first, correction)
    accumulated_b, count_b = _accumulate_corrections(len(positions), second, -correction)
    count = count_a + count_b
    update = accumulated_a + accumulated_b
    active = count > 0.0
    positions[active] += update[active] / count[active, None]
    return positions


def _project_seams(positions: np.ndarray, mesh: MotionMesh, gain: float) -> np.ndarray:
    if not mesh.seam_pairs.size:
        return positions
    first = mesh.seam_pairs[:, 0]
    second = mesh.seam_pairs[:, 1]
    delta = positions[second] - positions[first]
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1.0e-9
    scalar = np.zeros_like(length)
    scalar[valid] = 0.5 * gain * (length[valid] - mesh.seam_rest_lengths[valid]) / length[valid]
    correction = delta * scalar[:, None]
    accumulated_a, count_a = _accumulate_corrections(len(positions), first, correction)
    accumulated_b, count_b = _accumulate_corrections(len(positions), second, -correction)
    count = count_a + count_b
    update = accumulated_a + accumulated_b
    active = count > 0.0
    positions[active] += update[active] / count[active, None]
    return positions


def _warp_runtime() -> dict:
    wp.init()
    version = str(getattr(wp, "__version__", "UNKNOWN"))
    if version != "1.17.0":
        raise RuntimeError(f"unexpected Warp version: {version}")
    try:
        cuda_available = bool(wp.is_cuda_available())
    except Exception:
        cuda_available = False
    return {
        "package": "warp-lang",
        "version": version,
        "device": "cpu",
        "cuda_status": "AVAILABLE_NOT_USED" if cuda_available else "EXPLICIT_NO_CUDA_DEVICE",
        "kernel": "CP5_DRIVE_TO_POSE_V1",
    }


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
    weight_wp = wp.array(np.asarray(weights, dtype=np.float32), dtype=float, device="cpu")
    wp.launch(
        _drive_to_pose,
        dim=len(positions),
        inputs=[current_wp, previous_wp, target_wp, weight_wp, gain, damping],
        device="cpu",
    )
    wp.synchronize()
    return current_wp.numpy().astype(np.float64), previous_wp.numpy().astype(np.float64)


def solve_pose(
    mesh: MotionMesh,
    envelope: BodyEnvelope,
    pose: MotionPose,
    suite: MotionFitSuite,
    material_id: str,
    material_profile: dict,
) -> MotionSolveResult:
    controls = material_controls(material_profile)
    runtime = _warp_runtime()
    target, drive_weight = pose_target(mesh.positions, envelope, pose)
    positions = mesh.positions.copy()
    previous = positions.copy()
    frame_displacement = []
    contact_count = np.zeros(mesh.vertex_count, dtype=np.int32)
    maximum_correction = np.zeros(mesh.vertex_count, dtype=np.float64)
    for frame in range(suite.frames_per_pose):
        frame_start = positions.copy()
        phase = (frame + 1) / suite.frames_per_pose
        phase_target = mesh.positions + phase * (target - mesh.positions)
        for _ in range(suite.substeps_per_frame):
            positions, previous = _drive_step(
                positions,
                previous,
                phase_target,
                drive_weight,
                controls.drive_gain,
                controls.damping,
            )
            for _ in range(suite.projection_iterations):
                positions = _project_edges(positions, mesh, controls.structural_gain)
                positions = _project_seams(positions, mesh, controls.seam_gain)
                positions, correction = project_outside(positions, envelope, pose)
                contact_count += correction > 1.0e-9
                maximum_correction = np.maximum(maximum_correction, correction)
        displacement = np.linalg.norm(positions - frame_start, axis=1)
        frame_displacement.append(float(np.max(displacement)))
    if not np.isfinite(positions).all():
        raise FloatingPointError(f"non-finite motion-fit position: {material_id}/{pose.pose_id}")
    return MotionSolveResult(
        pose.pose_id,
        material_id,
        positions,
        target,
        drive_weight,
        np.asarray(frame_displacement, dtype=np.float64),
        contact_count,
        maximum_correction,
        runtime,
    )
