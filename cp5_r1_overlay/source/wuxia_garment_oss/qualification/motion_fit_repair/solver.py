"""Adaptive transition-and-settle solver for the six failed CP5 poses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..motion_fit.body import BodyEnvelope
from ..motion_fit.mesh import MotionMesh
from ..motion_fit.poses import MotionPose
from ..motion_fit.solver import material_controls
from .contact import project_outside
from .field import PoseField, build_pose_field
from .transforms import smoothstep


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
class RepairSchedule:
    transition_frames: int
    settle_frames: int
    substeps: int
    iterations: int
    drive_scale: float
    attachment_gain: float

    @property
    def frame_count(self) -> int:
        return self.transition_frames + self.settle_frames


@dataclass(frozen=True)
class RepairSolveResult:
    pose_id: str
    material_id: str
    positions: np.ndarray
    target_positions: np.ndarray
    drive_weights: np.ndarray
    frame_max_displacement_m: np.ndarray
    contact_frame_count: np.ndarray
    maximum_contact_correction_m: np.ndarray
    total_projection_steps: int
    schedule: RepairSchedule
    runtime: dict
    field: PoseField


def schedule_for(pose_id: str) -> RepairSchedule:
    values = {
        "ARMS_OVERHEAD": RepairSchedule(12, 14, 2, 7, 0.72, 0.30),
        "CROSS_BODY_REACH": RepairSchedule(12, 14, 2, 7, 0.66, 0.28),
        "FORWARD_BEND": RepairSchedule(18, 18, 3, 9, 0.45, 0.22),
        "SEATED": RepairSchedule(14, 14, 2, 7, 0.62, 0.20),
        "SQUAT": RepairSchedule(16, 16, 3, 8, 0.56, 0.20),
        "WALK_STRIDE": RepairSchedule(14, 14, 2, 8, 0.58, 0.18),
    }
    if pose_id not in values:
        raise KeyError(f"CP5-R1 only repairs failed poses: {pose_id}")
    return values[pose_id]


def _accumulate(count: int, indices: np.ndarray, corrections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.zeros((count, 3), dtype=np.float64)
    weights = np.zeros(count, dtype=np.float64)
    np.add.at(vectors, indices, corrections)
    np.add.at(weights, indices, 1.0)
    return vectors, weights


def _project_pairs(
    positions: np.ndarray,
    pairs: np.ndarray,
    rest_lengths: np.ndarray,
    gain: float,
) -> np.ndarray:
    if not pairs.size:
        return positions
    first = pairs[:, 0]
    second = pairs[:, 1]
    delta = positions[second] - positions[first]
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1.0e-9
    scalar = np.zeros_like(length)
    scalar[valid] = 0.5 * gain * (length[valid] - rest_lengths[valid]) / length[valid]
    correction = delta * scalar[:, None]
    a_vector, a_count = _accumulate(len(positions), first, correction)
    b_vector, b_count = _accumulate(len(positions), second, -correction)
    count = a_count + b_count
    active = count > 0.0
    positions[active] += (a_vector[active] + b_vector[active]) / count[active, None]
    return positions


def _project_attachments(
    positions: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    gain: float,
) -> np.ndarray:
    if indices.size:
        positions[indices] += gain * (targets[indices] - positions[indices])
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
    weight_wp = wp.array(np.asarray(weights, dtype=np.float32), dtype=float, device="cpu")
    wp.launch(_drive_to_pose, dim=len(positions), inputs=[current_wp, previous_wp, target_wp, weight_wp, gain, damping], device="cpu")
    wp.synchronize()
    return current_wp.numpy().astype(np.float64), previous_wp.numpy().astype(np.float64)


def _runtime() -> dict:
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
        "kernel": "CP5_R1_SEGMENTED_POSE_DRIVE_V1",
    }


def _phase_target(mesh: MotionMesh, field: PoseField, frame: int, schedule: RepairSchedule) -> np.ndarray:
    ratio = min((frame + 1) / schedule.transition_frames, 1.0)
    phase = float(smoothstep(0.0, 1.0, np.asarray([ratio]))[0])
    return mesh.positions + phase * (field.target_positions - mesh.positions)


def solve_failed_pose(
    mesh: MotionMesh,
    envelope: BodyEnvelope,
    pose: MotionPose,
    material_id: str,
    material_profile: dict,
) -> RepairSolveResult:
    controls = material_controls(material_profile)
    schedule = schedule_for(pose.pose_id)
    field = build_pose_field(mesh, envelope, pose)
    positions = mesh.positions.copy()
    previous = positions.copy()
    frame_motion = []
    contact_count = np.zeros(mesh.vertex_count, dtype=np.int32)
    maximum_correction = np.zeros(mesh.vertex_count, dtype=np.float64)
    for frame in range(schedule.frame_count):
        start = positions.copy()
        target = _phase_target(mesh, field, frame, schedule)
        settling = frame >= schedule.transition_frames
        damping = 0.0 if settling else min(controls.damping, 0.035)
        drive_gain = controls.drive_gain * schedule.drive_scale * (0.72 if settling else 1.0)
        for _ in range(schedule.substeps):
            positions, previous = _drive_step(positions, previous, target, field.drive_weights, drive_gain, damping)
            for _ in range(schedule.iterations):
                positions = _project_pairs(positions, mesh.edges, mesh.edge_rest_lengths, controls.structural_gain)
                positions = _project_pairs(positions, mesh.seam_pairs, mesh.seam_rest_lengths, controls.seam_gain)
                positions = _project_attachments(positions, target, mesh.attachment_indices, schedule.attachment_gain)
                positions, correction = project_outside(positions, envelope, field)
                contact_count += correction > 1.0e-9
                maximum_correction = np.maximum(maximum_correction, correction)
        frame_motion.append(float(np.max(np.linalg.norm(positions - start, axis=1))))
    if not np.isfinite(positions).all():
        raise FloatingPointError(f"non-finite repaired pose: {material_id}/{pose.pose_id}")
    total_steps = schedule.frame_count * schedule.substeps * schedule.iterations
    return RepairSolveResult(
        pose.pose_id,
        material_id,
        positions,
        field.target_positions,
        field.drive_weights,
        np.asarray(frame_motion, dtype=np.float64),
        contact_count,
        maximum_correction,
        total_steps,
        schedule,
        _runtime(),
        field,
    )
