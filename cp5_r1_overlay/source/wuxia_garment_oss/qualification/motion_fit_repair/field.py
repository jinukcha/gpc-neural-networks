"""Segmented torso, pelvis, shoulder, and leg-aware pose field."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..motion_fit.body import BodyEnvelope
from ..motion_fit.mesh import MotionMesh
from ..motion_fit.poses import MotionPose
from .transforms import rotate_x, rotate_z, smoothstep


@dataclass(frozen=True)
class PoseField:
    target_positions: np.ndarray
    drive_weights: np.ndarray
    twist_angles: np.ndarray
    bend_angles: np.ndarray
    translations: np.ndarray
    normalized_z: np.ndarray


def _zones(base: np.ndarray, envelope: BodyEnvelope) -> dict[str, np.ndarray]:
    span = max(envelope.z_max - envelope.z_min, 1.0e-6)
    z_norm = np.clip((base[:, 2] - envelope.z_min) / span, 0.0, 1.0)
    x_rel = base[:, 0] - envelope.center_xy[0]
    y_rel = base[:, 1] - envelope.center_xy[1]
    x_scale = max(float(np.quantile(np.abs(x_rel), 0.95)), 1.0e-4)
    y_scale = max(float(np.quantile(np.abs(y_rel), 0.95)), 1.0e-4)
    lateral = np.clip(np.abs(x_rel) / x_scale, 0.0, 1.0)
    return {
        "z": z_norm,
        "torso": smoothstep(0.40, 0.78, z_norm),
        "shoulder": smoothstep(0.67, 0.91, z_norm) * (0.30 + 0.70 * lateral),
        "pelvis": smoothstep(0.16, 0.40, z_norm) * (1.0 - smoothstep(0.58, 0.74, z_norm)),
        "skirt": 1.0 - smoothstep(0.46, 0.70, z_norm),
        "hem": 1.0 - smoothstep(0.18, 0.52, z_norm),
        "side": np.tanh(x_rel / max(x_scale * 0.22, 1.0e-5)),
        "front": np.tanh(y_rel / max(y_scale * 0.45, 1.0e-5)),
    }


def _translation(zones: dict[str, np.ndarray], pose: MotionPose) -> np.ndarray:
    shoulder = zones["shoulder"]
    pelvis = zones["pelvis"]
    skirt = zones["skirt"]
    hem = zones["hem"]
    side = zones["side"]
    front = zones["front"]
    output = np.zeros((len(shoulder), 3), dtype=np.float64)
    output[:, 1] += shoulder * pose.arm_forward * 0.082
    output[:, 2] += shoulder * pose.arm_raise * 0.098
    output[:, 0] -= side * shoulder * pose.arm_raise * 0.024
    output[:, 0] -= side * shoulder * pose.cross_body * 0.056
    output[:, 1] += shoulder * pose.cross_body * 0.019
    output[:, 1] += shoulder * pose.elbow_bend * 0.020
    output[:, 1] += pelvis * pose.forward_bend * 0.014
    output[:, 1] += pelvis * pose.hip_flexion * 0.020
    output[:, 2] += pelvis * front * pose.hip_flexion * 0.012
    output[:, 2] -= pelvis * pose.squat * 0.045
    output[:, 1] += pelvis * pose.squat * 0.018
    output[:, 0] += side * pelvis * pose.squat * 0.010
    leg_side = np.tanh(side * 1.35)
    output[:, 1] += leg_side * hem * pose.stride * 0.020
    output[:, 0] += side * hem * pose.stride * 0.006
    output[:, 0] += side * skirt * abs(pose.stride) * 0.004
    return output


def _drive(base: np.ndarray, target: np.ndarray, zones: dict[str, np.ndarray]) -> np.ndarray:
    displacement = np.linalg.norm(target - base, axis=1)
    motion = displacement / max(float(np.max(displacement)), 1.0e-8)
    weight = 0.18 + 0.44 * zones["torso"] + 0.38 * zones["skirt"] + 0.28 * zones["shoulder"]
    return np.clip(weight + 0.32 * motion, 0.16, 0.96)


def build_pose_field(mesh: MotionMesh, envelope: BodyEnvelope, pose: MotionPose) -> PoseField:
    base = np.asarray(mesh.positions, dtype=np.float64)
    zones = _zones(base, envelope)
    twist = zones["torso"] * pose.torso_twist * 0.34
    bend = smoothstep(0.44, 0.80, zones["z"]) * pose.forward_bend * 0.34
    target = rotate_z(base, twist, envelope.center_xy)
    target = rotate_x(target, bend, envelope.z_min + (envelope.z_max - envelope.z_min) * 0.50)
    translation = _translation(zones, pose)
    target += translation
    weights = _drive(base, target, zones)
    if pose.pose_id == "NEUTRAL_A":
        weights *= 0.25
    if mesh.attachment_indices.size:
        weights[mesh.attachment_indices] = np.maximum(weights[mesh.attachment_indices], 0.92)
    return PoseField(target, weights, twist, bend, translation, zones["z"])


def scaled_pose_field(base: np.ndarray, field: PoseField, phase: float) -> PoseField:
    """Scale every body-transform component to the current continuation phase."""
    bounded = float(np.clip(phase, 0.0, 1.0))
    target = np.asarray(base, dtype=np.float64) + bounded * (
        field.target_positions - np.asarray(base, dtype=np.float64)
    )
    return PoseField(
        target,
        field.drive_weights,
        field.twist_angles * bounded,
        field.bend_angles * bounded,
        field.translations * bounded,
        field.normalized_z,
    )
