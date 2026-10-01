"""Pose-aware analytic body envelope derived from immutable fitted garment bounds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .poses import MotionPose


@dataclass(frozen=True)
class BodyEnvelope:
    center_xy: np.ndarray
    z_min: float
    z_max: float
    z_nodes: np.ndarray
    radius_x: np.ndarray
    radius_y: np.ndarray
    contact_margin_m: float

    def to_dict(self) -> dict:
        return {
            "center_xy": self.center_xy.tolist(),
            "z_min": self.z_min,
            "z_max": self.z_max,
            "z_nodes": self.z_nodes.tolist(),
            "radius_x": self.radius_x.tolist(),
            "radius_y": self.radius_y.tolist(),
            "contact_margin_m": self.contact_margin_m,
            "authority": "DERIVED_FROM_IMMUTABLE_FITTED_TUNIC_BOUNDS_V1",
        }


def _section_radius(values: np.ndarray, fallback: float) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size < 8:
        return fallback
    return max(float(np.quantile(finite, 0.72)) * 0.82, fallback)


def derive_body_envelope(positions: np.ndarray, contact_margin_m: float = 0.0035) -> BodyEnvelope:
    center = np.median(positions[:, :2], axis=0)
    relative = positions[:, :2] - center
    z = positions[:, 2]
    z_min = float(np.quantile(z, 0.02))
    z_max = float(np.quantile(z, 0.98))
    nodes = np.linspace(z_min, z_max, 9)
    radius_x = []
    radius_y = []
    span = max(z_max - z_min, 1.0e-6)
    for node in nodes:
        mask = np.abs(z - node) <= 0.09 * span
        radius_x.append(_section_radius(relative[mask, 0], 0.055))
        radius_y.append(_section_radius(relative[mask, 1], 0.040))
    x_values = np.asarray(radius_x, dtype=np.float64)
    y_values = np.asarray(radius_y, dtype=np.float64)
    x_values = np.maximum(x_values, np.convolve(x_values, np.ones(3) / 3.0, mode="same") * 0.92)
    y_values = np.maximum(y_values, np.convolve(y_values, np.ones(3) / 3.0, mode="same") * 0.92)
    return BodyEnvelope(center, z_min, z_max, nodes, x_values, y_values, contact_margin_m)


def _rotation_x(points: np.ndarray, angle: np.ndarray, pivot_z: float) -> np.ndarray:
    output = points.copy()
    y = points[:, 1]
    z = points[:, 2] - pivot_z
    cosine = np.cos(angle)
    sine = np.sin(angle)
    output[:, 1] = cosine * y - sine * z
    output[:, 2] = sine * y + cosine * z + pivot_z
    return output


def _rotation_z(points: np.ndarray, angle: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    output = points.copy()
    x = points[:, 0] - center_xy[0]
    y = points[:, 1] - center_xy[1]
    cosine = np.cos(angle)
    sine = np.sin(angle)
    output[:, 0] = cosine * x - sine * y + center_xy[0]
    output[:, 1] = sine * x + cosine * y + center_xy[1]
    return output


def pose_target(base: np.ndarray, envelope: BodyEnvelope, pose: MotionPose) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(base, dtype=np.float64).copy()
    span = max(envelope.z_max - envelope.z_min, 1.0e-6)
    normalized_z = np.clip((target[:, 2] - envelope.z_min) / span, 0.0, 1.0)
    upper = np.clip((normalized_z - 0.48) / 0.32, 0.0, 1.0)
    shoulder = np.clip((normalized_z - 0.72) / 0.22, 0.0, 1.0)
    lower = 1.0 - np.clip((normalized_z - 0.18) / 0.40, 0.0, 1.0)
    lateral = np.clip(np.abs(target[:, 0] - envelope.center_xy[0]) / max(np.ptp(target[:, 0]) * 0.42, 1.0e-5), 0.0, 1.0)
    shoulder_zone = shoulder * (0.25 + 0.75 * lateral)
    twist_angle = upper * pose.torso_twist * 0.42
    target = _rotation_z(target, twist_angle, envelope.center_xy)
    bend_angle = upper * pose.forward_bend * 0.48 + lower * pose.hip_flexion * 0.20
    target = _rotation_x(target, bend_angle, envelope.z_min + span * 0.48)
    side_sign = np.sign(target[:, 0] - envelope.center_xy[0])
    target[:, 1] += shoulder_zone * pose.arm_forward * 0.105
    target[:, 2] += shoulder_zone * pose.arm_raise * 0.115
    target[:, 0] -= side_sign * shoulder_zone * pose.arm_raise * 0.040
    target[:, 0] -= side_sign * shoulder_zone * pose.cross_body * 0.095
    target[:, 1] += shoulder_zone * pose.cross_body * 0.035
    target[:, 1] += shoulder_zone * pose.elbow_bend * 0.025
    target[:, 2] -= lower * pose.squat * 0.105
    target[:, 1] += lower * pose.squat * 0.055
    target[:, 0] += side_sign * lower * pose.stride * 0.035
    target[:, 1] += side_sign * lower * pose.stride * 0.055
    drive_weight = np.clip(0.10 + 0.72 * upper + 0.42 * lower + 0.35 * shoulder_zone, 0.0, 1.0)
    if pose.pose_id == "NEUTRAL_A":
        drive_weight *= 0.35
    return target, drive_weight


def _inverse_pose(points: np.ndarray, envelope: BodyEnvelope, pose: MotionPose) -> np.ndarray:
    local = np.asarray(points, dtype=np.float64).copy()
    span = max(envelope.z_max - envelope.z_min, 1.0e-6)
    normalized_z = np.clip((local[:, 2] - envelope.z_min) / span, 0.0, 1.0)
    upper = np.clip((normalized_z - 0.48) / 0.32, 0.0, 1.0)
    lower = 1.0 - np.clip((normalized_z - 0.18) / 0.40, 0.0, 1.0)
    bend_angle = -(upper * pose.forward_bend * 0.48 + lower * pose.hip_flexion * 0.20)
    local = _rotation_x(local, bend_angle, envelope.z_min + span * 0.48)
    local = _rotation_z(local, -upper * pose.torso_twist * 0.42, envelope.center_xy)
    local[:, 2] += lower * pose.squat * 0.105
    local[:, 1] -= lower * pose.squat * 0.055
    return local


def clearance_and_normal(points: np.ndarray, envelope: BodyEnvelope, pose: MotionPose) -> tuple[np.ndarray, np.ndarray]:
    local = _inverse_pose(points, envelope, pose)
    z = np.clip(local[:, 2], envelope.z_min, envelope.z_max)
    radius_x = np.interp(z, envelope.z_nodes, envelope.radius_x)
    radius_y = np.interp(z, envelope.z_nodes, envelope.radius_y)
    x = local[:, 0] - envelope.center_xy[0]
    y = local[:, 1] - envelope.center_xy[1]
    normalized = np.sqrt((x / radius_x) ** 2 + (y / radius_y) ** 2)
    clearance = (normalized - 1.0) * np.minimum(radius_x, radius_y)
    gradient = np.column_stack((x / radius_x**2, y / radius_y**2, np.zeros_like(x)))
    norm = np.linalg.norm(gradient, axis=1)
    fallback = norm < 1.0e-12
    gradient[fallback, 0] = 1.0
    norm[fallback] = 1.0
    normal = gradient / norm[:, None]
    return clearance, normal


def project_outside(points: np.ndarray, envelope: BodyEnvelope, pose: MotionPose) -> tuple[np.ndarray, np.ndarray]:
    clearance, normal = clearance_and_normal(points, envelope, pose)
    correction = np.maximum(envelope.contact_margin_m - clearance, 0.0)
    corrected = points + normal * correction[:, None]
    return corrected, correction
