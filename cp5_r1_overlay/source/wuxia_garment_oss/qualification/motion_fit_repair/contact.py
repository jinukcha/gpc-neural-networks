"""Pose-consistent body contact with pelvis-to-thigh lower-body sections."""
from __future__ import annotations

import numpy as np

from ..motion_fit.body import BodyEnvelope
from .field import PoseField
from .transforms import rotate_vectors, rotate_x, rotate_z, smoothstep


def inverse_points(points: np.ndarray, envelope: BodyEnvelope, field: PoseField) -> np.ndarray:
    local = np.asarray(points, dtype=np.float64) - field.translations
    pivot = envelope.z_min + (envelope.z_max - envelope.z_min) * 0.50
    local = rotate_x(local, -field.bend_angles, pivot)
    return rotate_z(local, -field.twist_angles, envelope.center_xy)


def _ellipse_clearance(
    x: np.ndarray,
    y: np.ndarray,
    radius_x: np.ndarray,
    radius_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.sqrt((x / radius_x) ** 2 + (y / radius_y) ** 2)
    clearance = (normalized - 1.0) * np.minimum(radius_x, radius_y)
    gradient = np.column_stack((x / radius_x**2, y / radius_y**2, np.zeros_like(x)))
    norm = np.linalg.norm(gradient, axis=1)
    fallback = norm < 1.0e-12
    gradient[fallback, 0] = 1.0
    norm[fallback] = 1.0
    return clearance, gradient / norm[:, None]


def _lower_clearance(
    x: np.ndarray,
    y: np.ndarray,
    radius_x: np.ndarray,
    radius_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = radius_x * 0.31
    leg_x = np.maximum(radius_x * 0.46, 0.028)
    leg_y = np.maximum(radius_y * 0.76, 0.026)
    left_clearance, left_normal = _ellipse_clearance(x + center, y, leg_x, leg_y)
    right_clearance, right_normal = _ellipse_clearance(x - center, y, leg_x, leg_y)
    use_left = left_clearance <= right_clearance
    clearance = np.where(use_left, left_clearance, right_clearance)
    normal = np.where(use_left[:, None], left_normal, right_normal)
    return clearance, normal


def clearance_and_normal(
    points: np.ndarray,
    envelope: BodyEnvelope,
    field: PoseField,
) -> tuple[np.ndarray, np.ndarray]:
    local = inverse_points(points, envelope, field)
    z = np.clip(local[:, 2], envelope.z_min, envelope.z_max)
    radius_x = np.interp(z, envelope.z_nodes, envelope.radius_x)
    radius_y = np.interp(z, envelope.z_nodes, envelope.radius_y)
    x = local[:, 0] - envelope.center_xy[0]
    y = local[:, 1] - envelope.center_xy[1]
    torso_clearance, torso_normal = _ellipse_clearance(x, y, radius_x, radius_y)
    lower_clearance, lower_normal = _lower_clearance(x, y, radius_x, radius_y)
    lower_weight = 1.0 - smoothstep(0.24, 0.48, field.normalized_z)
    clearance = torso_clearance * (1.0 - lower_weight) + lower_clearance * lower_weight
    normal = torso_normal * (1.0 - lower_weight[:, None]) + lower_normal * lower_weight[:, None]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
    return clearance, rotate_vectors(normal, field.twist_angles, field.bend_angles)


def project_outside(
    points: np.ndarray,
    envelope: BodyEnvelope,
    field: PoseField,
    maximum_step_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a bounded XPBD contact step; repeated iterations close the full gap."""
    clearance, normal = clearance_and_normal(points, envelope, field)
    required = np.maximum(envelope.contact_margin_m - clearance, 0.0)
    applied = required if maximum_step_m is None else np.minimum(required, maximum_step_m)
    return points + normal * applied[:, None], applied
