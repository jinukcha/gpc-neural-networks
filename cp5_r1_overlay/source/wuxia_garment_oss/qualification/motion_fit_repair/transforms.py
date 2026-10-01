"""Pose-field transform primitives for CP5-R1."""
from __future__ import annotations

import numpy as np


def smoothstep(low: float, high: float, value: np.ndarray) -> np.ndarray:
    scaled = np.clip((value - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def rotate_z(points: np.ndarray, angle: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    output = points.copy()
    x = points[:, 0] - center_xy[0]
    y = points[:, 1] - center_xy[1]
    cosine = np.cos(angle)
    sine = np.sin(angle)
    output[:, 0] = cosine * x - sine * y + center_xy[0]
    output[:, 1] = sine * x + cosine * y + center_xy[1]
    return output


def rotate_x(points: np.ndarray, angle: np.ndarray, pivot_z: float) -> np.ndarray:
    output = points.copy()
    y = points[:, 1]
    z = points[:, 2] - pivot_z
    cosine = np.cos(angle)
    sine = np.sin(angle)
    output[:, 1] = cosine * y - sine * z
    output[:, 2] = sine * y + cosine * z + pivot_z
    return output


def rotate_vectors(vectors: np.ndarray, twist: np.ndarray, bend: np.ndarray) -> np.ndarray:
    output = vectors.copy()
    x = output[:, 0].copy()
    y = output[:, 1].copy()
    cosine = np.cos(twist)
    sine = np.sin(twist)
    output[:, 0] = cosine * x - sine * y
    output[:, 1] = sine * x + cosine * y
    y = output[:, 1].copy()
    z = output[:, 2].copy()
    cosine = np.cos(bend)
    sine = np.sin(bend)
    output[:, 1] = cosine * y - sine * z
    output[:, 2] = sine * y + cosine * z
    return output
