"""Component-owned body-clearance projection primitives."""
from __future__ import annotations

import numpy as np

from .body import BodyProfile, arm_datums, arm_radius, torso_radii


def project_point(point, instance_id: str, profile: BodyProfile, clearance: float):
    if instance_id.startswith("bodice"):
        return project_torso(point, profile, clearance)
    if instance_id == "collar":
        return project_neck(point, profile, clearance)
    if instance_id.endswith("left"):
        return project_arm(point, profile, "LEFT", clearance)
    if instance_id.endswith("right"):
        return project_arm(point, profile, "RIGHT", clearance)
    raise ValueError(f"unknown contact owner: {instance_id}")


def project_torso(point, profile: BodyProfile, clearance: float):
    result = np.asarray(point, dtype=np.float64).copy()
    rx, rz = torso_radii(profile, float(result[1]))
    ratio = np.sqrt((result[0] / rx) ** 2 + (result[2] / rz) ** 2)
    minimum = 1.0 + clearance / min(rx, rz)
    if ratio < minimum:
        if ratio <= 1.0e-12:
            result[2] = rz + clearance
        else:
            result[0] *= minimum / ratio
            result[2] *= minimum / ratio
    return result


def project_neck(point, profile: BodyProfile, clearance: float):
    result = np.asarray(point, dtype=np.float64).copy()
    radius = profile.neck_circumference_m / (2.0 * np.pi) + clearance
    radial = result[[0, 2]]
    distance = float(np.linalg.norm(radial))
    if distance < radius:
        direction = radial / distance if distance > 1.0e-12 else np.array([0.0, 1.0])
        result[0] = direction[0] * radius
        result[2] = direction[1] * radius
    return result


def project_arm(point, profile: BodyProfile, side: str, clearance: float):
    result = np.asarray(point, dtype=np.float64).copy()
    shoulder, elbow, wrist = arm_datums(profile, side)
    distance, projection, t = nearest_arm(result, shoulder, elbow, wrist)
    minimum = arm_radius(profile, t) + clearance
    if distance < minimum:
        direction = result - projection
        if np.linalg.norm(direction) <= 1.0e-12:
            direction = np.array([0.0, 0.0, 1.0])
        result = projection + direction / np.linalg.norm(direction) * minimum
    return result


def nearest_arm(point, shoulder, elbow, wrist):
    first = segment_query(point, shoulder, elbow, 0.0, 0.52)
    second = segment_query(point, elbow, wrist, 0.52, 1.0)
    return first if first[0] <= second[0] else second


def segment_query(point, start, end, t0: float, t1: float):
    vector = end - start
    local = float(np.dot(point - start, vector) / np.dot(vector, vector))
    local = min(max(local, 0.0), 1.0)
    projection = start + vector * local
    distance = float(np.linalg.norm(point - projection))
    return distance, projection, t0 * (1.0 - local) + t1 * local
