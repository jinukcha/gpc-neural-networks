"""Reference body geometry for CP4-R1 materialization."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class BodyProfile:
    stature_m: float = 1.7400
    shoulder_width_m: float = 0.4100
    chest_circumference_m: float = 0.9890
    waist_circumference_m: float = 0.8560
    hip_circumference_m: float = 1.0200
    arm_length_m: float = 0.5829
    upper_arm_circumference_m: float = 0.30659
    elbow_circumference_m: float = 0.24725
    wrist_circumference_m: float = 0.16813
    neck_circumference_m: float = 0.3608

    def to_dict(self) -> dict:
        return {"contract": "ReferenceBodyProfile/1", **self.__dict__}


def torso_radii(profile: BodyProfile, y: float) -> tuple[float, float]:
    values = {
        "hip": profile.hip_circumference_m / (2.0 * math.pi),
        "waist": profile.waist_circumference_m / (2.0 * math.pi),
        "chest": profile.chest_circumference_m / (2.0 * math.pi),
    }
    if y <= 1.02:
        t = min(max((y - 0.78) / 0.24, 0.0), 1.0)
        mean = values["hip"] * (1.0 - t) + values["waist"] * t
        depth_ratio = 0.90 * (1.0 - t) + 0.78 * t
    else:
        t = min(max((y - 1.02) / 0.36, 0.0), 1.0)
        mean = values["waist"] * (1.0 - t) + values["chest"] * t
        depth_ratio = 0.78 * (1.0 - t) + 0.83 * t
    return mean * 1.15, mean * depth_ratio


def arm_datums(profile: BodyProfile, side: str):
    sign = -1.0 if side == "LEFT" else 1.0
    shoulder = np.array([sign * profile.shoulder_width_m * 0.5, 1.385, 0.015])
    elbow = np.array([sign * 0.485, 1.205, 0.050])
    wrist = np.array([sign * (profile.shoulder_width_m * 0.5 + profile.arm_length_m), 1.055, 0.065])
    return shoulder, elbow, wrist


def arm_radius(profile: BodyProfile, t: float) -> float:
    upper = profile.upper_arm_circumference_m / (2.0 * math.pi)
    elbow = profile.elbow_circumference_m / (2.0 * math.pi)
    wrist = profile.wrist_circumference_m / (2.0 * math.pi)
    if t <= 0.52:
        blend = t / 0.52
        return upper * (1.0 - blend) + elbow * blend
    blend = (t - 0.52) / 0.48
    return elbow * (1.0 - blend) + wrist * blend


def arm_path(profile: BodyProfile, side: str, t: float):
    shoulder, elbow, wrist = arm_datums(profile, side)
    split = np.linalg.norm(elbow - shoulder) / (np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))
    if t <= split:
        local = t / split
        centre = shoulder * (1.0 - local) + elbow * local
        tangent = elbow - shoulder
    else:
        local = (t - split) / (1.0 - split)
        centre = elbow * (1.0 - local) + wrist * local
        tangent = wrist - elbow
    return centre, tangent / np.linalg.norm(tangent)


def local_frame(tangent: np.ndarray):
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(tangent, reference))) > 0.92:
        reference = np.array([0.0, 0.0, 1.0])
    first = np.cross(tangent, reference)
    first /= np.linalg.norm(first)
    second = np.cross(tangent, first)
    return first, second / np.linalg.norm(second)


def build_body_mesh(profile: BodyProfile):
    vertices, triangles, regions = [], [], []
    _append_torso(profile, vertices, triangles, regions)
    _append_ellipsoid(np.array([0.0, 1.49, 0.0]), (0.062, 0.11, 0.057), 1, vertices, triangles, regions)
    _append_ellipsoid(np.array([0.0, 1.67, 0.0]), (0.095, 0.13, 0.105), 2, vertices, triangles, regions)
    _append_arm(profile, "LEFT", vertices, triangles, regions)
    _append_arm(profile, "RIGHT", vertices, triangles, regions)
    return np.asarray(vertices), np.asarray(triangles, np.int32), np.asarray(regions, np.int16)


def _append_torso(profile, vertices, triangles, regions, rings=18, sides=36):
    offset = len(vertices)
    for ring in range(rings):
        y = 0.76 + 0.68 * ring / (rings - 1)
        rx, rz = torso_radii(profile, y)
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides
            vertices.append((rx * math.sin(angle), y, rz * math.cos(angle)))
            regions.append(0)
    _connect(offset, rings, sides, triangles)


def _append_arm(profile, side, vertices, triangles, regions, rings=14, sides=20):
    offset = len(vertices)
    for ring in range(rings):
        t = ring / (rings - 1)
        centre, tangent = arm_path(profile, side, t)
        first, second = local_frame(tangent)
        radius = arm_radius(profile, t)
        for index in range(sides):
            angle = 2.0 * math.pi * index / sides
            point = centre + radius * (first * math.cos(angle) + second * math.sin(angle))
            vertices.append(tuple(point))
            regions.append(3 if side == "LEFT" else 4)
    _connect(offset, rings, sides, triangles)


def _append_ellipsoid(centre, radii, region, vertices, triangles, regions, rings=10, sides=24):
    offset = len(vertices)
    for ring in range(rings):
        phi = math.pi * (ring + 0.5) / rings
        for side in range(sides):
            theta = 2.0 * math.pi * side / sides
            vertices.append(tuple(centre + np.array([
                radii[0] * math.sin(phi) * math.cos(theta),
                radii[1] * math.cos(phi),
                radii[2] * math.sin(phi) * math.sin(theta),
            ])))
            regions.append(region)
    _connect(offset, rings, sides, triangles)


def _connect(offset, rings, sides, triangles):
    for ring in range(rings - 1):
        for side in range(sides):
            lower = offset + ring * sides
            upper = lower + sides
            nxt = (side + 1) % sides
            triangles.extend(((lower + side, upper + side, upper + nxt), (lower + side, upper + nxt, lower + nxt)))
