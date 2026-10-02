"""Sleeve-cap, armhole, underarm, and robe-skirt geometry construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measurements import ArmMeasurementSet


@dataclass(frozen=True)
class ConstructedPrimitive:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    texcoords: np.ndarray
    indices: np.ndarray
    longitudinal: np.ndarray
    angular: np.ndarray
    metadata: dict


def _basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = tangent / np.linalg.norm(tangent)
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, reference))) > 0.92:
        reference = np.array([0.0, 0.0, 1.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return first, second


def _chain_point(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    split = np.linalg.norm(elbow - shoulder) / (np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))
    if t <= split:
        local = t / split
        centre = shoulder * (1.0 - local) + elbow * local
        tangent = elbow - shoulder
    else:
        local = (t - split) / (1.0 - split)
        centre = elbow * (1.0 - local) + wrist * local
        tangent = wrist - elbow
    return centre, tangent


def _radius(measurements: ArmMeasurementSet, t: float, style: str) -> tuple[float, float]:
    upper = measurements.upper_arm_circumference_m / (2.0 * np.pi)
    elbow = measurements.elbow_circumference_m / (2.0 * np.pi)
    wrist = measurements.wrist_circumference_m / (2.0 * np.pi)
    if t < 0.52:
        base = upper * (1.0 - t / 0.52) + elbow * (t / 0.52)
    else:
        base = elbow * (1.0 - (t - 0.52) / 0.48) + wrist * ((t - 0.52) / 0.48)
    cap = 0.88 + 0.12 * np.sin(min(t / 0.18, 1.0) * np.pi * 0.5)
    if style == "STRAIGHT_ROBE":
        base = max(base, wrist * 1.28)
        depth = base * 0.92
    else:
        depth = base * 0.86
    return base * cap, depth * cap


def _ring_vertices(centre: np.ndarray, tangent: np.ndarray, radii: tuple[float, float], sides: int):
    first, second = _basis(tangent)
    positions, normals, angles = [], [], []
    for side in range(sides):
        angle = 2.0 * np.pi * side / sides
        radial = first * (radii[0] * np.cos(angle)) + second * (radii[1] * np.sin(angle))
        normal = first * np.cos(angle) + second * np.sin(angle)
        positions.append(centre + radial)
        normals.append(normal / np.linalg.norm(normal))
        angles.append(angle)
    return positions, normals, angles


def _tube_indices(rings: int, sides: int) -> np.ndarray:
    triangles = []
    for ring in range(rings - 1):
        lower, upper = ring * sides, (ring + 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            triangles.extend(((lower + side, upper + side, upper + next_side), (lower + side, upper + next_side, lower + next_side)))
    return np.asarray(triangles, dtype=np.uint32)


def build_sleeve(
    side: str,
    joints: dict[str, np.ndarray],
    measurements: ArmMeasurementSet,
    style: str,
    rings: int = 21,
    sides: int = 28,
) -> ConstructedPrimitive:
    prefix = "L" if side == "LEFT" else "R"
    shoulder = joints[f"{prefix}_UPPER_ARM"]
    elbow = joints[f"{prefix}_FOREARM"]
    wrist = joints[f"{prefix}_HAND"]
    positions, normals, texcoords, longitudinal, angular = [], [], [], [], []
    for ring in range(rings):
        t = ring / (rings - 1)
        centre, tangent = _chain_point(shoulder, elbow, wrist, t)
        ring_positions, ring_normals, ring_angles = _ring_vertices(centre, tangent, _radius(measurements, t, style), sides)
        positions.extend(ring_positions)
        normals.extend(ring_normals)
        texcoords.extend((side_index / sides, t) for side_index in range(sides))
        longitudinal.extend([t] * sides)
        angular.extend(ring_angles)
    armhole = measurements.armscye_circumference_m
    cap = armhole * measurements.cap_ease_ratio
    return ConstructedPrimitive(
        name=f"{side}_{style}_SLEEVE",
        positions=np.asarray(positions, dtype=np.float32),
        normals=np.asarray(normals, dtype=np.float32),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        indices=_tube_indices(rings, sides),
        longitudinal=np.asarray(longitudinal, dtype=np.float32),
        angular=np.asarray(angular, dtype=np.float32),
        metadata={
            "side": side,
            "style": style,
            "ring_count": rings,
            "ring_vertex_count": sides,
            "armhole_curve_length_m": armhole,
            "sleeve_cap_curve_length_m": cap,
            "cap_ease_ratio": measurements.cap_ease_ratio,
            "underarm_seam_vertex_modulo": int(round(0.75 * sides)) % sides,
            "wrist_boundary_open": True,
            "armhole_boundary_open": True,
        },
    )


def build_robe_skirt(joints: dict[str, np.ndarray], rings: int = 12, sides: int = 40) -> ConstructedPrimitive:
    pelvis_y = float(joints["PELVIS"][1])
    calf_y = min(float(joints["L_CALF"][1]), float(joints["R_CALF"][1]))
    centre_z = float(joints["PELVIS"][2])
    positions, normals, texcoords, longitudinal, angular = [], [], [], [], []
    for ring in range(rings):
        t = ring / (rings - 1)
        y = pelvis_y * (1.0 - t) + (calf_y + 0.03) * t
        radius_x = 0.22 * (1.0 - t) + 0.34 * t
        radius_z = 0.13 * (1.0 - t) + 0.22 * t
        for side in range(sides):
            angle = 2.0 * np.pi * side / sides
            positions.append((radius_x * np.cos(angle), y, centre_z + radius_z * np.sin(angle)))
            normal = np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=np.float64)
            normals.append(normal / np.linalg.norm(normal))
            texcoords.append((side / sides, t))
            longitudinal.append(t)
            angular.append(angle)
    return ConstructedPrimitive(
        name="STRAIGHT_ROBE_SKIRT",
        positions=np.asarray(positions, dtype=np.float32),
        normals=np.asarray(normals, dtype=np.float32),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        indices=_tube_indices(rings, sides),
        longitudinal=np.asarray(longitudinal, dtype=np.float32),
        angular=np.asarray(angular, dtype=np.float32),
        metadata={
            "ring_count": rings,
            "ring_vertex_count": sides,
            "waist_boundary_open": True,
            "hem_boundary_open": True,
            "construction": "STRAIGHT_FLARED_ROBE_EXTENSION",
        },
    )
