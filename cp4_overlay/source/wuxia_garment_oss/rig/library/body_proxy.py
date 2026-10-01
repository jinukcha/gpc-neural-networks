"""CP4-only semantic body proxy derived from the CP0 canonical skeleton.

The diet CP3 bundle retains the accepted skeleton contract but not the CP0 body
arrays. This deterministic proxy exists only for body-occlusion masks and is
never presented as the original body mesh.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REGION_ID = {
    "TORSO": 0,
    "NECK": 1,
    "HEAD": 2,
    "LEFT_ARM": 3,
    "RIGHT_ARM": 4,
    "LEFT_HAND": 5,
    "RIGHT_HAND": 6,
    "LEFT_LEG": 7,
    "RIGHT_LEG": 8,
    "LEFT_FOOT": 9,
    "RIGHT_FOOT": 10,
}


def _ellipsoid(centre, radii, region_id, sides=24, rings=12):
    centre = np.asarray(centre, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    vertices = [centre + np.array([0.0, radii[1], 0.0])]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        for side in range(sides):
            theta = 2.0 * np.pi * side / sides
            unit = np.array([
                np.sin(phi) * np.cos(theta),
                np.cos(phi),
                np.sin(phi) * np.sin(theta),
            ])
            vertices.append(centre + radii * unit)
    bottom = len(vertices)
    vertices.append(centre - np.array([0.0, radii[1], 0.0]))
    triangles = [(0, 1 + side, 1 + (side + 1) % sides) for side in range(sides)]
    for ring in range(rings - 2):
        lower = 1 + ring * sides
        upper = lower + sides
        for side in range(sides):
            next_side = (side + 1) % sides
            triangles.extend(((lower + side, upper + side, upper + next_side), (lower + side, upper + next_side, lower + next_side)))
    last = 1 + (rings - 2) * sides
    triangles.extend((bottom, last + (side + 1) % sides, last + side) for side in range(sides))
    positions = np.asarray(vertices, dtype=np.float64)
    regions = np.full(len(positions), region_id, dtype=np.int16)
    return positions, np.asarray(triangles, dtype=np.int32), regions


def _joints(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("global_joint_positions")
    if not source:
        source = {item["semantic_id"]: item["global_translation"] for item in payload["bones"]}
    return {name: np.asarray(value, dtype=np.float64) for name, value in source.items()}


def _arm_spec(joints, prefix, region_name, hand_name):
    shoulder = joints[f"{prefix}_CLAVICLE"]
    hand = joints[f"{prefix}_HAND"]
    centre = (shoulder + hand) * 0.5
    length = abs(float(hand[0] - shoulder[0]))
    sign = -1.0 if prefix == "L" else 1.0
    return (
        (centre, (max(length * 0.55, 0.18), 0.065, 0.055), REGION_ID[region_name]),
        (hand + np.array([0.035 * sign, -0.01, 0.0]), (0.060, 0.040, 0.028), REGION_ID[hand_name]),
    )


def _leg_spec(joints, prefix, region_name, foot_name):
    sign = -1.0 if prefix == "L" else 1.0
    hip = joints["PELVIS"] + np.array([0.075 * sign, -0.02, 0.0])
    foot = joints[f"{prefix}_FOOT"]
    toe = joints[f"{prefix}_TOE"]
    centre = (hip + foot) * 0.5
    half_height = max(abs(float(hip[1] - foot[1])) * 0.52, 0.30)
    return (
        (centre, (0.085, half_height, 0.075), REGION_ID[region_name]),
        ((foot + toe) * 0.5, (0.065, 0.048, 0.115), REGION_ID[foot_name]),
    )


def build_semantic_occlusion_proxy(skeleton_path: Path):
    joints = _joints(skeleton_path)
    torso_centre = (joints["PELVIS"] + joints["CHEST"]) * 0.5 + np.array([0.0, 0.02, 0.0])
    specs = [
        (torso_centre, (0.215, 0.355, 0.135), REGION_ID["TORSO"]),
        ((joints["CHEST"] + joints["HEAD"]) * 0.5, (0.067, 0.105, 0.062), REGION_ID["NECK"]),
        (joints["HEAD"] + np.array([0.0, 0.07, 0.0]), (0.100, 0.140, 0.108), REGION_ID["HEAD"]),
    ]
    specs.extend(_arm_spec(joints, "L", "LEFT_ARM", "LEFT_HAND"))
    specs.extend(_arm_spec(joints, "R", "RIGHT_ARM", "RIGHT_HAND"))
    specs.extend(_leg_spec(joints, "L", "LEFT_LEG", "LEFT_FOOT"))
    specs.extend(_leg_spec(joints, "R", "RIGHT_LEG", "RIGHT_FOOT"))
    parts = []
    vertex_offset = 0
    for centre, radii, region_id in specs:
        positions, triangles, regions = _ellipsoid(centre, radii, region_id)
        parts.append((positions, triangles + vertex_offset, regions))
        vertex_offset += len(positions)
    return (
        np.concatenate([item[0] for item in parts], axis=0),
        np.concatenate([item[1] for item in parts], axis=0),
        np.concatenate([item[2] for item in parts], axis=0),
    )
