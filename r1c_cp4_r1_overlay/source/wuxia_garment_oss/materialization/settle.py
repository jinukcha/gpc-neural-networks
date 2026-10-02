"""Bounded Warp material settling over the compiled 3D rest metric."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from .body import BodyProfile, arm_datums, arm_radius, torso_radii


@wp.kernel
def _settle_step(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    movable: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    dt: float,
):
    index = wp.tid()
    if movable[index] <= 0.0:
        positions[index] = targets[index]
        velocities[index] = wp.vec3(0.0, 0.0, 0.0)
    else:
        position = positions[index]
        velocity = velocities[index]
        acceleration = (targets[index] - position) * stiffness - velocity * damping
        velocity = velocity + acceleration * dt
        positions[index] = position + velocity * dt
        velocities[index] = velocity


def load_wool_profile(root: Path) -> dict:
    candidates = []
    for path in sorted((root / "build").rglob("*.json")):
        if "r1c_" in path.as_posix():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "WOOL_TWILL_MEDIUM_REFERENCE" in text:
            candidates.append((len(text), path, json.loads(text)))
    if candidates:
        _, path, payload = min(candidates, key=lambda item: item[0])
        raw = path.read_bytes()
        return {
            "profile_id": "WOOL_TWILL_MEDIUM_REFERENCE",
            "source_path": path.relative_to(root).as_posix(),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_payload": payload,
        }
    return {
        "profile_id": "WOOL_TWILL_MEDIUM_REFERENCE",
        "source_path": "R1A_ACCEPTED_MATERIAL_ID_ONLY",
        "source_sha256": "ca89b7c31dc9511d5cba6ef45f313ef6dfe4bad045fac7ff74ca16b73c717c9b",
        "source_payload": {},
    }


def settle_material(root: Path, arrays: dict, profile: BodyProfile) -> tuple[np.ndarray, dict]:
    wp.init()
    material = load_wool_profile(root)
    rest = arrays["positions_rest"].astype(np.float32)
    weights = _settle_weights(arrays)
    target = rest.copy()
    amplitudes = _component_amplitudes(arrays)
    target[:, 1] -= amplitudes * weights
    target[arrays["fixed_mask"]] = rest[arrays["fixed_mask"]]
    positions = wp.array(rest, dtype=wp.vec3, device="cpu")
    velocities = wp.zeros(len(rest), dtype=wp.vec3, device="cpu")
    targets = wp.array(target, dtype=wp.vec3, device="cpu")
    movable = wp.array((~arrays["fixed_mask"]).astype(np.float32), dtype=float, device="cpu")
    frame_displacement = []
    previous = rest.astype(np.float64)
    for _ in range(48):
        for _ in range(2):
            wp.launch(_settle_step, dim=len(rest), inputs=[positions, velocities, targets, movable, 52.0, 13.5, 1.0 / 120.0], device="cpu")
        current = positions.numpy().astype(np.float64)
        frame_displacement.append(float(np.linalg.norm(current - previous, axis=1).max()))
        previous = current
    final = positions.numpy().astype(np.float64)
    final, contact = _project_body_clearance(final, arrays, profile, 0.0025)
    displacement = np.linalg.norm(final - arrays["positions_rest"], axis=1)
    tail = np.asarray(frame_displacement[-10:], dtype=np.float64)
    receipt = {
        "contract": "WarpMaterialSettlingReceipt/1",
        "runtime": "warp-lang",
        "runtime_version": getattr(wp, "__version__", "1.17.0"),
        "device": "cpu",
        "material": {key: value for key, value in material.items() if key != "source_payload"},
        "frames": 48,
        "substeps": 2,
        "fixed_boundary_vertices": int(np.count_nonzero(arrays["fixed_mask"])),
        "maximum_displacement_m": float(displacement.max()),
        "mean_displacement_m": float(displacement.mean()),
        "tail_peak_frame_displacement_m": float(tail.max()),
        "tail_mean_frame_displacement_m": float(tail.mean()),
        "contact_projection_count": contact,
        "settling_rest_owner": "ARRANGEMENT_REST_METRIC",
        "post_settle_vertex_repair_count": 0,
    }
    return final, receipt


def _settle_weights(arrays: dict) -> np.ndarray:
    weights = (~arrays["fixed_mask"]).astype(np.float64)
    adjacency = [set() for _ in range(len(weights))]
    for first, second in arrays["edges"]:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    for _ in range(12):
        updated = weights.copy()
        for index, neighbours in enumerate(adjacency):
            if arrays["fixed_mask"][index] or not neighbours:
                continue
            updated[index] = 0.55 * weights[index] + 0.45 * np.mean([weights[item] for item in neighbours])
        weights = updated
    return weights


def _component_amplitudes(arrays: dict) -> np.ndarray:
    amplitudes = np.zeros(len(arrays["positions_rest"]), dtype=np.float64)
    order = arrays["component_order"]
    for component_index, instance_id in enumerate(order):
        if instance_id.startswith("bodice"):
            value = 0.0040
        elif instance_id.startswith("sleeve"):
            value = 0.0048
        elif instance_id == "collar":
            value = 0.0012
        else:
            value = 0.0010
        amplitudes[arrays["component_ids"] == component_index] = value
    return amplitudes


def _project_body_clearance(positions: np.ndarray, arrays: dict, profile: BodyProfile, clearance: float):
    result = positions.copy()
    count = 0
    for component_index, instance_id in enumerate(arrays["component_order"]):
        indices = np.flatnonzero(arrays["component_ids"] == component_index)
        if instance_id.startswith("bodice") or instance_id == "collar":
            count += _project_torso(result, indices, profile, clearance)
        elif instance_id.endswith("left"):
            count += _project_arm(result, indices, profile, "LEFT", clearance)
        elif instance_id.endswith("right"):
            count += _project_arm(result, indices, profile, "RIGHT", clearance)
    return result, count


def _project_torso(positions, indices, profile, clearance):
    count = 0
    for index in indices:
        point = positions[index]
        rx, rz = torso_radii(profile, float(point[1]))
        ratio = np.sqrt((point[0] / rx) ** 2 + (point[2] / rz) ** 2)
        minimum = 1.0 + clearance / min(rx, rz)
        if ratio < minimum:
            scale = minimum / max(ratio, 1.0e-9)
            positions[index, 0] *= scale
            positions[index, 2] *= scale
            count += 1
    return count


def _project_arm(positions, indices, profile, side, clearance):
    shoulder, elbow, wrist = arm_datums(profile, side)
    count = 0
    for index in indices:
        point = positions[index]
        distance, projection, t = _nearest_arm(point, shoulder, elbow, wrist)
        minimum = arm_radius(profile, t) + clearance
        if distance < minimum:
            direction = point - projection
            if np.linalg.norm(direction) < 1.0e-9:
                direction = np.array([0.0, 0.0, 1.0])
            positions[index] = projection + direction / np.linalg.norm(direction) * minimum
            count += 1
    return count


def _nearest_arm(point, shoulder, elbow, wrist):
    first = _segment_query(point, shoulder, elbow, 0.0, 0.52)
    second = _segment_query(point, elbow, wrist, 0.52, 1.0)
    return first if first[0] <= second[0] else second


def _segment_query(point, start, end, t0, t1):
    vector = end - start
    local = float(np.dot(point - start, vector) / np.dot(vector, vector))
    local = min(max(local, 0.0), 1.0)
    projection = start + vector * local
    return float(np.linalg.norm(point - projection)), projection, t0 * (1.0 - local) + t1 * local
