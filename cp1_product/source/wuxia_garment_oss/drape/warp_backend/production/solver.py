"""Bounded CP3 recovery runner for the four-panel Warp garment pilot.

This module consumes the immutable CP1 static package. It intentionally does
not rebuild the 2D pattern, provider, arrangement, or static ownership. The
recovery path uses upstream Warp for prediction and deterministic host-side
XPBD projections because the exact CP2 source bundle was not materialized in
the execution surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import warp as wp
from scipy.spatial import cKDTree


@wp.kernel
def _predict_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
    gravity_z: float,
    damping: float,
):
    i = wp.tid()
    v = velocities[i] * damping
    v = v + wp.vec3(0.0, 0.0, gravity_z * dt)
    positions[i] = positions[i] + v * dt
    velocities[i] = v


@dataclass(frozen=True)
class ProductionProfile:
    frames: int = 180
    substeps: int = 8
    iterations: int = 12
    fps: float = 60.0
    gravity_z: float = -9.81
    early_damping: float = 0.985
    tail_damping: float = 0.90
    structural_stiffness: float = 0.38
    bending_stiffness: float = 0.08
    seam_stiffness: float = 0.62
    tether_stiffness: float = 0.010
    body_clearance_m: float = 0.004
    seam_target_m: float = 0.0015


@dataclass
class ProductionResult:
    positions_initial: np.ndarray
    positions_final: np.ndarray
    velocities_final: np.ndarray
    triangles: np.ndarray
    panel_ids: np.ndarray
    frame_metrics: list[dict]
    self_contact_metrics: list[dict]
    profile: ProductionProfile


@dataclass
class _RunState:
    positions: np.ndarray
    initial: np.ndarray
    velocities: np.ndarray
    triangles: np.ndarray
    panel_ids: np.ndarray
    inverse_mass: np.ndarray
    structural: np.ndarray
    structural_rest: np.ndarray
    bending: np.ndarray
    bending_rest: np.ndarray
    seam_pairs: np.ndarray
    seam_initial: np.ndarray
    attachment_indices: np.ndarray
    attachment_targets: np.ndarray
    seam_vertices: np.ndarray
    dt: float
    frame_metrics: list[dict]
    self_metrics: list[dict]


def _unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
    )
    edges = np.sort(edges.astype(np.int32), axis=1)
    return np.unique(edges, axis=0)


def _opposite_vertex(triangle: np.ndarray, edge: np.ndarray) -> int:
    for value in triangle:
        if int(value) not in (int(edge[0]), int(edge[1])):
            return int(value)
    raise ValueError("interior edge has no opposite vertex")


def _bending_pairs(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    triangles = arrays["triangles"]
    pairs = []
    for edge, faces in zip(arrays["interior_edges"], arrays["interior_edge_faces"]):
        a = _opposite_vertex(triangles[int(faces[0])], edge)
        b = _opposite_vertex(triangles[int(faces[1])], edge)
        pairs.append((a, b))
    return np.asarray(pairs, dtype=np.int32)


def _rest_lengths(positions: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    return np.linalg.norm(delta.astype(np.float64), axis=1).astype(np.float32)


def _project_pairs(
    positions: np.ndarray,
    pairs: np.ndarray,
    rest: np.ndarray,
    inverse_mass: np.ndarray,
    stiffness: float,
) -> None:
    a, b = pairs[:, 0], pairs[:, 1]
    delta = positions[b] - positions[a]
    length = np.linalg.norm(delta.astype(np.float64), axis=1)
    valid = length > 1.0e-9
    if not np.any(valid):
        return
    av, bv = a[valid], b[valid]
    dv, lv, rv = delta[valid], length[valid], rest[valid]
    wa, wb = inverse_mass[av], inverse_mass[bv]
    denominator = np.maximum(wa + wb, 1.0e-12)
    scale = ((lv - rv) / lv) * float(stiffness)
    correction = dv * scale[:, None]
    accumulator = np.zeros_like(positions)
    counts = np.zeros(len(positions), dtype=np.float32)
    np.add.at(accumulator, av, correction * (wa / denominator)[:, None])
    np.add.at(accumulator, bv, -correction * (wb / denominator)[:, None])
    np.add.at(counts, av, 1.0)
    np.add.at(counts, bv, 1.0)
    occupied = counts > 0.0
    positions[occupied] += accumulator[occupied] / counts[occupied, None]


def _smoothstep(value: float) -> float:
    t = min(1.0, max(0.0, float(value)))
    return t * t * (3.0 - 2.0 * t)


def _seam_target(initial: np.ndarray, frame: int, profile: ProductionProfile) -> np.ndarray:
    closure = _smoothstep(frame / 80.0)
    terminal = np.minimum(initial, profile.seam_target_m)
    return initial * (1.0 - closure) + terminal * closure


def _attachment_strength(frame: int) -> float:
    fade = _smoothstep((frame - 20.0) / 120.0)
    return 0.28 * (1.0 - fade) + 0.055 * fade


def _body_radii(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    levels = np.asarray((0.24, 0.72, 1.06, 1.30, 1.46, 1.585), dtype=np.float64)
    rx = np.asarray((0.282, 0.212, 0.148, 0.166, 0.179, 0.200), dtype=np.float64)
    ry = np.asarray((0.220, 0.162, 0.108, 0.125, 0.139, 0.136), dtype=np.float64)
    return np.interp(z, levels, rx), np.interp(z, levels, ry)


def _project_body(
    positions: np.ndarray,
    panel_ids: np.ndarray,
    clearance: float,
) -> int:
    z = positions[:, 2]
    rx, ry = _body_radii(z)
    target = 1.0 + clearance / np.maximum(np.minimum(rx, ry), 1.0e-6)
    radial = np.sqrt((positions[:, 0] / rx) ** 2 + (positions[:, 1] / ry) ** 2)
    inside = radial < target
    if not np.any(inside):
        return 0
    sign = np.where(np.isin(panel_ids, (0, 2)), 1.0, -1.0)
    tiny = inside & (radial < 1.0e-8)
    positions[tiny, 0] = 0.0
    positions[tiny, 1] = sign[tiny] * ry[tiny] * target[tiny]
    regular = inside & ~tiny
    factor = target[regular] / radial[regular]
    positions[regular, 0] *= factor
    positions[regular, 1] *= factor
    wrong_side = sign * positions[:, 1] < 0.0
    positions[wrong_side, 1] *= -1.0
    return int(np.count_nonzero(inside))


def _project_attachments(
    positions: np.ndarray,
    indices: np.ndarray,
    targets: np.ndarray,
    strength: float,
) -> None:
    positions[indices] += (targets - positions[indices]) * float(strength)


def _predict_with_warp(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    gravity_z: float,
    damping: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = wp.array(positions, dtype=wp.vec3, device="cpu")
    v = wp.array(velocities, dtype=wp.vec3, device="cpu")
    wp.launch(
        _predict_kernel,
        dim=len(positions),
        inputs=[x, v, float(dt), float(gravity_z), float(damping)],
        device="cpu",
    )
    return x.numpy(), v.numpy()


def _sample_self_contact(
    positions: np.ndarray,
    panel_ids: np.ndarray,
    seam_vertices: np.ndarray,
) -> dict:
    ignored = np.zeros(len(positions), dtype=bool)
    ignored[seam_vertices] = True
    front = np.where(np.isin(panel_ids, (0, 2)) & ~ignored)[0][::8]
    back = np.where(np.isin(panel_ids, (1, 3)) & ~ignored)[0][::8]
    if not len(front) or not len(back):
        return {"sampled_pairs_under_2mm": 0, "minimum_sample_distance_m": None}
    tree = cKDTree(positions[back])
    distances, _ = tree.query(positions[front], k=1)
    return {
        "sampled_pairs_under_2mm": int(np.count_nonzero(distances < 0.002)),
        "minimum_sample_distance_m": float(np.min(distances)),
    }


def _frame_metric(
    frame: int,
    previous: np.ndarray,
    positions: np.ndarray,
    seam_pairs: np.ndarray,
    body_contacts: int,
) -> dict:
    displacement = np.linalg.norm((positions - previous).astype(np.float64), axis=1)
    seam_gap = np.linalg.norm(
        (positions[seam_pairs[:, 1]] - positions[seam_pairs[:, 0]]).astype(np.float64),
        axis=1,
    )
    return {
        "frame": int(frame),
        "maximum_displacement_m": float(np.max(displacement)),
        "mean_displacement_m": float(np.mean(displacement)),
        "seam_gap_mean_m": float(np.mean(seam_gap)),
        "seam_gap_p95_m": float(np.quantile(seam_gap, 0.95)),
        "body_projection_count": int(body_contacts),
    }


def _prepare_state(
    arrays: Dict[str, np.ndarray], profile: ProductionProfile
) -> _RunState:
    positions = arrays["positions_initial"].astype(np.float32, copy=True)
    triangles = arrays["triangles"].astype(np.int32, copy=False)
    structural = _unique_edges(triangles)
    bending = _bending_pairs(arrays)
    seam_pairs = arrays["seam_pairs"].astype(np.int32, copy=False)
    return _RunState(
        positions=positions,
        initial=positions.copy(),
        velocities=arrays["velocities_initial"].astype(np.float32, copy=True),
        triangles=triangles,
        panel_ids=arrays["panel_ids"].astype(np.int32, copy=False),
        inverse_mass=arrays["inverse_mass"].astype(np.float32, copy=False),
        structural=structural,
        structural_rest=_rest_lengths(positions, structural),
        bending=bending,
        bending_rest=_rest_lengths(positions, bending),
        seam_pairs=seam_pairs,
        seam_initial=arrays["seam_rest_length"].astype(np.float32, copy=False),
        attachment_indices=arrays["attachment_indices"].astype(np.int32, copy=False),
        attachment_targets=arrays["attachment_targets"].astype(np.float32, copy=False),
        seam_vertices=np.unique(seam_pairs.reshape(-1)),
        dt=1.0 / (profile.fps * profile.substeps),
        frame_metrics=[],
        self_metrics=[],
    )


def _project_iteration(
    state: _RunState,
    profile: ProductionProfile,
    seam_target: np.ndarray,
    support: float,
    iteration: int,
) -> int:
    _project_pairs(
        state.positions,
        state.seam_pairs,
        seam_target,
        state.inverse_mass,
        profile.seam_stiffness,
    )
    _project_attachments(
        state.positions, state.attachment_indices, state.attachment_targets, support
    )
    if iteration % 3 != 0:
        return 0
    _project_pairs(
        state.positions,
        state.structural,
        state.structural_rest,
        state.inverse_mass,
        profile.structural_stiffness,
    )
    _project_pairs(
        state.positions,
        state.bending,
        state.bending_rest,
        state.inverse_mass,
        profile.bending_stiffness,
    )
    state.positions += (state.initial - state.positions) * profile.tether_stiffness
    return _project_body(state.positions, state.panel_ids, profile.body_clearance_m)


def _run_frame(state: _RunState, frame: int, profile: ProductionProfile) -> np.ndarray:
    frame_start = state.positions.copy()
    damping = profile.tail_damping if frame >= 140 else profile.early_damping
    seam_target = _seam_target(state.seam_initial, frame, profile)
    support = _attachment_strength(frame)
    body_contacts = 0
    for _substep in range(profile.substeps):
        before = state.positions.copy()
        state.positions, state.velocities = _predict_with_warp(
            state.positions, state.velocities, state.dt, profile.gravity_z, damping
        )
        for iteration in range(profile.iterations):
            body_contacts += _project_iteration(
                state, profile, seam_target, support, iteration
            )
        state.velocities = (
            (state.positions - before) / state.dt
        ).astype(np.float32) * damping
    state.frame_metrics.append(
        _frame_metric(frame, frame_start, state.positions, state.seam_pairs, body_contacts)
    )
    state.self_metrics.append(
        _sample_self_contact(state.positions, state.panel_ids, state.seam_vertices)
    )
    return frame_start


def _write_checkpoint(
    output_dir: Path, state: _RunState, frame: int, previous: np.ndarray
) -> None:
    if frame % 20 != 0:
        return
    np.savez_compressed(
        output_dir / "simulation_state.npz",
        positions=state.positions.astype(np.float32),
        velocities=state.velocities.astype(np.float32),
        previous_positions=previous.astype(np.float32),
        frame_completed=np.asarray([frame], dtype=np.int32),
    )


def run_production(
    arrays: Dict[str, np.ndarray],
    output_dir: Path,
    profile: ProductionProfile | None = None,
) -> ProductionResult:
    profile = profile or ProductionProfile()
    wp.init()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _prepare_state(arrays, profile)
    for frame in range(1, profile.frames + 1):
        previous = _run_frame(state, frame, profile)
        _write_checkpoint(output_dir, state, frame, previous)
    _project_body(state.positions, state.panel_ids, profile.body_clearance_m)
    return ProductionResult(
        positions_initial=state.initial,
        positions_final=state.positions,
        velocities_final=state.velocities,
        triangles=state.triangles,
        panel_ids=state.panel_ids,
        frame_metrics=state.frame_metrics,
        self_contact_metrics=state.self_metrics,
        profile=profile,
    )
