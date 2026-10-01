"""Canonical per-vertex motion-fit maps and aggregate metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body import BodyEnvelope, clearance_and_normal
from .mesh import MotionMesh
from .poses import MotionFitSuite, MotionPose
from .solver import MotionSolveResult, material_controls


@dataclass(frozen=True)
class FitMaps:
    stress_n_m: np.ndarray
    strain_ratio: np.ndarray
    pressure_pa: np.ndarray
    clearance_m: np.ndarray
    seam_tension_n_m: np.ndarray
    contact_persistence: np.ndarray
    mobility_restriction: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "stress_n_m": self.stress_n_m,
            "strain_ratio": self.strain_ratio,
            "pressure_pa": self.pressure_pa,
            "clearance_m": self.clearance_m,
            "seam_tension_n_m": self.seam_tension_n_m,
            "contact_persistence": self.contact_persistence,
            "mobility_restriction": self.mobility_restriction,
        }


def _edge_to_vertex_max(vertex_count: int, edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.zeros(vertex_count, dtype=np.float64)
    np.maximum.at(output, edges[:, 0], values)
    np.maximum.at(output, edges[:, 1], values)
    return output


def _edge_strain(mesh: MotionMesh, positions: np.ndarray) -> np.ndarray:
    current = np.linalg.norm(positions[mesh.edges[:, 1]] - positions[mesh.edges[:, 0]], axis=1)
    return (current - mesh.edge_rest_lengths) / mesh.edge_rest_lengths


def _seam_tension_map(mesh: MotionMesh, positions: np.ndarray, stiffness: float) -> np.ndarray:
    output = np.zeros(mesh.vertex_count, dtype=np.float64)
    if not mesh.seam_pairs.size:
        return output
    current = np.linalg.norm(
        positions[mesh.seam_pairs[:, 1]] - positions[mesh.seam_pairs[:, 0]],
        axis=1,
    )
    extension = np.maximum(current - mesh.seam_rest_lengths, 0.0)
    normalized = extension / np.maximum(mesh.seam_rest_lengths, 0.002)
    tension = stiffness * normalized
    np.maximum.at(output, mesh.seam_pairs[:, 0], tension)
    np.maximum.at(output, mesh.seam_pairs[:, 1], tension)
    return output


def _pressure_map(result: MotionSolveResult, controls) -> np.ndarray:
    thickness = max(controls.thickness_m, 1.0e-5)
    compression = np.clip(result.maximum_contact_correction_m / thickness, 0.0, 0.45)
    return controls.compression_scale * (np.exp(controls.compression_exponent * compression) - 1.0)


def _mobility_map(mesh: MotionMesh, result: MotionSolveResult) -> np.ndarray:
    requested = np.linalg.norm(result.target_positions - mesh.positions, axis=1)
    residual = np.linalg.norm(result.target_positions - result.positions, axis=1)
    restriction = np.zeros(mesh.vertex_count, dtype=np.float64)
    active = requested > 0.002
    restriction[active] = residual[active] / requested[active]
    restriction *= np.clip(result.drive_weights, 0.0, 1.0)
    return np.clip(restriction, 0.0, 1.0)


def compute_fit_maps(
    mesh: MotionMesh,
    envelope: BodyEnvelope,
    pose: MotionPose,
    suite: MotionFitSuite,
    material_profile: dict,
    result: MotionSolveResult,
) -> FitMaps:
    controls = material_controls(material_profile)
    strain_edges = _edge_strain(mesh, result.positions)
    vertex_strain = _edge_to_vertex_max(mesh.vertex_count, mesh.edges, np.abs(strain_edges))
    stress = vertex_strain * controls.effective_tensile_n_m
    clearance, _ = clearance_and_normal(result.positions, envelope, pose)
    pressure = _pressure_map(result, controls)
    seam_tension = _seam_tension_map(mesh, result.positions, controls.effective_tensile_n_m)
    total_contact_steps = suite.frames_per_pose * suite.substeps_per_frame * suite.projection_iterations
    persistence = result.contact_frame_count.astype(np.float64) / max(total_contact_steps, 1)
    mobility = _mobility_map(mesh, result)
    return FitMaps(stress, vertex_strain, pressure, clearance, seam_tension, persistence, mobility)


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    return float(np.percentile(finite, percentile))


def map_metrics(fit_maps: FitMaps, result: MotionSolveResult) -> dict[str, float]:
    return {
        "stress_p95_n_m": _percentile(fit_maps.stress_n_m, 95.0),
        "stress_p99_n_m": _percentile(fit_maps.stress_n_m, 99.0),
        "strain_p95_ratio": _percentile(fit_maps.strain_ratio, 95.0),
        "strain_p99_ratio": _percentile(fit_maps.strain_ratio, 99.0),
        "strain_max_ratio": float(np.max(fit_maps.strain_ratio)),
        "pressure_p95_kpa": _percentile(fit_maps.pressure_pa, 95.0) / 1000.0,
        "pressure_p99_kpa": _percentile(fit_maps.pressure_pa, 99.0) / 1000.0,
        "pressure_max_kpa": float(np.max(fit_maps.pressure_pa)) / 1000.0,
        "clearance_p01_m": _percentile(fit_maps.clearance_m, 1.0),
        "clearance_min_m": float(np.min(fit_maps.clearance_m)),
        "seam_tension_p95_n_m": _percentile(fit_maps.seam_tension_n_m, 95.0),
        "seam_tension_max_n_m": float(np.max(fit_maps.seam_tension_n_m)),
        "contact_persistence_p99": _percentile(fit_maps.contact_persistence, 99.0),
        "mobility_restriction_p95": _percentile(fit_maps.mobility_restriction, 95.0),
        "mobility_restriction_max": float(np.max(fit_maps.mobility_restriction)),
        "final_frame_max_displacement_m": float(result.frame_max_displacement_m[-1]),
        "tail_peak_displacement_m": float(np.max(result.frame_max_displacement_m[-3:])),
        "non_finite_map_count": float(
            sum(np.count_nonzero(~np.isfinite(value)) for value in fit_maps.arrays().values())
        ),
    }
