"""Load the immutable fitted tunic mesh and static constraint authority."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionMesh:
    positions: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    edge_rest_lengths: np.ndarray
    seam_pairs: np.ndarray
    seam_rest_lengths: np.ndarray
    attachment_indices: np.ndarray
    source_paths: dict[str, str]
    source_keys: dict[str, str]

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])


def _first_array(data, names: tuple[str, ...], dimensions: int | None = None):
    for name in names:
        if name not in data.files:
            continue
        value = np.asarray(data[name])
        if dimensions is None or value.ndim == dimensions:
            return name, value
    return None, None


def _require_vectors(data, names: tuple[str, ...], label: str) -> tuple[str, np.ndarray]:
    name, value = _first_array(data, names, 2)
    if value is None or value.shape[1] != 3:
        raise ValueError(f"missing Nx3 {label}; available={sorted(data.files)}")
    return name, np.asarray(value, dtype=np.float64)


def _require_triangles(data) -> tuple[str, np.ndarray]:
    name, value = _first_array(data, ("triangles", "faces", "indices"), 2)
    if value is None or value.shape[1] != 3:
        raise ValueError(f"missing triangle array; available={sorted(data.files)}")
    triangles = np.asarray(value, dtype=np.int64)
    if triangles.size and int(triangles.min()) < 0:
        raise ValueError("negative triangle index")
    return name, triangles


def unique_edges(triangles: np.ndarray) -> np.ndarray:
    pairs = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0).astype(np.int64)


def _seam_pairs(static) -> tuple[str, np.ndarray]:
    name, value = _first_array(static, ("seam_pairs", "seams", "seam_vertex_pairs"), 2)
    if value is not None and value.shape[1] == 2:
        return name, np.asarray(value, dtype=np.int64)
    required = ("seam_a_vertices", "seam_a_weights", "seam_b_vertices", "seam_b_weights")
    if not all(item in static.files for item in required):
        return "NONE", np.empty((0, 2), dtype=np.int64)
    a_vertices = np.asarray(static["seam_a_vertices"], dtype=np.int64)
    a_weights = np.asarray(static["seam_a_weights"], dtype=np.float64)
    b_vertices = np.asarray(static["seam_b_vertices"], dtype=np.int64)
    b_weights = np.asarray(static["seam_b_weights"], dtype=np.float64)
    if a_vertices.ndim != 2 or b_vertices.ndim != 2:
        return "NONE", np.empty((0, 2), dtype=np.int64)
    a = a_vertices[np.arange(len(a_vertices)), np.argmax(a_weights, axis=1)]
    b = b_vertices[np.arange(len(b_vertices)), np.argmax(b_weights, axis=1)]
    return "BARYCENTRIC_DOMINANT_VERTEX", np.column_stack((a, b)).astype(np.int64)


def _attachment_indices(static) -> tuple[str, np.ndarray]:
    name, value = _first_array(
        static,
        ("attachment_indices", "attachment_vertices", "pinned_indices"),
        1,
    )
    if value is None:
        return "DERIVED_UPPER_BOUNDARY", np.empty(0, dtype=np.int64)
    return name, np.unique(np.asarray(value, dtype=np.int64))


def _select_paths(root: Path) -> tuple[Path, Path]:
    final_candidates = (
        root / "build/tunic_pilot/warp_cp3/final_simulation_mesh.npz",
        root / "build/tunic_pilot/warp_cp3/simulation_state.npz",
    )
    static_candidates = (
        root / "build/tunic_pilot/warp_cp1/warp_model_package.npz",
        root / "build/tunic_pilot/native_input_mesh.npz",
    )
    final = next((item for item in final_candidates if item.is_file()), None)
    static = next((item for item in static_candidates if item.is_file()), None)
    if final is None or static is None:
        raise FileNotFoundError(f"motion-fit mesh authority missing: final={final}, static={static}")
    return final, static


def load_motion_mesh(root: Path) -> MotionMesh:
    final_path, static_path = _select_paths(root)
    with np.load(final_path, allow_pickle=False) as final, np.load(static_path, allow_pickle=False) as static:
        position_key, positions = _require_vectors(
            final,
            ("positions", "final_positions", "vertices", "x"),
            "final positions",
        )
        triangle_key, triangles = _require_triangles(final if any(name in final.files for name in ("triangles", "faces", "indices")) else static)
        seam_key, seams = _seam_pairs(static)
        attachment_key, attachments = _attachment_indices(static)
        static_keys = sorted(static.files)
        final_keys = sorted(final.files)
    if triangles.size and int(triangles.max()) >= len(positions):
        raise ValueError("triangle index exceeds fitted mesh vertex count")
    if seams.size:
        valid = np.all((seams >= 0) & (seams < len(positions)), axis=1)
        seams = np.unique(seams[valid], axis=0)
    if not attachments.size:
        z = positions[:, 2]
        x = positions[:, 0]
        upper = z >= np.quantile(z, 0.90)
        lateral = np.abs(x - np.median(x)) >= np.quantile(np.abs(x - np.median(x)), 0.55)
        attachments = np.flatnonzero(upper & lateral).astype(np.int64)
    attachments = attachments[(attachments >= 0) & (attachments < len(positions))]
    edges = unique_edges(triangles)
    rest_lengths = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
    valid_edges = rest_lengths > 1.0e-8
    edges = edges[valid_edges]
    rest_lengths = rest_lengths[valid_edges]
    seam_rest = (
        np.linalg.norm(positions[seams[:, 1]] - positions[seams[:, 0]], axis=1)
        if seams.size
        else np.empty(0, dtype=np.float64)
    )
    return MotionMesh(
        positions=np.asarray(positions, dtype=np.float64),
        triangles=triangles,
        edges=edges,
        edge_rest_lengths=rest_lengths,
        seam_pairs=seams,
        seam_rest_lengths=seam_rest,
        attachment_indices=np.unique(attachments),
        source_paths={"final_mesh": str(final_path.relative_to(root)), "static_model": str(static_path.relative_to(root))},
        source_keys={
            "positions": position_key,
            "triangles": triangle_key,
            "seams": seam_key,
            "attachments": attachment_key,
            "final_npz_keys": ",".join(final_keys),
            "static_npz_keys": ",".join(static_keys),
        },
    )
