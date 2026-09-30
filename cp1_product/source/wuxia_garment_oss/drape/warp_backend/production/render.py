"""PNG evidence rendering for the CP3 garment recovery run."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .solver import ProductionResult, _body_radii


def _project(points: np.ndarray, view: str) -> Tuple[np.ndarray, np.ndarray]:
    if view == "front":
        return points[:, (0, 2)], points[:, 1]
    if view == "back":
        return points[:, (0, 2)], -points[:, 1]
    if view == "left":
        return points[:, (1, 2)], -points[:, 0]
    if view == "right":
        return points[:, (1, 2)], points[:, 0]
    raise ValueError(f"unknown view: {view}")


def _body_outline(view: str) -> np.ndarray:
    z = np.linspace(0.24, 1.585, 240)
    rx, ry = _body_radii(z)
    radius = rx if view in ("front", "back") else ry
    left = np.column_stack((-radius, z))
    right = np.column_stack((radius[::-1], z[::-1]))
    return np.vstack((left, right))


def _draw_mesh(
    axis,
    positions: np.ndarray,
    triangles: np.ndarray,
    panel_ids: np.ndarray,
    view: str,
    title: str,
) -> None:
    projected, depth = _project(positions, view)
    sample = triangles[::3]
    polygons = projected[sample]
    face_panel = panel_ids[sample[:, 0]]
    face_depth = depth[sample].mean(axis=1)
    order = np.argsort(face_depth)
    shade = 0.48 + 0.10 * (face_panel % 2) + 0.08 * np.tanh(face_depth * 6.0)
    colors = np.column_stack((shade, shade, shade, np.full_like(shade, 0.94)))
    collection = PolyCollection(
        polygons[order],
        facecolors=colors[order],
        edgecolors="none",
        rasterized=True,
    )
    axis.add_collection(collection)
    outline = _body_outline(view)
    axis.fill(outline[:, 0], outline[:, 1], alpha=0.10)
    axis.plot(outline[:, 0], outline[:, 1], linewidth=0.65, alpha=0.55)
    axis.set_aspect("equal", adjustable="box")
    axis.autoscale_view()
    axis.set_title(title)
    axis.set_xlabel("lateral / depth (m)")
    axis.set_ylabel("height (m)")
    axis.grid(alpha=0.15)


def _draw_convergence(axis, result: ProductionResult) -> None:
    frames = np.asarray([row["frame"] for row in result.frame_metrics])
    maximum = np.asarray([row["maximum_displacement_m"] for row in result.frame_metrics])
    mean = np.asarray([row["mean_displacement_m"] for row in result.frame_metrics])
    seam = np.asarray([row["seam_gap_p95_m"] for row in result.frame_metrics])
    axis.plot(frames, maximum, label="max displacement")
    axis.plot(frames, mean, label="mean displacement")
    axis.plot(frames, seam, label="seam p95")
    axis.axvspan(161, 180, alpha=0.10, label="review tail")
    axis.set_yscale("log")
    axis.set_xlabel("frame")
    axis.set_ylabel("metres")
    axis.set_title("Frame 1–180 convergence trace")
    axis.grid(alpha=0.20)
    axis.legend(fontsize=8)


def _draw_metric_text(axis, qualification: dict) -> None:
    body = qualification["body_contact"]
    edge = qualification["edge_strain"]
    seam = qualification["seam_closure"]
    conv = qualification["convergence"]
    proxy = qualification["self_contact"]
    lines = [
        "CP3 recovery qualification",
        f"numeric_pass: {qualification['numeric_pass']}",
        f"technical_pass: {qualification['technical_pass']}",
        "",
        f"body penetration max: {body['penetration_max_m']:.6f} m",
        f"body penetrated ratio: {body['penetrated_vertex_ratio']:.6f}",
        f"edge stretch p99 / max: {edge['edge_stretch_p99']:.4f} / {edge['edge_stretch_max']:.4f}",
        f"seam mean / p95: {seam['seam_gap_mean_m']:.6f} / {seam['seam_gap_p95_m']:.6f} m",
        f"tail final max movement: {conv['tail_final_max_displacement_m']:.6f} m",
        f"sample self-contact peak: {proxy['tail_peak_sampled_pairs_under_2mm']}",
        "",
        "Authority gate: HOLD",
        "Exact CP2 source bytes unavailable;",
        "self-intersection gate is a bounded proxy.",
    ]
    axis.axis("off")
    axis.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)


def write_evidence(
    result: ProductionResult,
    qualification: dict,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = output_dir / "cp3_frame180_evidence.png"
    convergence = output_dir / "cp3_convergence.png"

    figure, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=140)
    _draw_mesh(
        axes[0, 0], result.positions_initial, result.triangles, result.panel_ids,
        "front", "Frame 0 — front"
    )
    _draw_mesh(
        axes[0, 1], result.positions_final, result.triangles, result.panel_ids,
        "front", "Frame 180 — front"
    )
    _draw_mesh(
        axes[0, 2], result.positions_final, result.triangles, result.panel_ids,
        "back", "Frame 180 — back"
    )
    _draw_mesh(
        axes[1, 0], result.positions_final, result.triangles, result.panel_ids,
        "left", "Frame 180 — left"
    )
    _draw_mesh(
        axes[1, 1], result.positions_final, result.triangles, result.panel_ids,
        "right", "Frame 180 — right"
    )
    _draw_metric_text(axes[1, 2], qualification)
    figure.suptitle(
        "OSS-R0A-CP2B-R4-REV1 / CP3 — Warp garment run evidence",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(evidence, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 7), dpi=160)
    _draw_convergence(axis, result)
    figure.tight_layout()
    figure.savefig(convergence, bbox_inches="tight")
    plt.close(figure)
    return evidence, convergence
