"""Render actual CP6 manufacturing, game-product, and trousers-fit evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..garments.trousers.fit import TrousersFitResult
from ..garments.trousers.mesh import TrousersMesh


def _panel_plot(ax, package: Mapping[str, object], title: str) -> None:
    for panel in package["panels"]:
        for curve in panel["curves"]:
            cut = np.asarray(curve["cut_line"], dtype=np.float64)
            stitch = np.asarray(curve["stitch_line"], dtype=np.float64)
            ax.plot(cut[:, 0], cut[:, 1], linewidth=1.0)
            ax.plot(stitch[:, 0], stitch[:, 1], linewidth=0.55, linestyle="--")
        label = panel["label_position"]
        ax.text(float(label[0]), float(label[1]), str(panel["panel_id"]), fontsize=7, ha="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("metres")
    ax.set_ylabel("metres")
    ax.grid(True, alpha=0.18)


def render_manufacturing_board(
    path: Path,
    tunic_package: Mapping[str, object],
    trousers_package: Mapping[str, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(25, 15), constrained_layout=True)
    _panel_plot(axes[0], tunic_package, "Sleeveless tunic — stitch / cut / allowance")
    _panel_plot(axes[1], trousers_package, "Trousers — waistband / darts / gusset / leg panels")
    fig.suptitle("GARMENT-CAD-PRO-R1A / CP6 — manufacturing 2D export", fontsize=19)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _sample_faces(triangles: np.ndarray, maximum: int = 12000) -> np.ndarray:
    if len(triangles) <= maximum:
        return triangles
    indices = np.linspace(0, len(triangles) - 1, maximum, dtype=np.int64)
    return triangles[indices]


def _mesh_view(ax, positions: np.ndarray, triangles: np.ndarray, view: str, title: str) -> None:
    faces = _sample_faces(triangles)
    if view == "front":
        x, y = positions[:, 0], positions[:, 2]
    else:
        x, y = positions[:, 1], positions[:, 2]
    ax.triplot(x, y, faces, linewidth=0.16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.axis("off")


def render_game_product_board(
    path: Path,
    tunic_positions: np.ndarray,
    tunic_triangles: np.ndarray,
    trousers: TrousersMesh,
    tunic_morph_count: int,
    trousers_morph_count: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(25, 16), constrained_layout=True)
    _mesh_view(axes[0, 0], tunic_positions, tunic_triangles, "front", "Tunic — front")
    _mesh_view(axes[0, 1], tunic_positions, tunic_triangles, "side", "Tunic — side")
    _mesh_view(axes[1, 0], trousers.positions, trousers.triangles, "front", "Trousers — front")
    _mesh_view(axes[1, 1], trousers.positions, trousers.triangles, "side", "Trousers — side")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.02, 0.98,
        "TUNIC GAME PRODUCT\n\n"
        f"vertices: {len(tunic_positions):,}\n"
        f"triangles: {len(tunic_triangles):,}\n"
        f"motion morph targets: {tunic_morph_count}\n"
        "material: neutral gray\n"
        "coordinate export: glTF Y-up",
        va="top", family="monospace", fontsize=12,
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02, 0.98,
        "TROUSERS GAME PRODUCT\n\n"
        f"shell vertices: {len(trousers.positions):,}\n"
        f"shell triangles: {len(trousers.triangles):,}\n"
        f"waistband triangles: {len(trousers.waistband_triangles):,}\n"
        f"gusset triangles: {len(trousers.gusset_triangles):,}\n"
        f"motion morph targets: {trousers_morph_count}\n"
        "material: neutral gray",
        va="top", family="monospace", fontsize=12,
    )
    fig.suptitle("CP6 neutral-gray game garment products", fontsize=19)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fit_record(results: Sequence[TrousersFitResult], material_id: str, pose_id: str) -> TrousersFitResult:
    return next(item for item in results if item.material_id == material_id and item.pose_id == pose_id)


def _scatter_map(ax, result: TrousersFitResult, map_id: str, title: str) -> None:
    values = result.maps[map_id]
    positions = result.positions
    image = ax.scatter(positions[:, 0], positions[:, 2], c=values, s=2.0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def render_trousers_fit_board(
    path: Path,
    results: Sequence[TrousersFitResult],
    material_id: str = "WOOL_TWILL_MEDIUM_REFERENCE",
) -> None:
    poses = ("SEATED", "SQUAT", "WALK_STRIDE")
    fig, axes = plt.subplots(3, 3, figsize=(24, 20), constrained_layout=True)
    for row, pose_id in enumerate(poses):
        result = _fit_record(results, material_id, pose_id)
        _scatter_map(axes[row, 0], result, "strain_ratio", f"{pose_id} — strain")
        _scatter_map(axes[row, 1], result, "pressure_pa", f"{pose_id} — pressure")
        axes[row, 2].axis("off")
        metrics = result.metrics
        gates = result.receipt["gates"]
        lines = [
            f"material: {material_id}",
            f"pose: {pose_id}",
            f"result: {'PASS' if result.receipt['pose_pass'] else 'FAIL'}",
            "",
            f"strain p99: {metrics['strain_p99_ratio']:.5f}",
            f"pressure p99: {metrics['pressure_p99_kpa']:.4f} kPa",
            f"clearance min: {metrics['clearance_min_m'] * 1000.0:.3f} mm",
            f"mobility p95: {metrics['mobility_restriction_p95']:.5f}",
            f"tail movement: {metrics['tail_peak_displacement_m'] * 1000.0:.3f} mm",
            "",
            *[f"{name}: {value}" for name, value in gates.items()],
        ]
        axes[row, 2].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.suptitle("CP6 trousers sit / squat / stride qualification", fontsize=19)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
