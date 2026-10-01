"""Render actual motion-fit positions and canonical per-vertex maps."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .poses import MotionFitSuite


_MAP_SPECS = (
    ("stress_n_m", "Stress (N/m)", "magma"),
    ("strain_ratio", "Strain ratio", "viridis"),
    ("pressure_pa", "Pressure (Pa)", "inferno"),
    ("clearance_m", "Clearance (m)", "coolwarm"),
    ("seam_tension_n_m", "Seam tension (N/m)", "plasma"),
    ("contact_persistence", "Contact persistence", "cividis"),
    ("mobility_restriction", "Mobility restriction", "magma"),
)


def _sample_indices(count: int, maximum: int = 9000) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _scatter_map(ax, positions: np.ndarray, values: np.ndarray, title: str, cmap: str) -> None:
    indices = _sample_indices(len(positions))
    x = positions[indices, 0]
    z = positions[indices, 2]
    plotted = values[indices]
    lower, upper = np.percentile(plotted[np.isfinite(plotted)], (2.0, 98.0))
    if upper <= lower:
        upper = lower + 1.0e-9
    image = ax.scatter(x, z, c=plotted, s=2.2, cmap=cmap, vmin=lower, vmax=upper, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def render_motion_overview(
    path: Path,
    suite: MotionFitSuite,
    material_id: str,
    scenario_data: dict[tuple[str, str], dict],
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(25, 15), constrained_layout=True)
    for ax, pose in zip(axes.flat, suite.poses):
        record = scenario_data[(material_id, pose.pose_id)]
        receipt = record["receipt"]
        title = (
            f"{pose.display_name}\n"
            f"strain p99 {receipt['metrics']['strain_p99_ratio']:.3f} | "
            f"pressure p99 {receipt['metrics']['pressure_p99_kpa']:.2f} kPa | "
            f"{'PASS' if receipt['pose_pass'] else 'FAIL'}"
        )
        _scatter_map(ax, record["positions"], record["maps"]["strain_ratio"], title, "viridis")
    fig.suptitle(f"CP5 professional motion-fit overview — {material_id}", fontsize=18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _worst_scenario(suite: MotionFitSuite, material_id: str, scenario_data: dict) -> tuple[str, dict]:
    rows = []
    for pose in suite.poses:
        record = scenario_data[(material_id, pose.pose_id)]
        metrics = record["receipt"]["metrics"]
        score = metrics["strain_p99_ratio"] + metrics["pressure_p99_kpa"] / 20.0
        rows.append((score, pose.pose_id, record))
    _, pose_id, record = max(rows, key=lambda item: item[0])
    return pose_id, record


def _summary_panel(ax, pose_id: str, material_id: str, receipt: dict) -> None:
    ax.axis("off")
    metrics = receipt["metrics"]
    lines = [
        "GARMENT-CAD-PRO-R1A / CP5",
        "",
        f"material: {material_id}",
        f"worst pose: {pose_id}",
        f"pose pass: {receipt['pose_pass']}",
        "",
        f"strain p99: {metrics['strain_p99_ratio']:.5f}",
        f"stress p99: {metrics['stress_p99_n_m']:.2f} N/m",
        f"pressure p99: {metrics['pressure_p99_kpa']:.3f} kPa",
        f"clearance min: {metrics['clearance_min_m'] * 1000.0:.3f} mm",
        f"seam tension p95: {metrics['seam_tension_p95_n_m']:.2f} N/m",
        f"contact persistence p99: {metrics['contact_persistence_p99']:.3f}",
        f"mobility restriction p95: {metrics['mobility_restriction_p95']:.3f}",
        f"tail peak movement: {metrics['tail_peak_displacement_m'] * 1000.0:.3f} mm",
        "",
        "maps rendered from canonical scenario archive",
        "CP3 feature-complete topology: not triangulated",
        "product acceptance: false",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=11)
    ax.set_title("scope / metric summary")


def render_fit_maps(
    path: Path,
    suite: MotionFitSuite,
    material_id: str,
    scenario_data: dict[tuple[str, str], dict],
) -> str:
    pose_id, record = _worst_scenario(suite, material_id, scenario_data)
    fig, axes = plt.subplots(2, 4, figsize=(25, 17.5), constrained_layout=True)
    for ax, (map_id, title, cmap) in zip(axes.flat[:7], _MAP_SPECS):
        _scatter_map(ax, record["positions"], record["maps"][map_id], title, cmap)
    _summary_panel(axes.flat[7], pose_id, material_id, record["receipt"])
    fig.suptitle(f"CP5 canonical fit maps — {material_id} / {pose_id}", fontsize=18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return pose_id


def render_material_comparison(
    path: Path,
    suite: MotionFitSuite,
    material_ids: tuple[str, ...],
    scenario_data: dict[tuple[str, str], dict],
) -> str:
    reference = material_ids[1]
    pose_id, _ = _worst_scenario(suite, reference, scenario_data)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
    for column, material_id in enumerate(material_ids):
        record = scenario_data[(material_id, pose_id)]
        _scatter_map(axes[0, column], record["positions"], record["maps"]["stress_n_m"], f"{material_id}\nstress", "magma")
        _scatter_map(axes[1, column], record["positions"], record["maps"]["pressure_pa"], f"{material_id}\npressure", "inferno")
    fig.suptitle(f"CP5 calibrated-material comparison — {pose_id}", fontsize=18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return pose_id
