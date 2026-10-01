"""Render CP5-R1 before/after and requalified fit-map evidence."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAILED_POSES = (
    "ARMS_OVERHEAD",
    "CROSS_BODY_REACH",
    "FORWARD_BEND",
    "SEATED",
    "SQUAT",
    "WALK_STRIDE",
)

MAP_SPECS = (
    ("stress_n_m", "Stress (N/m)", "magma"),
    ("strain_ratio", "Strain ratio", "viridis"),
    ("pressure_pa", "Pressure (Pa)", "inferno"),
    ("clearance_m", "Clearance (m)", "coolwarm"),
    ("seam_tension_n_m", "Seam tension (N/m)", "plasma"),
    ("contact_persistence", "Contact persistence", "cividis"),
    ("mobility_restriction", "Mobility restriction", "magma"),
)


def _indices(count: int, maximum: int = 8500) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _scatter(ax, positions: np.ndarray, values: np.ndarray, title: str, cmap: str) -> None:
    indices = _indices(len(positions))
    selected = values[indices]
    finite = selected[np.isfinite(selected)]
    low, high = np.percentile(finite, (2.0, 98.0)) if finite.size else (0.0, 1.0)
    if high <= low:
        high = low + 1.0e-9
    image = ax.scatter(
        positions[indices, 0],
        positions[indices, 2],
        c=selected,
        s=2.0,
        cmap=cmap,
        vmin=low,
        vmax=high,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def render_overview(path: Path, suite, material_id: str, merged: dict) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(25, 15), constrained_layout=True)
    for ax, pose in zip(axes.flat, suite.poses):
        record = merged[(material_id, pose.pose_id)]
        receipt = record["receipt"]
        metrics = receipt["metrics"]
        title = (
            f"{pose.display_name}\n"
            f"strain {metrics['strain_p99_ratio']:.3f} | "
            f"pressure {metrics['pressure_p99_kpa']:.2f} kPa | "
            f"{'PASS' if receipt['pose_pass'] else 'FAIL'}"
        )
        _scatter(ax, record["positions"], record["maps"]["strain_ratio"], title, "viridis")
    fig.suptitle(f"CP5-R1 motion-fit requalification — {material_id}", fontsize=18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_before_after(path: Path, material_id: str, original: dict, repaired: dict) -> None:
    fig, axes = plt.subplots(6, 2, figsize=(16, 34), constrained_layout=True)
    for row, pose_id in enumerate(FAILED_POSES):
        before = original[(material_id, pose_id)]
        after = repaired[(material_id, pose_id)]
        before_metric = before["receipt"]["metrics"]
        after_metric = after["receipt"]["metrics"]
        before_title = (
            f"{pose_id} — CP5\n"
            f"strain {before_metric['strain_p99_ratio']:.3f}, "
            f"tail {before_metric['tail_peak_displacement_m'] * 1000.0:.1f} mm"
        )
        after_title = (
            f"{pose_id} — CP5-R1\n"
            f"strain {after_metric['strain_p99_ratio']:.3f}, "
            f"tail {after_metric['tail_peak_displacement_m'] * 1000.0:.1f} mm | "
            f"{'PASS' if after['receipt']['pose_pass'] else 'FAIL'}"
        )
        _scatter(axes[row, 0], before["positions"], before["maps"]["strain_ratio"], before_title, "viridis")
        _scatter(axes[row, 1], after["positions"], after["maps"]["strain_ratio"], after_title, "viridis")
    fig.suptitle(f"CP5 → CP5-R1 failed-pose comparison — {material_id}", fontsize=18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _worst_repaired(material_id: str, repaired: dict) -> tuple[str, dict]:
    candidates = []
    for pose_id in FAILED_POSES:
        record = repaired[(material_id, pose_id)]
        metrics = record["receipt"]["metrics"]
        score = metrics["strain_p99_ratio"] + metrics["pressure_p99_kpa"] / 20.0
        candidates.append((score, pose_id, record))
    _, pose_id, record = max(candidates, key=lambda item: item[0])
    return pose_id, record


def render_maps(path: Path, material_id: str, repaired: dict) -> str:
    pose_id, record = _worst_repaired(material_id, repaired)
    fig, axes = plt.subplots(2, 4, figsize=(25, 17.5), constrained_layout=True)
    for ax, (map_id, title, cmap) in zip(axes.flat[:7], MAP_SPECS):
        _scatter(ax, record["positions"], record["maps"][map_id], title, cmap)
    receipt = record["receipt"]
    metrics = receipt["metrics"]
    axes.flat[7].axis("off")
    lines = [
        "GARMENT-CAD-PRO-R1A / CP5-R1",
        "",
        f"material: {material_id}",
        f"worst repaired pose: {pose_id}",
        f"pose pass: {receipt['pose_pass']}",
        "",
        f"strain p99: {metrics['strain_p99_ratio']:.5f}",
        f"pressure p99: {metrics['pressure_p99_kpa']:.3f} kPa",
        f"clearance min: {metrics['clearance_min_m'] * 1000.0:.3f} mm",
        f"contact p99: {metrics['contact_persistence_p99']:.3f}",
        f"mobility p95: {metrics['mobility_restriction_p95']:.3f}",
        f"tail movement: {metrics['tail_peak_displacement_m'] * 1000.0:.3f} mm",
        "",
        "thresholds unchanged from CP5",
        "passed CP5 scenarios not re-executed",
    ]
    axes.flat[7].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=11)
    axes.flat[7].set_title("repair summary")
    fig.suptitle(f"CP5-R1 canonical fit maps — {material_id} / {pose_id}", fontsize=18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return pose_id
