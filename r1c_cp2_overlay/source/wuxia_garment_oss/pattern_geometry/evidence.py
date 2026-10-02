"""Render CP2 evidence from exact 2D curve authorities and solver receipts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wuxia_garment_oss.pattern_geometry.curve import evaluate


_DISPOSITION_STYLE = {
    "SEWN": "-",
    "FINISHED": "--",
    "FOLD": ":",
    "OPEN": "-.",
    "CUT_INTERNAL": ":",
}


def _sample_segment(segment, count: int = 80) -> np.ndarray:
    return np.asarray([
        [evaluate(segment, index / count).x, evaluate(segment, index / count).y]
        for index in range(count + 1)
    ])


def _plot_component(ax, geometry, title: str) -> None:
    segment_map = geometry.segment_map()
    for boundary in geometry.boundaries:
        for segment_id in boundary.segment_ids:
            values = _sample_segment(segment_map[segment_id])
            ax.plot(values[:, 0], values[:, 1], linestyle=_DISPOSITION_STYLE[boundary.disposition], linewidth=1.5)
    for segment_id in geometry.internal_segment_ids:
        values = _sample_segment(segment_map[segment_id])
        ax.plot(values[:, 0], values[:, 1], linestyle=":", linewidth=0.8, alpha=0.7)
    for notch in geometry.notches:
        ax.scatter([notch.point.x], [notch.point.y], marker="|", s=40)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.15)


def _plot_component_group(ax, geometries, title: str) -> None:
    x_offset = 0.0
    all_x, all_y = [], []
    for geometry in geometries:
        segment_map = geometry.segment_map()
        points = [point for segment in geometry.segments for point in segment.points]
        lower = min(point.x for point in points)
        upper = max(point.x for point in points)
        shift = x_offset - lower
        for boundary in geometry.boundaries:
            for segment_id in boundary.segment_ids:
                values = _sample_segment(segment_map[segment_id])
                ax.plot(values[:, 0] + shift, values[:, 1], linestyle=_DISPOSITION_STYLE[boundary.disposition], linewidth=1.25)
                all_x.extend(values[:, 0] + shift)
                all_y.extend(values[:, 1])
        x_offset += upper - lower + 0.08
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    if all_x and all_y:
        ax.set_xlim(min(all_x) - 0.03, max(all_x) + 0.03)
        ax.set_ylim(min(all_y) - 0.03, max(all_y) + 0.03)


def _plot_interface_ratios(ax, interface_receipt: dict) -> None:
    rows = interface_receipt["interface_receipts"]
    names = [item["interface_id"] for item in rows]
    ratios = [item["directed_or_symmetric_ratio"] for item in rows]
    limits = [item["ratio_max"] for item in rows]
    positions = np.arange(len(rows))
    ax.bar(positions - 0.2, ratios, 0.4, label="actual")
    ax.bar(positions + 0.2, limits, 0.4, label="limit")
    ax.set_xticks(positions, names, rotation=65, ha="right", fontsize=7)
    ax.set_ylim(0.96, max(limits) + 0.03)
    ax.set_title("physical boundary / ease ratios")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)


def _plot_notches(ax, interface_receipt: dict) -> None:
    rows = []
    for interface in interface_receipt["interface_receipts"]:
        for notch in interface["notch_correspondence"]:
            rows.append((interface["interface_id"], notch["semantic_role"], notch["delta"]))
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "no notch correspondence", ha="center", va="center")
        return
    labels = [f"{interface}\n{role}" for interface, role, _ in rows]
    values = [delta for _, _, delta in rows]
    positions = np.arange(len(rows))
    ax.bar(positions, values)
    ax.axhline(0.025, linestyle="--", label="gate")
    ax.set_xticks(positions, labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("normalized arc delta")
    ax.set_title("front/back sleeve-cap notch correspondence")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)


def _plot_summary(ax, library: dict, assembly: dict, rejected: dict, receipt: dict) -> None:
    ax.axis("off")
    lines = [
        "GARMENT-CAD-PRO-R1C / CP2",
        "",
        f"geometry authorities: {library['authority_count']}",
        f"registered definitions: {receipt['component_definition_count']}",
        f"compiled instances: {assembly['component_instance_count']}",
        f"compiled seams: {assembly['seam_count']}",
        f"notch pairs: {assembly['notch_pair_count']}",
        f"interface solver: {receipt['interface_solver_pass']}",
        f"rejection fixture: {rejected['status']}",
        "",
        "geometry authority: EXACT_2D_COMPONENTS",
        "triangulation: NOT EXECUTED",
        "cloth simulation: NOT EXECUTED",
        "3D primitive fallback: FORBIDDEN",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax.set_title("terminal authority summary")


def render_cp2_evidence(path: Path, geometries: dict, library: dict, interface_receipt: dict, assembly: dict, rejected: dict, receipt: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), constrained_layout=True)
    _plot_component(axes[0, 0], geometries["bodice_front"], "bodice front exact 2D authority")
    _plot_component(axes[0, 1], geometries["bodice_back"], "bodice back exact 2D authority")
    _plot_component_group(axes[0, 2], [geometries["sleeve_left"], geometries["collar"], geometries["cuff_left"], geometries["gore_left"]], "sleeve / collar / cuff / gore library")
    _plot_interface_ratios(axes[1, 0], interface_receipt)
    _plot_notches(axes[1, 1], interface_receipt)
    _plot_summary(axes[1, 2], library, assembly, rejected, receipt)
    fig.suptitle("Exact pattern components, interface compatibility, and assembly compilation")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
