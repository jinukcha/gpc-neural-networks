"""Render construction authority evidence from canonical CP3 outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt


PANEL_OFFSETS = {
    "bodice_front": (-0.75, 0.0),
    "skirt_front": (-0.75, 0.0),
    "bodice_back": (0.75, 0.0),
    "skirt_back": (0.75, 0.0),
    "gusset_underarm": (0.0, -0.15),
}


def _offset_rows(rows, panel_id: str):
    dx, dy = PANEL_OFFSETS.get(panel_id, (0.0, 0.0))
    return [([float(item[0]) + dx for item in rows], [float(item[1]) + dy for item in rows])]


def _plot_line(ax, row: Mapping[str, object], key: str, style: str, width: float) -> None:
    for xs, ys in _offset_rows(row[key], str(row["panel_id"])):
        ax.plot(xs, ys, linestyle=style, linewidth=width)


def plot_stitch_cut(ax, package: Mapping[str, object]) -> None:
    for seam in package["seam_lines"]:
        for side in (seam["side_a"], seam["side_b"]):
            _plot_line(ax, side, "stitch_line", "-", 0.9)
            _plot_line(ax, side, "cut_line", "--", 0.7)
    for finish in package["edge_finishes"]:
        line = finish["line"]
        _plot_line(ax, line, "stitch_line", "-", 1.0)
        _plot_line(ax, line, "cut_line", "--", 0.8)
    for notch in package["notch_correspondence"]:
        seam = next(item for item in package["seam_specs"] if item["seam_id"] == notch["seam_id"])
        for key, side in (("side_a_position", seam["side_a"]), ("side_b_position", seam["side_b"])):
            dx, dy = PANEL_OFFSETS.get(side["panel_id"], (0.0, 0.0))
            point = notch[key]
            ax.scatter([point[0] + dx], [point[1] + dy], marker="x", s=16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Stitch lines, cut lines, seam allowance, notch correspondence")
    ax.set_xlabel("front pieces ← pattern x → back pieces (m)")
    ax.set_ylabel("pattern y (m)")
    ax.grid(True, alpha=0.2)


def plot_facing_closure(ax, package: Mapping[str, object]) -> None:
    for facing in package["facings"]:
        dx, dy = PANEL_OFFSETS.get(facing["owner_panel_id"], (0.0, 0.0))
        for key, style in (("outer_stitch_line", "-"), ("inner_cut_line", "--")):
            rows = facing[key]
            ax.plot([item[0] + dx for item in rows], [item[1] + dy for item in rows], linestyle=style)
    for closure in package["closures"]:
        dx, dy = PANEL_OFFSETS.get(closure["owner_panel_id"], (0.0, 0.0))
        rows = closure["internal_cut_line"]
        ax.plot([item[0] + dx for item in rows], [item[1] + dy for item in rows], linewidth=2.0)
        ax.scatter([rows[0][0] + dx, rows[-1][0] + dx], [rows[0][1] + dy, rows[-1][1] + dy], s=22)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Neck facings, centre-back closure, turn-of-cloth")
    ax.grid(True, alpha=0.2)


def _operation_levels(sequence) -> dict[str, int]:
    levels: dict[str, int] = {}
    for row in sequence:
        parents = row["depends_on"]
        levels[row["operation_id"]] = 0 if not parents else 1 + max(levels[parent] for parent in parents)
    return levels


def plot_assembly_dag(ax, package: Mapping[str, object]) -> None:
    sequence = package["assembly_plan"]["sequence"]
    levels = _operation_levels(sequence)
    grouped: dict[int, list[dict]] = {}
    for row in sequence:
        grouped.setdefault(levels[row["operation_id"]], []).append(row)
    positions = {}
    for level, rows in grouped.items():
        for index, row in enumerate(rows):
            y = index - (len(rows) - 1) * 0.5
            positions[row["operation_id"]] = (level, y)
            ax.scatter([level], [y], s=85)
            ax.text(level + 0.05, y, row["operation_id"].replace("OP_", ""), fontsize=7, va="center")
    for row in sequence:
        x1, y1 = positions[row["operation_id"]]
        for parent in row["depends_on"]:
            x0, y0 = positions[parent]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    ax.set_title("Cycle-free assembly DAG")
    ax.set_xlabel("dependency level")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.2)


def plot_summary(ax, package: Mapping[str, object], failures: Mapping[str, object]) -> None:
    gathered = [item for item in package["seam_specs"] if item["seam_type"] == "GATHERED_SEAM"]
    lines = [
        "GARMENT-CAD-PRO-R1A / CP3",
        "",
        f"SeamSpec/2: {len(package['seam_specs'])}",
        f"edge finishes: {len(package['edge_finishes'])}",
        f"notch pairs: {len(package['notch_correspondence'])}",
        f"closures: {len(package['closures'])}",
        f"facings: {len(package['facings'])}",
        f"lining/interfacing: {len(package['layer_pieces'])}",
        f"assembly operations: {package['assembly_plan']['operation_count']}",
        f"assembly cycle-free: {package['assembly_plan']['cycle_free']}",
        f"gather ratio: {gathered[0]['gather_ratio'] if gathered else 1.0}",
        f"failure probes atomic: {failures['all_atomic']} ({failures['probe_count']})",
        "",
        "triangulation: NOT EXECUTED",
        "Warp simulation: NOT EXECUTED",
        "mesh scaling: FORBIDDEN",
    ]
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=11)
    ax.set_title("Construction authority summary")


def render_construction_evidence(path: Path, package: Mapping[str, object], failures: Mapping[str, object]) -> None:
    figure = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    plot_stitch_cut(figure.add_subplot(grid[0, 0]), package)
    plot_facing_closure(figure.add_subplot(grid[0, 1]), package)
    plot_assembly_dag(figure.add_subplot(grid[1, 0]), package)
    plot_summary(figure.add_subplot(grid[1, 1]), package, failures)
    figure.suptitle("Professional construction graph execution evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
