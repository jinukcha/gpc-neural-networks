"""Render actual rigged GLB geometry, dominant joints, and equip transactions."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .glb_skin import BONES, _accessor_array, _all_positions, _axis_frame, read_glb


def _project(values: np.ndarray, horizontal: int, vertical: int) -> np.ndarray:
    return values[:, [horizontal, vertical]]


def _triangles(glb, primitive: dict, projection: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    positions = _accessor_array(glb, primitive["attributes"]["POSITION"]).astype(np.float64)
    indices = _accessor_array(glb, primitive["indices"]).reshape(-1).astype(np.int64)
    triangles = indices.reshape(-1, 3)
    projected = _project(positions, *projection)
    weights = _accessor_array(glb, primitive["attributes"]["WEIGHTS_0"]).astype(np.float64)
    joints = _accessor_array(glb, primitive["attributes"]["JOINTS_0"]).astype(np.int64)
    dominant = joints[np.arange(len(joints)), np.argmax(weights, axis=1)]
    tri_dom = np.asarray([np.bincount(dominant[row], minlength=len(BONES)).argmax() for row in triangles])
    return projected[triangles], tri_dom


def _draw_garment(ax, path: Path, view: str, title: str) -> None:
    glb = read_glb(path)
    all_positions = _all_positions(glb)
    lateral, vertical, depth, _, _ = _axis_frame(all_positions)
    projection = (lateral, vertical) if view == "front" else (depth, vertical)
    collections = []
    for mesh in glb.document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            polys, dominant = _triangles(glb, primitive, projection)
            collections.append((polys, dominant))
    cmap = plt.get_cmap("tab20", len(BONES))
    for polys, dominant in collections:
        colors = cmap(dominant % 20)
        colors[:, :3] = 0.70 * colors[:, :3] + 0.30
        collection = PolyCollection(polys, facecolors=colors, edgecolors="none", linewidths=0.0)
        ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def _draw_skeleton(ax, skeleton_receipt: dict) -> None:
    names = [item[0] for item in BONES]
    parents = {item[0]: item[1] for item in BONES}
    positions = skeleton_receipt["global_joint_positions"]
    for name in names:
        point = positions[name]
        ax.scatter(point[0], point[1], s=20)
        ax.text(point[0], point[1], name, fontsize=6)
        parent = parents[name]
        if parent is not None:
            other = positions[parent]
            ax.plot((other[0], point[0]), (other[1], point[1]), linewidth=1.0)
    ax.set_aspect("equal")
    ax.set_title("Canonical 23-bone rig")
    ax.axis("off")


def _draw_transactions(ax, receipt: dict) -> None:
    ax.axis("off")
    rows = receipt["transactions"]
    columns = ("operation", "slot", "target", "result", "state_count")
    table_rows = [[str(row.get(key, "")) for key in columns] for row in rows]
    table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax.set_title("Atomic equip / unequip / character swap")


def _draw_summary(ax, receipt: dict) -> None:
    ax.axis("off")
    lines = [
        "GARMENT-CAD-PRO-R1B / CP3",
        "",
        f"terminal: {receipt['terminal_decision']}",
        f"tunic bones: {receipt['tunic']['bone_count']}",
        f"tunic vertices: {receipt['tunic']['vertex_count']:,}",
        f"tunic primitives: {receipt['tunic']['primitive_count']}",
        f"trousers bones: {receipt['trousers']['bone_count']}",
        f"trousers vertices: {receipt['trousers']['vertex_count']:,}",
        f"trousers primitives: {receipt['trousers']['primitive_count']}",
        "",
        f"GLB fresh reopen: {receipt['glb_fresh_reopen_pass']}",
        f"Godot 4.7.2 consumer: {receipt['godot_consumer_pass']}",
        f"atomic equip: {receipt['atomic_equip_pass']}",
        f"character swap: {receipt['character_swap_pass']}",
        f"incompatible swap preserved: {receipt['failed_swap_preserved']}",
        "",
        "secondary motion: NOT EXECUTED",
        "LOD: NOT EXECUTED",
        "outfit composition: NOT EXECUTED",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("Product receipt")


def render_evidence(
    path: Path,
    tunic_glb: Path,
    trousers_glb: Path,
    skeleton_receipt: dict,
    equip_receipt: dict,
    cp3_receipt: dict,
) -> None:
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 4)
    _draw_garment(fig.add_subplot(grid[0:2, 0]), tunic_glb, "front", "Rigged tunic — front")
    _draw_garment(fig.add_subplot(grid[0:2, 1]), tunic_glb, "side", "Rigged tunic — side")
    _draw_garment(fig.add_subplot(grid[0, 2]), trousers_glb, "front", "Rigged trousers — front")
    _draw_garment(fig.add_subplot(grid[1, 2]), trousers_glb, "side", "Rigged trousers — side")
    _draw_skeleton(fig.add_subplot(grid[0:2, 3]), skeleton_receipt)
    _draw_transactions(fig.add_subplot(grid[2, 0:3]), equip_receipt)
    _draw_summary(fig.add_subplot(grid[2, 3]), cp3_receipt)
    fig.suptitle("R1B CP3 — actual skinned GLB and Godot equip evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
