"""Render CP5 evidence from generated GLBs and their published contracts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wuxia_garment_oss.rig.game_product.glb_skin import BONE_INDEX, _accessor_array, read_glb


CP5_MESH_SUFFIX = "_CP5_COMPONENTS"


def _cp5_arrays(path: Path) -> dict:
    glb = read_glb(path)
    whole_positions = []
    for mesh in glb.document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            accessor = primitive.get("attributes", {}).get("POSITION")
            if accessor is not None:
                whole_positions.append(_accessor_array(glb, accessor).astype(np.float64))
    cp5_mesh = next(
        item for item in glb.document["meshes"]
        if item.get("name", "").endswith(CP5_MESH_SUFFIX)
    )
    positions, joints, weights, corrective = [], [], [], []
    for primitive in cp5_mesh["primitives"]:
        positions.append(_accessor_array(glb, primitive["attributes"]["POSITION"]).astype(np.float64))
        joints.append(_accessor_array(glb, primitive["attributes"]["JOINTS_0"]).astype(np.int32))
        weights.append(_accessor_array(glb, primitive["attributes"]["WEIGHTS_0"]).astype(np.float64))
        corrective.append(_accessor_array(glb, primitive["targets"][2]["POSITION"]).astype(np.float64))
    return {
        "all_positions": np.concatenate(whole_positions),
        "positions": np.concatenate(positions),
        "joints": np.concatenate(joints),
        "weights": np.concatenate(weights),
        "elbow_corrective": np.concatenate(corrective),
    }


def _equal(ax, first: int, second: int, positions: np.ndarray) -> None:
    values = positions[:, (first, second)]
    lower, upper = values.min(axis=0), values.max(axis=0)
    centre = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.55, 0.1)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal")
    ax.axis("off")


def _plot_geometry(ax, arrays: dict, view: str, title: str) -> None:
    first, second = (0, 1) if view == "front" else (2, 1)
    whole = arrays["all_positions"]
    owned = arrays["positions"]
    whole_step = max(1, len(whole) // 15000)
    owned_step = max(1, len(owned) // 6000)
    ax.scatter(whole[::whole_step, first], whole[::whole_step, second], s=0.18, alpha=0.22, label="immutable CP3 base")
    ax.scatter(owned[::owned_step, first], owned[::owned_step, second], s=0.7, label="CP5 owned components")
    _equal(ax, first, second, whole)
    ax.set_title(f"{title} — {view}")
    ax.legend(markerscale=6, loc="lower right")


def _upper_forearm_weight(arrays: dict) -> np.ndarray:
    target_indices = {
        BONE_INDEX["L_UPPER_ARM"], BONE_INDEX["R_UPPER_ARM"],
        BONE_INDEX["L_FOREARM"], BONE_INDEX["R_FOREARM"],
    }
    values = np.zeros(len(arrays["weights"]), dtype=np.float64)
    for column in range(4):
        mask = np.isin(arrays["joints"][:, column], list(target_indices))
        values += arrays["weights"][:, column] * mask
    return values


def _plot_weight_field(ax, arrays: dict) -> None:
    positions = arrays["positions"]
    weight = _upper_forearm_weight(arrays)
    plot = ax.scatter(positions[:, 0], positions[:, 1], c=weight, s=1.0)
    _equal(ax, 0, 1, arrays["all_positions"])
    ax.set_title("upper-arm + forearm transferred weight")
    plt.colorbar(plot, ax=ax, fraction=0.046)


def _plot_corrective(ax, arrays: dict) -> None:
    positions = arrays["positions"]
    magnitude = np.linalg.norm(arrays["elbow_corrective"], axis=1)
    active = magnitude > 1.0e-8
    ax.scatter(positions[~active, 0], positions[~active, 1], s=0.2, alpha=0.15)
    plot = ax.scatter(positions[active, 0], positions[active, 1], c=magnitude[active] * 1000.0, s=2.0)
    _equal(ax, 0, 1, arrays["all_positions"])
    ax.set_title("ELBOW_BEND sparse corrective — mm")
    if np.any(active):
        plt.colorbar(plot, ax=ax, fraction=0.046)


def _plot_measurements(ax, measurement: dict, products: tuple[dict, dict]) -> None:
    ax.axis("off")
    values = measurement["measurements"]
    lines = [
        "DIRECT ARM MEASUREMENTS",
        "",
        f"shoulder→elbow  {values['shoulder_to_elbow_m']:.4f} m",
        f"elbow→wrist     {values['elbow_to_wrist_m']:.4f} m",
        f"sleeve length   {values['sleeve_length_m']:.4f} m",
        f"upper-arm circ. {values['upper_arm_circumference_m']:.4f} m",
        f"elbow circ.     {values['elbow_circumference_m']:.4f} m",
        f"wrist circ.     {values['wrist_circumference_m']:.4f} m",
        f"armscye circ.   {values['armscye_circumference_m']:.4f} m",
        f"cap height      {values['sleeve_cap_height_m']:.4f} m",
        f"cap ease        {values['cap_ease_ratio']:.3f}",
        "",
        f"sleeved tunic CP5 vertices  {products[0]['vertex_count']}",
        f"straight robe CP5 vertices  {products[1]['vertex_count']}",
        "secondary motion: NOT EXECUTED",
        "rig-aware LOD: NOT EXECUTED",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax.set_title("measurement and terminal scope")


def render_cp5_evidence(root: Path, measurement: dict, products: tuple[dict, dict]) -> Path:
    tunic = _cp5_arrays(root / products[0]["path"])
    robe = _cp5_arrays(root / products[1]["path"])
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), constrained_layout=True)
    _plot_geometry(axes[0, 0], tunic, "front", "sleeved tunic")
    _plot_geometry(axes[0, 1], tunic, "side", "sleeved tunic")
    _plot_geometry(axes[0, 2], robe, "front", "straight-sleeve robe")
    _plot_weight_field(axes[1, 0], tunic)
    _plot_corrective(axes[1, 1], tunic)
    _plot_measurements(axes[1, 2], measurement, products)
    output = root / "build/rig_cp5/cp5_sleeve_generalization_evidence.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("GARMENT-CAD-PRO-R1B CP5 — complete products, sleeve weights, and correctives")
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output
