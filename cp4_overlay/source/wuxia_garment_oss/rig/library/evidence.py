"""Render actual rigged GLB geometry and CP4 coverage diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import matplotlib.pyplot as plt
import numpy as np


_COMPONENT_DTYPE = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}
_COMPONENT_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _glb_chunks(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total != len(raw):
        raise ValueError(f"invalid GLB: {path}")
    offset = 12
    document = None
    binary = b""
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(chunk.rstrip(b" \t\r\n\0"))
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + length
    if document is None:
        raise ValueError("missing GLB JSON chunk")
    return document, binary


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(_COMPONENT_DTYPE[accessor["componentType"]]).newbyteorder("<")
    count = int(accessor["count"])
    width = _COMPONENT_COUNT[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", dtype.itemsize * width))
    if stride == dtype.itemsize * width:
        return np.frombuffer(binary, dtype=dtype, count=count * width, offset=offset).reshape(count, width).copy()
    result = np.empty((count, width), dtype=dtype)
    for row in range(count):
        result[row] = np.frombuffer(binary, dtype=dtype, count=width, offset=offset + row * stride)
    return result


def load_glb_positions(path: Path) -> np.ndarray:
    document, binary = _glb_chunks(path)
    arrays = []
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            position_index = primitive.get("attributes", {}).get("POSITION")
            if position_index is not None:
                arrays.append(_accessor(document, binary, int(position_index)).astype(np.float64))
    if not arrays:
        raise ValueError(f"no POSITION accessors: {path}")
    return np.concatenate(arrays, axis=0)


def _equal_axes(ax, first: int, second: int, arrays: tuple[np.ndarray, ...]) -> None:
    combined = np.concatenate([array[:, (first, second)] for array in arrays], axis=0)
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    centre = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.55, 0.1)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal")


def _plot_outfit(ax, tunic: np.ndarray, trousers: np.ndarray, view: str) -> None:
    if view == "front":
        first, second = 0, 2
    else:
        first, second = 1, 2
    step_tunic = max(1, len(tunic) // 9000)
    step_trousers = max(1, len(trousers) // 4000)
    ax.scatter(tunic[::step_tunic, first], tunic[::step_tunic, second], s=0.2, label="tunic")
    ax.scatter(trousers[::step_trousers, first], trousers[::step_trousers, second], s=0.3, label="trousers")
    _equal_axes(ax, first, second, (tunic, trousers))
    ax.set_title(f"compatible two-piece outfit — {view}")
    ax.legend(markerscale=8)
    ax.axis("off")


def _plot_occlusion(ax, mask_path: Path) -> None:
    with np.load(mask_path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
        triangles = np.asarray(data["triangles"], dtype=np.int32)
        hide = np.asarray(data["hide_triangle_mask"], dtype=np.bool_)
    centroids = positions[triangles].mean(axis=1)
    visible = ~hide
    ax.scatter(centroids[visible, 0], centroids[visible, 2], s=0.5, label="visible body")
    ax.scatter(centroids[hide, 0], centroids[hide, 2], s=0.5, label="hidden body")
    _equal_axes(ax, 0, 2, (positions,))
    ax.set_title("semantic body occlusion mask")
    ax.legend(markerscale=6)
    ax.axis("off")


def _plot_thickness(ax, plan: dict, registry: dict) -> None:
    regions = list(plan["region_thickness_m"])
    values = [plan["region_thickness_m"][item] * 1000.0 for item in regions]
    limits = [registry["regional_thickness_limits_m"][item] * 1000.0 for item in regions]
    x = np.arange(len(regions))
    ax.bar(x - 0.2, values, 0.4, label="outfit thickness")
    ax.bar(x + 0.2, limits, 0.4, label="regional limit")
    ax.set_xticks(x, regions, rotation=40, ha="right")
    ax.set_ylabel("mm")
    ax.set_title("layer-thickness admission")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)


def _plot_summary(ax, receipt: dict, transactions: dict) -> None:
    ax.axis("off")
    lines = [
        "GARMENT-CAD-PRO-R1B / CP4",
        "",
        f"registry entries: {receipt['registry_entry_count']}",
        f"compatible outfit: {receipt['compatible_outfit_pass']}",
        f"body occlusion: {receipt['body_occlusion_pass']}",
        f"Godot consumer: {receipt.get('godot_outfit_consumer_pass', False)}",
        "",
        "atomic transaction",
        f"  compatible commit: {transactions['compatible_commit_pass']}",
        f"  incompatible reject: {transactions['incompatible_rejection_pass']}",
        f"  source state preserved: {transactions['rejected_state_preserved']}",
        f"  unequip: {transactions['unequip_pass']}",
        "",
        "secondary motion: NOT EXECUTED",
        "rig-aware LOD: NOT EXECUTED",
    ]
    ax.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace", fontsize=11)
    ax.set_title("terminal evidence summary")


def render_cp4_evidence(root: Path, receipt: dict, transactions: dict) -> Path:
    build = root / "build/rig_cp4"
    tunic = load_glb_positions(root / "build/rig_cp3/products/tunic/tunic_rigged.glb")
    trousers = load_glb_positions(root / "build/rig_cp3/products/trousers/trousers_rigged.glb")
    registry = json.loads((build / "garment_library_registry.json").read_text(encoding="utf-8"))
    plan = json.loads((build / "outfits/reference_two_piece.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), constrained_layout=True)
    _plot_outfit(axes[0, 0], tunic, trousers, "front")
    _plot_outfit(axes[0, 1], tunic, trousers, "side")
    _plot_occlusion(axes[0, 2], build / "body_occlusion/body_hide_mask.npz")
    _plot_thickness(axes[1, 0], plan, registry)
    rejected = json.loads((build / "outfits/incompatible_duplicate_tunic.json").read_text(encoding="utf-8"))
    axes[1, 1].axis("off")
    axes[1, 1].text(0.03, 0.97, "incompatible outfit rejection\n\n" + "\n".join(rejected["rejection_reasons"]), va="top", family="monospace", fontsize=10)
    axes[1, 1].set_title("atomic rejection evidence")
    _plot_summary(axes[1, 2], receipt, transactions)
    output = build / "cp4_outfit_layering_evidence.png"
    fig.suptitle("Multi-garment registry, layering, body occlusion, and atomic outfit transaction")
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output
