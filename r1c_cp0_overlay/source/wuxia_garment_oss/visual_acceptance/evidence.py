"""Render CP0 contract evidence from canonical JSON products."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _component_graph(ax, registry: dict, recipe: dict, interfaces: list[dict]) -> None:
    instances = recipe["component_instances"]
    positions = {
        "bodice_front": (0.0, 1.0),
        "bodice_back": (1.0, 1.0),
        "sleeve_left": (-0.8, 0.0),
        "sleeve_right": (1.8, 0.0),
    }
    component_by_id = {item["component_id"]: item for item in registry["components"]}
    for item in instances:
        x, y = positions.get(item["instance_id"], (0.0, 0.0))
        component = component_by_id[item["component_id"]]
        ax.scatter([x], [y], s=1100)
        ax.text(x, y, f"{item['instance_id']}\n{component['category']}", ha="center", va="center", fontsize=8)
    for spec in interfaces:
        left = spec["endpoint_a"]["component_instance_id"]
        right = spec["endpoint_b"]["component_instance_id"]
        if left not in positions or right not in positions:
            continue
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        ax.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=0.7)
    ax.set_title("modular component / interface graph")
    ax.set_xlim(-1.4, 2.4)
    ax.set_ylim(-0.6, 1.6)
    ax.axis("off")


def _boundary_summary(ax, registry: dict) -> None:
    counts = Counter(
        boundary["disposition"]
        for component in registry["components"]
        for boundary in component["boundaries"]
    )
    labels = sorted(counts)
    ax.bar(labels, [counts[item] for item in labels])
    ax.set_title("boundary authority by disposition")
    ax.set_ylabel("boundary count")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.2)


def _gate_matrix(ax, profile: dict) -> None:
    severities = {"INFORMATIONAL": 0, "BOUNDED": 1, "ZERO_TOLERANCE": 2}
    dispositions = {"SAFE_AUTO": 0, "GUIDED": 1, "HOLD": 2}
    gates = profile["gates"]
    matrix = np.asarray(
        [[severities[item["severity"]], dispositions[item["repair_disposition"]]] for item in gates],
        dtype=np.float64,
    )
    ax.imshow(matrix, aspect="auto")
    ax.set_yticks(np.arange(len(gates)), [item["gate_id"] for item in gates], fontsize=7)
    ax.set_xticks((0, 1), ("severity", "repair disposition"))
    ax.set_title("visual gates: severity and repair boundary")


def _view_summary(ax, profile: dict) -> None:
    counts = Counter(item["category"] for item in profile["required_views"])
    labels = sorted(counts)
    ax.barh(labels, [counts[item] for item in labels])
    ax.set_xlabel("required view count")
    ax.set_title("evidence authority by view category")
    ax.grid(True, axis="x", alpha=0.2)


def _receipt_summary(ax, clean: dict, rejected: dict, assembly: dict, broken: dict) -> None:
    ax.axis("off")
    lines = [
        "GARMENT-CAD-PRO-R1C / CP0",
        "",
        "assembly authority",
        f"  accepted recipe: {assembly['accepted']}",
        f"  component instances: {assembly['component_instance_count']}",
        f"  interfaces: {assembly['interface_count']}",
        f"  all sewn boundaries bound: {not assembly['unbound_sewn_boundaries']}",
        f"  incomplete fixture rejected: {not broken['accepted']}",
        "",
        "visual authority",
        f"  clean visual review: {clean['visual_review']}",
        f"  clean product acceptance: {clean['product_acceptance']}",
        f"  rejected visual review: {rejected['visual_review']}",
        f"  rejected product acceptance: {rejected['product_acceptance']}",
        f"  failed gates: {len(rejected['failed_gate_ids'])}",
        "",
        "geometry executed: false",
        "simulation executed: false",
        "Godot executed: false",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax.set_title("terminal contract evidence")


def render_cp0_evidence(
    path: Path,
    registry: dict,
    recipe: dict,
    interfaces: list[dict],
    profile: dict,
    clean_receipt: dict,
    rejected_receipt: dict,
    assembly_receipt: dict,
    broken_receipt: dict,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), constrained_layout=True)
    _component_graph(axes[0, 0], registry, recipe, interfaces)
    _boundary_summary(axes[0, 1], registry)
    _gate_matrix(axes[0, 2], profile)
    _view_summary(axes[1, 0], profile)
    failed = rejected_receipt["failed_gate_ids"]
    axes[1, 1].barh(failed, np.arange(1, len(failed) + 1))
    axes[1, 1].set_title("rejected fixture — blocking visual gates")
    axes[1, 1].set_xlabel("diagnostic order")
    _receipt_summary(axes[1, 2], clean_receipt, rejected_receipt, assembly_receipt, broken_receipt)
    fig.suptitle("R1C CP0 — modular pattern contract and visual acceptance authority")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
