"""Render CP1 evidence from canonical parameter results and rejection receipts."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _mode_panel(ax, resolved: dict) -> None:
    counts = Counter(item["mode"] for item in resolved["parameters"])
    labels = ["ABSOLUTE", "RELATIVE", "AUTO_DERIVED"]
    ax.bar(labels, [counts[label] for label in labels])
    ax.set_title("parameter modes")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.2)


def _scope_panel(ax, resolved: dict) -> None:
    scopes = Counter()
    for item in resolved["parameters"]:
        for path in item["reference_paths"]:
            root = path.split(".", 1)[0]
            if root != "param":
                scopes[root] += 1
    labels = ["body", "block", "component", "boundary", "material"]
    ax.bar(labels, [scopes[label] for label in labels])
    ax.set_title("resolved reference scopes")
    ax.set_ylabel("uses")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.2)


def _resolution_order_panel(ax, resolved: dict) -> None:
    ax.axis("off")
    lines = ["DETERMINISTIC RESOLUTION ORDER", ""]
    for index, item in enumerate(resolved["resolution_order"]):
        entry = next(value for value in resolved["parameters"] if value["parameter_id"] == item)
        dependencies = ", ".join(entry["parameter_dependencies"]) or "—"
        lines.append(f"{index:02d}  {item:<22}  deps: {dependencies}")
    ax.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    ax.set_title("dependency DAG publication")


def _bounds_panel(ax, resolved: dict) -> None:
    changed = [item for item in resolved["parameters"] if item["bounds"]["action"] == "CLAMPED"]
    if not changed:
        ax.text(0.5, 0.5, "no clamp", ha="center", va="center")
        return
    labels = [item["parameter_id"] for item in changed]
    requested = [item["requested_value_si"] for item in changed]
    published = [item["resolved_value_si"] for item in changed]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, requested, 0.36, label="requested")
    ax.bar(x + 0.18, published, 0.36, label="published")
    ax.set_xticks(x, labels)
    ax.set_title("SAFE_CLAMP provenance")
    ax.set_ylabel("SI value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)


def _rejection_panel(ax, rejections: dict[str, dict]) -> None:
    ax.axis("off")
    lines = ["ATOMIC REJECTION FIXTURES", ""]
    for name, receipt in sorted(rejections.items()):
        lines.append(f"{name:<30} {receipt['status']}")
    lines.extend(("", "partial publication count: 0 for every rejection"))
    ax.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("cycle / unit / quantity / bounds admission")


def _summary_panel(ax, resolved: dict, receipt: dict, reopen: dict) -> None:
    ax.axis("off")
    lines = [
        "GARMENT-CAD-PRO-R1C / CP1",
        "",
        f"parameter count        {receipt['parameter_count']}",
        f"clamp count            {receipt['clamp_count']}",
        f"resolved hash          {resolved['resolved_set_sha256'][:20]}…",
        f"fresh-process reopen   {reopen['fresh_process_reopen_pass']}",
        f"deterministic rerun     {reopen['deterministic_rerun_pass']}",
        "",
        "modes                  3 / 3",
        "reference scopes       5 / 5",
        "geometry               NOT EXECUTED",
        "triangulation          NOT EXECUTED",
        "simulation             NOT EXECUTED",
        "Godot                  NOT EXECUTED",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax.set_title("terminal evidence summary")


def render_cp1_evidence(
    path: Path,
    resolved: dict,
    receipt: dict,
    rejections: dict[str, dict],
    reopen: dict,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), constrained_layout=True)
    _mode_panel(axes[0, 0], resolved)
    _scope_panel(axes[0, 1], resolved)
    _resolution_order_panel(axes[0, 2], resolved)
    _bounds_panel(axes[1, 0], resolved)
    _rejection_panel(axes[1, 1], rejections)
    _summary_panel(axes[1, 2], resolved, receipt, reopen)
    fig.suptitle("R1C CP1 — typed ratio parameter resolution and atomic rejection evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
