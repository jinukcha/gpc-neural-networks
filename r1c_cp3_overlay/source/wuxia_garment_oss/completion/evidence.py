"""Render actual CP3 source-pattern defects, repairs, and transaction results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geometry import sample_segment


_LAYOUT = {
    "bodice_front": (-0.48, 0.0),
    "sleeve_right": (0.36, 0.0),
    "cuff_left": (0.28, -0.23),
    "collar": (-0.42, -0.18),
}


def _plot_snapshot(ax, snapshot: dict, title: str) -> None:
    geometry = snapshot["geometry_by_instance"]
    for instance_id, offset in _LAYOUT.items():
        component = geometry.get(instance_id)
        if component is None:
            continue
        internal = set(component.get("internal_segment_ids", []))
        for segment in component["segments"]:
            points = np.asarray(sample_segment(segment, 96), dtype=np.float64)
            points[:, 0] += offset[0]
            points[:, 1] += offset[1]
            alpha = 0.35 if segment["segment_id"] in internal else 1.0
            ax.plot(points[:, 0], points[:, 1], linewidth=1.0, alpha=alpha)
        for notch in component.get("notches", []):
            point = notch["point_m"]
            ax.scatter(point[0] + offset[0], point[1] + offset[1], s=8)
        ax.text(offset[0], offset[1] - 0.035, instance_id, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def _plot_issue_counts(ax, report: dict) -> None:
    labels = ["SAFE_AUTO", "GUIDED", "HOLD"]
    values = [report["disposition_counts"][name] for name in labels]
    ax.bar(labels, values)
    ax.set_ylabel("diagnosed issues")
    ax.set_title("completion diagnosis disposition")
    ax.grid(True, axis="y", alpha=0.2)


def _plot_operations(ax, plan: dict) -> None:
    operations = plan["operations"]
    labels = [item["operation_kind"].replace("RESTORE_", "") for item in operations]
    values = [
        0.0 if item.get("maximum_delta") is None else float(item["maximum_delta"])
        for item in operations
    ]
    positions = np.arange(len(operations))
    ax.barh(positions, values)
    ax.set_yticks(positions, labels, fontsize=7)
    ax.set_xlabel("maximum diagnosed delta")
    ax.set_title("bounded repair operations")
    ax.grid(True, axis="x", alpha=0.2)


def _plot_transactions(ax, receipts: dict[str, dict]) -> None:
    labels = list(receipts)
    committed = [item["committed_operation_count"] for item in receipts.values()]
    working = [item["working_operation_count_before_failure"] for item in receipts.values()]
    positions = np.arange(len(labels))
    ax.bar(positions - 0.18, committed, 0.36, label="committed")
    ax.bar(positions + 0.18, working, 0.36, label="private working")
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel("operations")
    ax.set_title("atomic transaction outcomes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)


def _plot_summary(ax, receipt: dict, previews: dict[str, dict]) -> None:
    ax.axis("off")
    lines = [
        "GARMENT-CAD-PRO-R1C / CP3",
        "",
        f"terminal: {receipt.get('terminal_decision', 'PENDING')}",
        f"SAFE_AUTO committed: {receipt['safe_auto_commit_pass']}",
        f"GUIDED preserved: {receipt['guided_wait_pass']}",
        f"HOLD preserved: {receipt['topology_hold_pass']}",
        f"rollback atomic: {receipt['rollback_pass']}",
        "",
        f"safe operations: {receipt['safe_operation_count']}",
        f"safe issues: {receipt['safe_issue_count']}",
        f"guided issues: {receipt['guided_issue_count']}",
        f"hold issues: {receipt['hold_issue_count']}",
        "",
        f"safe preview matches authority: {previews['safe']['projected_matches_canonical']}",
        f"guided approval required: {previews['guided']['requires_user_approval']}",
        f"topology blocked: {previews['hold']['blocked_by_topology_change']}",
        "",
        "triangulation: NOT EXECUTED",
        "simulation: NOT EXECUTED",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax.set_title("terminal completion authority")


def render_cp3_evidence(
    output: Path,
    safe_candidate: dict,
    repaired: dict,
    report: dict,
    plan: dict,
    transactions: dict[str, dict],
    previews: dict[str, dict],
    receipt: dict,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), constrained_layout=True)
    _plot_snapshot(axes[0, 0], safe_candidate, "before — incomplete / drifted source pattern")
    _plot_snapshot(axes[0, 1], repaired, "after — atomically committed source repair")
    _plot_issue_counts(axes[0, 2], report)
    _plot_operations(axes[1, 0], plan)
    _plot_transactions(axes[1, 1], transactions)
    _plot_summary(axes[1, 2], receipt, previews)
    fig.suptitle("R1C CP3 — completion diagnosis, bounded repair, and atomic transaction")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
