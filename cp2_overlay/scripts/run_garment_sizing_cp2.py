#!/usr/bin/env python3
"""Publish CP2 tunic POM and named-landmark packages without meshing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wuxia_garment_oss.garments.sleeveless_tunic.pattern.resolver import (
    resolve_pattern_parameters,
)
from wuxia_garment_oss.garments.sleeveless_tunic.reference import (
    body_fixtures,
    reference_size_table,
)
from wuxia_garment_oss.sizing.contracts import contract_schemas
from wuxia_garment_oss.sizing.instance.model import canonical_sha256
from wuxia_garment_oss.sizing.selection.resolver import SelectionRequest, resolve_selection


STANDARD_SIZES = ("S", "M", "L")
CUSTOM_CASES = (
    ("AUTO_REFERENCE", "REFERENCE"),
    ("AUTO_MILD_CUSTOM", "MILD_CUSTOM"),
    ("AUTO_BROAD_SHOULDER", "BROAD_SHOULDER"),
    ("AUTO_FULL_CHEST", "FULL_CHEST"),
    ("AUTO_FULL_ABDOMEN", "FULL_ABDOMEN"),
    ("AUTO_TALL", "TALL"),
    ("AUTO_SHORT", "SHORT"),
)
BLOCKED_CASES = ("CUSTOM_FORCED_M", "AUTO_TOPOLOGY", "AUTO_HOLD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_contracts(root: Path) -> None:
    for name, schema in contract_schemas().items():
        write_json(root / "contracts/sizing" / name, schema)


def load_receipt(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = str(payload["receipt_sha256"])
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256")
    if canonical_sha256(unsigned) != stored:
        raise ValueError(f"receipt hash mismatch: {path}")
    return payload


def standard_receipt(root: Path, size_id: str, table) -> dict:
    if size_id == "M":
        source = root / "build/tunic_pilot/sizing_cp1/selection_receipts/STANDARD_M.json"
        return load_receipt(source)
    request = SelectionRequest(f"STANDARD_{size_id}", "STANDARD_SIZE", size_id)
    return resolve_selection(request, table, None).to_dict()


def execute_packages(root: Path) -> tuple[dict[str, dict], list[dict]]:
    table = reference_size_table()
    bodies = body_fixtures()
    build = root / "build/tunic_pilot/sizing_cp2"
    packages: dict[str, dict] = {}
    blocked: list[dict] = []
    for size_id in STANDARD_SIZES:
        name = f"STANDARD_{size_id}"
        receipt = standard_receipt(root, size_id, table)
        write_json(build / "selection_inputs" / f"{name}.json", receipt)
        packages[name] = resolve_pattern_parameters(receipt, table, None)
    cp1_dir = root / "build/tunic_pilot/sizing_cp1/selection_receipts"
    for name, body_name in CUSTOM_CASES:
        receipt = load_receipt(cp1_dir / f"{name}.json")
        packages[name] = resolve_pattern_parameters(receipt, table, bodies[body_name])
    for name in BLOCKED_CASES:
        receipt = load_receipt(cp1_dir / f"{name}.json")
        blocked.append({
            "request_id": receipt["request_id"],
            "admission": receipt["admission"],
            "selection_receipt_sha256": receipt["receipt_sha256"],
            "pattern_parameter_package": "NOT_PUBLISHED",
            "warnings": receipt.get("warnings", []),
        })
    for name, package in packages.items():
        write_json(build / "pattern_parameter_packages" / f"{name}.json", package)
    write_json(build / "blocked_admissions.json", {"blocked": blocked})
    return packages, blocked


def _panel(package: dict, panel_id: str) -> dict:
    return next(panel for panel in package["panels"] if panel["panel_id"] == panel_id)


def _outline(panel: dict) -> tuple[np.ndarray, np.ndarray]:
    points = [panel["landmarks"][name] for name in panel["outline_order"]]
    values = np.asarray(points, dtype=np.float64)
    return values[:, 0], values[:, 1]


def _draw_front_pattern(ax, package: dict, label: str, offset_x: float = 0.0) -> None:
    for panel_id in ("bodice_front", "skirt_front"):
        x, y = _outline(_panel(package, panel_id))
        ax.plot(x + offset_x, y, linewidth=1.5, label=label if panel_id == "bodice_front" else None)


def plot_standard(ax, packages: dict[str, dict]) -> None:
    for index, size_id in enumerate(STANDARD_SIZES):
        _draw_front_pattern(ax, packages[f"STANDARD_{size_id}"], size_id, index * 0.78)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Standard S / M / L — front 2D parameter outlines")
    ax.set_xlabel("pattern x (m), separated for review")
    ax.set_ylabel("pattern y (m)")
    ax.legend()
    ax.grid(True, alpha=0.25)


def plot_custom(ax, packages: dict[str, dict]) -> None:
    names = (
        "AUTO_REFERENCE", "AUTO_BROAD_SHOULDER", "AUTO_FULL_CHEST",
        "AUTO_FULL_ABDOMEN", "AUTO_TALL", "AUTO_SHORT",
    )
    for name in names:
        _draw_front_pattern(ax, packages[name], name.replace("AUTO_", ""))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Detailed-measurement alterations — front outline overlay")
    ax.set_xlabel("pattern x (m)")
    ax.set_ylabel("pattern y (m)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)


def plot_poms(ax, packages: dict[str, dict]) -> None:
    keys = (
        "finished_chest_circumference_m",
        "finished_waist_circumference_m",
        "shoulder_half_m",
        "skirt_length_m",
    )
    labels = ("chest", "waist", "shoulder half", "skirt length")
    values = np.asarray([
        [packages[f"STANDARD_{size}"]["poms"][key] for key in keys]
        for size in STANDARD_SIZES
    ])
    x = np.arange(len(keys))
    width = 0.24
    for index, size in enumerate(STANDARD_SIZES):
        ax.bar(x + (index - 1) * width, values[index], width, label=size)
    ax.set_xticks(x, labels)
    ax.set_ylabel("metres")
    ax.set_title("Resolved POM progression")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)


def plot_summary(ax, packages: dict[str, dict], blocked: list[dict]) -> None:
    ax.axis("off")
    lines = [
        "fixture | size | shape | height | admission | package",
        "-" * 86,
    ]
    for name in sorted(packages):
        selection = packages[name]["selection"]
        lines.append(
            f"{name:<23} {selection['selected_size_id']:<4} "
            f"{selection['shape_block']:<16} {selection['height_block']:<7} "
            f"{selection['admission']:<18} {packages[name]['package_id'][:10]}"
        )
    lines.append("")
    for item in blocked:
        lines.append(f"BLOCKED {item['request_id']}: {item['admission']}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("Receipt-bound CP2 publication summary")


def render_evidence(path: Path, packages: dict[str, dict], blocked: list[dict]) -> None:
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    plot_standard(fig.add_subplot(grid[0, 0]), packages)
    plot_custom(fig.add_subplot(grid[0, 1]), packages)
    plot_poms(fig.add_subplot(grid[1, 0]), packages)
    plot_summary(fig.add_subplot(grid[1, 1]), packages, blocked)
    fig.suptitle(
        "GARMENT-SIZING-R0A / CP2 — 2D POM & landmark evidence; no triangulation, no Warp"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def reference_parity(packages: dict[str, dict]) -> dict:
    poms = packages["STANDARD_M"]["poms"]
    expected = {
        "front_chest_half_m": 0.275,
        "back_chest_half_m": 0.269,
        "front_waist_half_m": 0.244,
        "back_waist_half_m": 0.244,
        "shoulder_half_m": 0.205,
        "neck_half_m": 0.083,
        "underarm_height_m": 0.285,
        "front_shoulder_height_m": 0.455,
        "back_shoulder_height_m": 0.455,
        "front_top_height_m": 0.525,
        "back_top_height_m": 0.525,
        "front_neck_depth_m": 0.160,
        "back_neck_depth_m": 0.072,
        "skirt_length_m": 0.820,
        "hem_half_m": 0.310,
    }
    errors = {name: abs(float(poms[name]) - value) for name, value in expected.items()}
    return {"expected": expected, "errors": errors, "maximum_error": max(errors.values())}


def publish_status(root: Path, packages: dict[str, dict], blocked: list[dict]) -> dict:
    parity = reference_parity(packages)
    status = {
        "checkpoint": "GARMENT_SIZING_R0A_CP2",
        "terminal_decision": "CP2_COMPLETE_PARAMETERS_ONLY",
        "published_package_count": len(packages),
        "standard_sizes": list(STANDARD_SIZES),
        "custom_package_count": len(packages) - len(STANDARD_SIZES),
        "blocked_count": len(blocked),
        "reference_parity_max_error": parity["maximum_error"],
        "selection_receipts_mutated": False,
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "next_checkpoint": "GARMENT_SIZING_R0A_CP3",
    }
    build = root / "build/tunic_pilot/sizing_cp2"
    write_json(build / "reference_parity.json", parity)
    write_json(build / "cp2_receipt.json", status)
    write_json(root / "SIZING_STATUS.json", status)
    return status


def write_report(root: Path, status: dict) -> None:
    report = f"""# GARMENT-SIZING-R0A / CP2 실행 보고서

## 판정

```text
terminal decision          {status['terminal_decision']}
published packages         {status['published_package_count']}
blocked admissions         {status['blocked_count']}
reference parity max error {status['reference_parity_max_error']}
triangulation              false
Warp simulation            false
```

S/M/L과 지원되는 상세 치수 fixture를 실제 POM 및 named 2D landmark package로 발행했다.
CP1 selection receipt는 읽기 전용 입력이며 변경하지 않았다. 차단 admission은 pattern package를
발행하지 않는다. 현재 결과는 2D parameter authority이고 polygon tessellation이나 cloth solve가 아니다.
"""
    path = root / "docs/cp2b/GARMENT_SIZING_CP2_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`GARMENT-SIZING-R0A / CP3 — DYNAMIC PANEL CURVES / BOUNDARY RESAMPLING / "
        "SEAM CORRESPONDENCE / SIZE-AWARE TRIANGULATION ADMISSION`\n\n"
        "Consume CP2 pattern-parameter packages without changing CP1 receipts. Build actual panel "
        "curves, normalized arc-length seam pairs, and size-aware meshing admission. Do not run "
        "the 180-frame Warp product simulation yet.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    publish_contracts(root)
    packages, blocked = execute_packages(root)
    evidence = root / "build/tunic_pilot/sizing_cp2/cp2_pattern_parameter_evidence.png"
    render_evidence(evidence, packages, blocked)
    status = publish_status(root, packages, blocked)
    write_report(root, status)
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
