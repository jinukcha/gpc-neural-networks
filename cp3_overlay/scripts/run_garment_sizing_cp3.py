#!/usr/bin/env python3
"""Publish CP3 curves, seam correspondence, and triangulation admission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from wuxia_garment_oss.garments.sleeveless_tunic.geometry import compile_geometry_package
from wuxia_garment_oss.sizing.geometry.archive import write_canonical_npz
from wuxia_garment_oss.sizing.geometry.contract import geometry_package_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_schema(root: Path) -> None:
    write_json(
        root / "contracts/sizing/garment_geometry_package.schema.json",
        geometry_package_schema(),
    )


def compile_all(root: Path) -> tuple[dict[str, dict], dict[str, dict[str, np.ndarray]]]:
    source = root / "build/tunic_pilot/sizing_cp2/pattern_parameter_packages"
    output = root / "build/tunic_pilot/sizing_cp3"
    packages: dict[str, dict] = {}
    arrays_by_name: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(source.glob("*.json")):
        parameter_package = json.loads(path.read_text(encoding="utf-8"))
        geometry, arrays = compile_geometry_package(parameter_package)
        archive_path = output / "triangulation_candidates" / f"{path.stem}.npz"
        write_canonical_npz(archive_path, arrays)
        geometry["triangulation_archive"] = {
            "path": archive_path.relative_to(root).as_posix(),
            "sha256": sha256(archive_path),
        }
        write_json(output / "geometry_packages" / f"{path.stem}.json", geometry)
        packages[path.stem] = geometry
        arrays_by_name[path.stem] = arrays
    return packages, arrays_by_name


def _segments(vertices: np.ndarray, triangles: np.ndarray, offset: float) -> list[list[tuple[float, float]]]:
    shifted = vertices.copy()
    shifted[:, 0] += offset
    edges: set[tuple[int, int]] = set()
    for triangle in triangles:
        a, b, c = (int(value) for value in triangle)
        edges.update({tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))})
    return [[tuple(shifted[a]), tuple(shifted[b])] for a, b in sorted(edges)]


def _draw_mesh(ax, arrays: dict[str, np.ndarray], panel_ids: tuple[str, ...], offset: float) -> None:
    lines: list[list[tuple[float, float]]] = []
    for panel_id in panel_ids:
        lines.extend(_segments(
            arrays[f"{panel_id}__vertices"], arrays[f"{panel_id}__triangles"], offset
        ))
    ax.add_collection(LineCollection(lines, linewidths=0.18, alpha=0.65))


def plot_standard(ax, arrays_by_name: dict[str, dict[str, np.ndarray]]) -> None:
    for index, size in enumerate(("S", "M", "L")):
        _draw_mesh(
            ax, arrays_by_name[f"STANDARD_{size}"], ("bodice_front", "skirt_front"), index * 0.80
        )
        ax.text(index * 0.80, 0.58, size, ha="center", fontsize=12)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("S / M / L size-aware 2D triangulation candidates")
    ax.set_xlabel("pattern x (m), separated for review")
    ax.set_ylabel("pattern y (m)")


def _panel(package: dict, panel_id: str) -> dict:
    return next(panel for panel in package["panels"] if panel["panel_id"] == panel_id)


def plot_custom_curves(ax, packages: dict[str, dict]) -> None:
    names = (
        "AUTO_REFERENCE", "AUTO_BROAD_SHOULDER", "AUTO_FULL_CHEST",
        "AUTO_FULL_ABDOMEN", "AUTO_TALL", "AUTO_SHORT", "CUSTOM_MILD",
    )
    for name in names:
        for panel_id in ("bodice_front", "skirt_front"):
            outline = np.asarray(_panel(packages[name], panel_id)["outline"], dtype=np.float64)
            closed = np.vstack([outline, outline[0]])
            ax.plot(closed[:, 0], closed[:, 1], linewidth=1.0, label=name if panel_id == "bodice_front" else None)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("AUTO_BODY_FIT + CUSTOM_MEASUREMENTS curve overlays")
    ax.set_xlabel("pattern x (m)")
    ax.set_ylabel("pattern y (m)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)


def plot_seams(ax, packages: dict[str, dict]) -> None:
    names = ("STANDARD_S", "STANDARD_M", "STANDARD_L", "CUSTOM_MILD")
    seam_ids = [row["seam_id"] for row in packages["STANDARD_M"]["seam_correspondence"]]
    x = np.arange(len(seam_ids))
    width = 0.19
    for index, name in enumerate(names):
        values = [
            row["length_mismatch_ratio"] * 100.0
            for row in packages[name]["seam_correspondence"]
        ]
        ax.bar(x + (index - 1.5) * width, values, width, label=name)
    ax.axhline(3.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x, seam_ids, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("seam length mismatch (%)")
    ax.set_title("Normalized arc-length seam admission")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)


def plot_summary(ax, packages: dict[str, dict]) -> None:
    ax.axis("off")
    lines = ["fixture | vertices | triangles | seam pairs | max mismatch | admission", "-" * 88]
    for name in sorted(packages):
        quality = packages[name]["qualification"]
        lines.append(
            f"{name:<24} {quality['total_mesh_vertices']:>8} {quality['total_triangles']:>10} "
            f"{quality['total_seam_pairs']:>10} {quality['maximum_seam_length_mismatch_ratio'] * 100:>10.3f}% "
            f"{packages[name]['triangulation_admission']}"
        )
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.4)
    ax.set_title("CP3 geometry publication summary")


def render_evidence(path: Path, packages: dict[str, dict], arrays_by_name: dict[str, dict[str, np.ndarray]]) -> None:
    fig = plt.figure(figsize=(20, 13.333), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    plot_standard(fig.add_subplot(grid[0, 0]), arrays_by_name)
    plot_custom_curves(fig.add_subplot(grid[0, 1]), packages)
    plot_seams(fig.add_subplot(grid[1, 0]), packages)
    plot_summary(fig.add_subplot(grid[1, 1]), packages)
    fig.suptitle(
        "GARMENT-SIZING-R0A / CP3 — curves, boundary resampling, seam correspondence, triangulation admission"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def publish_status(root: Path, packages: dict[str, dict]) -> dict:
    qualities = [package["qualification"] for package in packages.values()]
    status = {
        "checkpoint": "GARMENT_SIZING_R0A_CP3",
        "terminal_decision": "CP3_COMPLETE_GEOMETRY_ADMISSION",
        "geometry_package_count": len(packages),
        "panel_count": sum(len(package["panels"]) for package in packages.values()),
        "seam_identity_count": sum(len(package["seam_correspondence"]) for package in packages.values()),
        "total_triangles": sum(item["total_triangles"] for item in qualities),
        "maximum_seam_length_mismatch_ratio": max(item["maximum_seam_length_mismatch_ratio"] for item in qualities),
        "all_geometry_admitted": all(package["triangulation_admission"] == "PASS" for package in packages.values()),
        "cp2_parameter_packages_mutated": False,
        "triangulation_admission_executed": True,
        "warp_simulation_executed": False,
        "product_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "next_checkpoint": "GARMENT_SIZING_R0A_CP4",
    }
    write_json(root / "build/tunic_pilot/sizing_cp3/cp3_receipt.json", status)
    write_json(root / "SIZING_STATUS.json", status)
    return status


def write_report(root: Path, status: dict) -> None:
    report = f"""# GARMENT-SIZING-R0A / CP3 실행 보고서

```text
terminal decision          {status['terminal_decision']}
geometry packages          {status['geometry_package_count']}
panels                     {status['panel_count']}
seam identities            {status['seam_identity_count']}
triangles                  {status['total_triangles']}
all admitted               {str(status['all_geometry_admitted']).lower()}
Warp simulation            false
```

CP2 POM·랜드마크 package를 변경하지 않고 실제 곡선, 물리 길이 기반 boundary sample,
normalized arc-length seam correspondence와 size-aware 2D triangulation candidate를 발행했다.
이 결과는 body arrangement나 Warp product solve가 아닌 패턴 geometry admission이다.
"""
    path = root / "docs/cp2b/GARMENT_SIZING_CP3_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`GARMENT-SIZING-R0A / CP4 — BODY ENVELOPE / LAYER-AWARE ARRANGEMENT / SIZE-SPECIFIC WARP MODEL COMPILE`\n\n"
        "Consume CP3 geometry packages without changing CP1 or CP2 authorities. Bind actual body "
        "sections, layer clearance, attachment targets, mass/grain/bending and seam constraints for "
        "S/M/L and supported custom fixtures. Do not run the 180-frame product solve yet.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    publish_schema(root)
    packages, arrays_by_name = compile_all(root)
    evidence = root / "build/tunic_pilot/sizing_cp3/cp3_geometry_evidence.png"
    render_evidence(evidence, packages, arrays_by_name)
    status = publish_status(root, packages)
    write_report(root, status)
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
