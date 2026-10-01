"""Render raw measurements, calibrated responses, and backend parity."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..calibration.fit import CalibratedMaterialProfile, predict_series
from ..measurement.model import MaterialMeasurementSet


def _plot_tension(ax, measurement: MaterialMeasurementSet, profile: CalibratedMaterialProfile) -> None:
    for experiment in ("WARP_TENSILE", "WEFT_TENSILE", "BIAS_SHEAR"):
        series = measurement.series_by_type(experiment)
        x = np.asarray(series.x, dtype=np.float64)
        ax.errorbar(x, series.y, yerr=series.sigma, marker="o", linestyle="none", label=f"{experiment} raw")
        dense = np.linspace(float(x.min()), float(x.max()), 120)
        ax.plot(dense, predict_series(profile.parameters, experiment, tuple(dense)), label=f"{experiment} fit")
    ax.set_title(f"{measurement.display_name}\ntensile / shear")
    ax.set_xlabel("strain or angle")
    ax.set_ylabel("force per width")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7)


def _plot_compression_bending(ax, measurement: MaterialMeasurementSet, profile: CalibratedMaterialProfile) -> None:
    compression = measurement.series_by_type("THICKNESS_COMPRESSION")
    x = np.asarray(compression.x, dtype=np.float64)
    ax.errorbar(x, compression.y, yerr=compression.sigma, marker="o", linestyle="none", label="compression raw")
    dense = np.linspace(float(x.min()), float(x.max()), 120)
    ax.plot(dense, predict_series(profile.parameters, "THICKNESS_COMPRESSION", tuple(dense)), label="compression fit")
    secondary = ax.twinx()
    for experiment in ("WARP_BENDING", "WEFT_BENDING"):
        series = measurement.series_by_type(experiment)
        secondary.plot(series.x, series.y, marker="x", linestyle="none", label=f"{experiment} raw")
        dense_bend = np.linspace(min(series.x), max(series.x), 80)
        secondary.plot(dense_bend, predict_series(profile.parameters, experiment, tuple(dense_bend)), label=f"{experiment} fit")
    ax.set_title("compression / bending")
    ax.set_xlabel("compression strain or curvature")
    ax.set_ylabel("pressure (Pa)")
    secondary.set_ylabel("bending moment per width")
    ax.grid(True, alpha=0.2)
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = secondary.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b, fontsize=7)


def _plot_residuals(ax, profile: CalibratedMaterialProfile, parity: dict) -> None:
    experiments = list(profile.experiment_metrics)
    residuals = [profile.experiment_metrics[name]["normalized_rmse"] for name in experiments]
    ax.bar(np.arange(len(experiments)), residuals)
    ax.axhline(1.0, linestyle="--")
    ax.set_xticks(np.arange(len(experiments)), [name.replace("_", "\n") for name in experiments], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("normalized RMSE")
    ax.set_title(
        "fit residual / CPU–Warp parity\n"
        f"response rel {parity['maximum_relative_response_error']:.2e}, "
        f"metric rel {parity['maximum_relative_metric_error']:.2e}"
    )
    ax.grid(True, axis="y", alpha=0.2)


def _plot_summary(ax, profiles: tuple[CalibratedMaterialProfile, ...], parities: tuple[dict, ...], resume: dict) -> None:
    ax.axis("off")
    lines = [
        "GARMENT-CAD-PRO-R1A / CP4",
        "",
        "measurement fixtures: 3",
        "required experiment families: 10 each",
        "provenance: PROJECT_REFERENCE_LAB_SERIES",
        "certification: NOT_EXTERNAL_LAB_CERTIFIED",
        "",
        f"fresh-process checkpoint resume: {resume['fresh_process_resume_pass']}",
        f"Warp runtime: {parities[0]['runtime']['version']} / {parities[0]['runtime']['device']}",
        f"CUDA: {parities[0]['runtime']['cuda_status']}",
        "",
    ]
    for profile, parity in zip(profiles, parities):
        receipt = profile.calibration_receipt_dict()
        lines.extend(
            (
                profile.material_id,
                f"  fit RMSE max: {receipt['maximum_normalized_rmse']:.4f}",
                f"  fit sigma max: {receipt['maximum_sigma_error']:.4f}",
                f"  parity response rel: {parity['maximum_relative_response_error']:.2e}",
                f"  meshing edge: {profile.recommended_meshing_edge_m * 1000.0:.1f} mm",
            )
        )
    lines.extend(("", "triangulation: NOT EXECUTED", "motion fit: NOT EXECUTED"))
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax.set_title("authority / runtime summary")


def render_evidence(
    path: Path,
    measurements: tuple[MaterialMeasurementSet, ...],
    profiles: tuple[CalibratedMaterialProfile, ...],
    parities: tuple[dict, ...],
    resume_receipt: dict,
) -> None:
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)
    for index, (measurement, profile, parity) in enumerate(zip(measurements, profiles, parities)):
        _plot_tension(fig.add_subplot(grid[0, index]), measurement, profile)
        _plot_compression_bending(fig.add_subplot(grid[1, index]), measurement, profile)
        _plot_residuals(fig.add_subplot(grid[2, index]), profile, parity)
    summary = fig.add_axes((0.735, 0.015, 0.25, 0.30))
    _plot_summary(summary, profiles, parities, resume_receipt)
    fig.suptitle("Material measurement, calibration, and CPU–Warp parity evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
