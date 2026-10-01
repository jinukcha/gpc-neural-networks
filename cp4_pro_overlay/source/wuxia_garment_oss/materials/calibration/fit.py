"""Deterministic weighted calibration from raw measurement series."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...pattern_cad.document.model import canonical_sha256
from ..measurement.model import MaterialMeasurementSet, MeasurementSeries


@dataclass(frozen=True)
class CalibratedMaterialProfile:
    material_id: str
    measurement_set_sha256: str
    areal_density_kg_m2: float
    thickness_m: float
    recommended_meshing_edge_m: float
    parameters: dict[str, float]
    experiment_metrics: dict[str, dict[str, float]]

    def sizing_profile_dict(self) -> dict:
        payload = {
            "contract": "MaterialSizingProfile/2",
            "material_id": self.material_id,
            "measurement_set_sha256": self.measurement_set_sha256,
            "areal_density_kg_m2": self.areal_density_kg_m2,
            "thickness_m": self.thickness_m,
            "recommended_meshing_edge_m": self.recommended_meshing_edge_m,
            "warp_shrinkage_ratio": self.parameters["warp_shrinkage_ratio"],
            "weft_shrinkage_ratio": self.parameters["weft_shrinkage_ratio"],
            "negative_ease_allowed": False,
        }
        payload["profile_sha256"] = canonical_sha256(payload)
        return payload

    def warp_profile_dict(self) -> dict:
        payload = {
            "contract": "WarpMaterialProfile/1",
            "material_id": self.material_id,
            "measurement_set_sha256": self.measurement_set_sha256,
            "parameters": dict(sorted(self.parameters.items())),
            "calibration_metrics": self.experiment_metrics,
            "solver_mapping": "ORTHOTROPIC_CLOTH_REFERENCE_V1",
        }
        payload["profile_sha256"] = canonical_sha256(payload)
        return payload

    def calibration_receipt_dict(self) -> dict:
        worst_rmse = max(item["normalized_rmse"] for item in self.experiment_metrics.values())
        worst_sigma = max(item["maximum_sigma_error"] for item in self.experiment_metrics.values())
        payload = {
            "contract": "CalibrationExperiment/1",
            "material_id": self.material_id,
            "measurement_set_sha256": self.measurement_set_sha256,
            "experiment_count": len(self.experiment_metrics),
            "maximum_normalized_rmse": worst_rmse,
            "maximum_sigma_error": worst_sigma,
            "calibration_pass": worst_rmse <= 1.0 and worst_sigma <= 1.5,
            "parameter_count": len(self.parameters),
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        return payload


def _weighted_lstsq(series: MeasurementSeries, columns: tuple[np.ndarray, ...]) -> np.ndarray:
    matrix = np.column_stack(columns).astype(np.float64)
    target = np.asarray(series.y, dtype=np.float64)
    weight = 1.0 / np.asarray(series.sigma, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(matrix * weight[:, None], target * weight, rcond=None)
    return solution


def _series_metrics(series: MeasurementSeries, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(series.y, dtype=np.float64)
    sigma = np.asarray(series.sigma, dtype=np.float64)
    residual = predicted - observed
    normalized = residual / sigma
    denominator = np.maximum(np.abs(observed), sigma)
    return {
        "normalized_rmse": float(np.sqrt(np.mean(normalized**2))),
        "maximum_sigma_error": float(np.max(np.abs(normalized))),
        "maximum_relative_error": float(np.max(np.abs(residual) / denominator)),
        "sample_count": float(len(observed)),
    }


def _fit_tensile(series: MeasurementSeries) -> tuple[dict[str, float], np.ndarray]:
    x = np.asarray(series.x, dtype=np.float64)
    linear, cubic = _weighted_lstsq(series, (x, x**3))
    predicted = linear * x + cubic * x**3
    return {"linear": float(linear), "cubic": float(cubic)}, predicted


def _fit_linear(series: MeasurementSeries) -> tuple[float, np.ndarray]:
    x = np.asarray(series.x, dtype=np.float64)
    slope = float(_weighted_lstsq(series, (x,))[0])
    return slope, slope * x


def _fit_compression(series: MeasurementSeries) -> tuple[dict[str, float], np.ndarray]:
    x = np.asarray(series.x, dtype=np.float64)
    observed = np.asarray(series.y, dtype=np.float64)
    sigma = np.asarray(series.sigma, dtype=np.float64)
    best: tuple[float, float, np.ndarray] | None = None
    for exponent in np.linspace(2.0, 6.5, 181):
        basis = np.exp(exponent * x) - 1.0
        scale = float(np.sum(basis * observed / sigma**2) / np.sum(basis**2 / sigma**2))
        predicted = scale * basis
        score = float(np.mean(((predicted - observed) / sigma) ** 2))
        if best is None or score < best[0]:
            best = (score, scale, predicted)
            best_exponent = float(exponent)
    if best is None:
        raise AssertionError("compression calibration produced no candidate")
    return {"scale": best[1], "exponent": best_exponent}, best[2]


def _fit_weighted_mean(series: MeasurementSeries) -> tuple[float, np.ndarray]:
    observed = np.asarray(series.y, dtype=np.float64)
    weight = 1.0 / np.asarray(series.sigma, dtype=np.float64) ** 2
    mean = float(np.sum(observed * weight) / np.sum(weight))
    return mean, np.full_like(observed, mean)


def predict_series(parameters: dict[str, float], experiment: str, x_values: tuple[float, ...]) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)
    if experiment == "WARP_TENSILE":
        return parameters["warp_tensile_linear"] * x + parameters["warp_tensile_cubic"] * x**3
    if experiment == "WEFT_TENSILE":
        return parameters["weft_tensile_linear"] * x + parameters["weft_tensile_cubic"] * x**3
    if experiment == "BIAS_SHEAR":
        return parameters["shear_linear"] * x + parameters["shear_cubic"] * x**3
    if experiment == "WARP_BENDING":
        return parameters["warp_bending_rigidity"] * x
    if experiment == "WEFT_BENDING":
        return parameters["weft_bending_rigidity"] * x
    if experiment == "THICKNESS_COMPRESSION":
        return parameters["compression_scale"] * (np.exp(parameters["compression_exponent"] * x) - 1.0)
    if experiment == "STATIC_FRICTION":
        return np.full_like(x, parameters["friction_static"])
    if experiment == "DYNAMIC_FRICTION":
        return np.full_like(x, parameters["friction_dynamic"])
    if experiment == "WARP_SHRINKAGE":
        return np.full_like(x, parameters["warp_shrinkage_ratio"])
    if experiment == "WEFT_SHRINKAGE":
        return np.full_like(x, parameters["weft_shrinkage_ratio"])
    raise KeyError(experiment)


def calibrate_material(measurement: MaterialMeasurementSet) -> CalibratedMaterialProfile:
    measurement.validate()
    parameters: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    for experiment in ("WARP_TENSILE", "WEFT_TENSILE", "BIAS_SHEAR"):
        series = measurement.series_by_type(experiment)
        fitted, predicted = _fit_tensile(series)
        prefix = {"WARP_TENSILE": "warp_tensile", "WEFT_TENSILE": "weft_tensile", "BIAS_SHEAR": "shear"}[experiment]
        parameters[f"{prefix}_linear"] = fitted["linear"]
        parameters[f"{prefix}_cubic"] = fitted["cubic"]
        metrics[experiment] = _series_metrics(series, predicted)
    for experiment, name in (("WARP_BENDING", "warp_bending_rigidity"), ("WEFT_BENDING", "weft_bending_rigidity")):
        series = measurement.series_by_type(experiment)
        value, predicted = _fit_linear(series)
        parameters[name] = value
        metrics[experiment] = _series_metrics(series, predicted)
    compression = measurement.series_by_type("THICKNESS_COMPRESSION")
    fitted, predicted = _fit_compression(compression)
    parameters["compression_scale"] = fitted["scale"]
    parameters["compression_exponent"] = fitted["exponent"]
    metrics["THICKNESS_COMPRESSION"] = _series_metrics(compression, predicted)
    for experiment, name in (
        ("STATIC_FRICTION", "friction_static"),
        ("DYNAMIC_FRICTION", "friction_dynamic"),
        ("WARP_SHRINKAGE", "warp_shrinkage_ratio"),
        ("WEFT_SHRINKAGE", "weft_shrinkage_ratio"),
    ):
        series = measurement.series_by_type(experiment)
        value, predicted = _fit_weighted_mean(series)
        parameters[name] = value
        metrics[experiment] = _series_metrics(series, predicted)
    payload = measurement.to_dict()
    return CalibratedMaterialProfile(
        measurement.material_id,
        payload["measurement_set_sha256"],
        measurement.areal_density_kg_m2,
        measurement.thickness_m,
        measurement.recommended_meshing_edge_m,
        parameters,
        metrics,
    )
