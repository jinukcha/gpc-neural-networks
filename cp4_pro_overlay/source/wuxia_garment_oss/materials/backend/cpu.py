"""CPU reference evaluation for calibrated garment material profiles."""
from __future__ import annotations

import numpy as np


EXPERIMENT_ORDER = (
    "WARP_TENSILE",
    "WEFT_TENSILE",
    "BIAS_SHEAR",
    "WARP_BENDING",
    "WEFT_BENDING",
    "THICKNESS_COMPRESSION",
    "STATIC_FRICTION",
    "DYNAMIC_FRICTION",
    "WARP_SHRINKAGE",
    "WEFT_SHRINKAGE",
)


def static_fixture() -> dict[str, np.ndarray]:
    return {
        "WARP_TENSILE": np.asarray((0.0, 0.013, 0.031, 0.057, 0.079), dtype=np.float64),
        "WEFT_TENSILE": np.asarray((0.0, 0.017, 0.036, 0.061, 0.083), dtype=np.float64),
        "BIAS_SHEAR": np.asarray((0.0, 0.043, 0.097, 0.161, 0.238), dtype=np.float64),
        "WARP_BENDING": np.asarray((0.0, 3.0, 7.0, 13.0, 19.0), dtype=np.float64),
        "WEFT_BENDING": np.asarray((0.0, 5.0, 9.0, 15.0, 21.0), dtype=np.float64),
        "THICKNESS_COMPRESSION": np.asarray((0.0, 0.07, 0.14, 0.23, 0.37), dtype=np.float64),
        "STATIC_FRICTION": np.asarray((1.0, 3.0, 7.0, 11.0), dtype=np.float64),
        "DYNAMIC_FRICTION": np.asarray((1.0, 3.0, 7.0, 11.0), dtype=np.float64),
        "WARP_SHRINKAGE": np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
        "WEFT_SHRINKAGE": np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
    }


def _response(parameters: dict[str, float], experiment: str, x: np.ndarray) -> np.ndarray:
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


def response_metrics(responses: dict[str, np.ndarray]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total = 0.0
    for experiment in EXPERIMENT_ORDER:
        values = np.asarray(responses[experiment], dtype=np.float64)
        metrics[f"{experiment}.sum"] = float(np.sum(values))
        metrics[f"{experiment}.maximum"] = float(np.max(values))
        total += float(np.sum(values))
    metrics["TOTAL_RESPONSE_SUM"] = total
    metrics["SHRUNK_AREA_RATIO"] = float(
        responses["WARP_SHRINKAGE"][0] * responses["WEFT_SHRINKAGE"][0]
    )
    metrics["FRICTION_SPREAD"] = float(
        responses["STATIC_FRICTION"][0] - responses["DYNAMIC_FRICTION"][0]
    )
    return metrics


def evaluate_cpu(parameters: dict[str, float]) -> dict:
    fixture = static_fixture()
    responses = {name: _response(parameters, name, values) for name, values in fixture.items()}
    return {
        "backend": "CPU_REFERENCE_FLOAT64",
        "fixture": {name: values.tolist() for name, values in fixture.items()},
        "responses": {name: values.tolist() for name, values in responses.items()},
        "metrics": response_metrics(responses),
    }
