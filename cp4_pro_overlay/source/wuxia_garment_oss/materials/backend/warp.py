"""Exact warp-lang 1.17.0 CPU material-response backend."""
from __future__ import annotations

import numpy as np
import warp as wp

from .cpu import EXPERIMENT_ORDER, response_metrics, static_fixture


@wp.kernel
def _polynomial_response(
    x: wp.array(dtype=float),
    linear: float,
    cubic: float,
    output: wp.array(dtype=float),
):
    index = wp.tid()
    value = x[index]
    output[index] = linear * value + cubic * value * value * value


@wp.kernel
def _linear_response(
    x: wp.array(dtype=float),
    slope: float,
    output: wp.array(dtype=float),
):
    index = wp.tid()
    output[index] = slope * x[index]


@wp.kernel
def _compression_response(
    x: wp.array(dtype=float),
    scale: float,
    exponent: float,
    output: wp.array(dtype=float),
):
    index = wp.tid()
    output[index] = scale * (wp.exp(exponent * x[index]) - 1.0)


@wp.kernel
def _constant_response(
    x: wp.array(dtype=float),
    value: float,
    output: wp.array(dtype=float),
):
    index = wp.tid()
    output[index] = value + 0.0 * x[index]


def _launch(values: np.ndarray, kernel, inputs: list[float]) -> np.ndarray:
    source = wp.array(np.asarray(values, dtype=np.float32), dtype=float, device="cpu")
    output = wp.empty(len(values), dtype=float, device="cpu")
    wp.launch(kernel, dim=len(values), inputs=[source, *inputs, output], device="cpu")
    wp.synchronize()
    return output.numpy().astype(np.float64)


def _runtime_identity() -> dict:
    version = str(getattr(wp, "__version__", "UNKNOWN"))
    if version != "1.17.0":
        raise RuntimeError(f"unexpected Warp runtime: {version}")
    try:
        cuda_available = bool(wp.is_cuda_available())
    except Exception:
        cuda_available = False
    return {
        "package": "warp-lang",
        "version": version,
        "device": "cpu",
        "cuda_status": "AVAILABLE_NOT_USED" if cuda_available else "EXPLICIT_NO_CUDA_DEVICE",
        "kernel_family": "CP4_MATERIAL_RESPONSE_V1",
    }


def _response(parameters: dict[str, float], experiment: str, x: np.ndarray) -> np.ndarray:
    if experiment == "WARP_TENSILE":
        return _launch(x, _polynomial_response, [parameters["warp_tensile_linear"], parameters["warp_tensile_cubic"]])
    if experiment == "WEFT_TENSILE":
        return _launch(x, _polynomial_response, [parameters["weft_tensile_linear"], parameters["weft_tensile_cubic"]])
    if experiment == "BIAS_SHEAR":
        return _launch(x, _polynomial_response, [parameters["shear_linear"], parameters["shear_cubic"]])
    if experiment == "WARP_BENDING":
        return _launch(x, _linear_response, [parameters["warp_bending_rigidity"]])
    if experiment == "WEFT_BENDING":
        return _launch(x, _linear_response, [parameters["weft_bending_rigidity"]])
    if experiment == "THICKNESS_COMPRESSION":
        return _launch(x, _compression_response, [parameters["compression_scale"], parameters["compression_exponent"]])
    constant_name = {
        "STATIC_FRICTION": "friction_static",
        "DYNAMIC_FRICTION": "friction_dynamic",
        "WARP_SHRINKAGE": "warp_shrinkage_ratio",
        "WEFT_SHRINKAGE": "weft_shrinkage_ratio",
    }.get(experiment)
    if constant_name is None:
        raise KeyError(experiment)
    return _launch(x, _constant_response, [parameters[constant_name]])


def evaluate_warp(parameters: dict[str, float]) -> dict:
    wp.init()
    runtime = _runtime_identity()
    fixture = static_fixture()
    responses = {name: _response(parameters, name, values) for name, values in fixture.items()}
    return {
        "backend": "WARP_CPU_FLOAT32",
        "runtime": runtime,
        "fixture": {name: values.tolist() for name, values in fixture.items()},
        "responses": {name: values.tolist() for name, values in responses.items()},
        "metrics": response_metrics(responses),
    }
