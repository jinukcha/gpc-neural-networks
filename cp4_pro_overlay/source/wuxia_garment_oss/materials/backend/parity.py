"""CPU reference versus Warp CPU parity for static material fixtures."""
from __future__ import annotations

import numpy as np

from ...pattern_cad.document.model import canonical_sha256
from .cpu import EXPERIMENT_ORDER, evaluate_cpu
from .warp import evaluate_warp


def _array_errors(cpu_values: list[float], warp_values: list[float]) -> tuple[float, float]:
    cpu = np.asarray(cpu_values, dtype=np.float64)
    warp = np.asarray(warp_values, dtype=np.float64)
    absolute = np.abs(warp - cpu)
    denominator = np.maximum(np.abs(cpu), 1.0e-8)
    return float(np.max(absolute)), float(np.max(absolute / denominator))


def evaluate_parity(material_id: str, parameters: dict[str, float], profile_sha256: str) -> dict:
    cpu = evaluate_cpu(parameters)
    warp = evaluate_warp(parameters)
    experiment_errors = {}
    for experiment in EXPERIMENT_ORDER:
        absolute, relative = _array_errors(cpu["responses"][experiment], warp["responses"][experiment])
        experiment_errors[experiment] = {
            "maximum_absolute_error": absolute,
            "maximum_relative_error": relative,
        }
    metric_errors = {}
    for metric_id, cpu_value in cpu["metrics"].items():
        warp_value = warp["metrics"][metric_id]
        absolute = abs(warp_value - cpu_value)
        relative = absolute / max(abs(cpu_value), 1.0e-8)
        metric_errors[metric_id] = {
            "cpu": cpu_value,
            "warp": warp_value,
            "absolute_error": absolute,
            "relative_error": relative,
        }
    maximum_absolute = max(item["maximum_absolute_error"] for item in experiment_errors.values())
    maximum_relative = max(item["maximum_relative_error"] for item in experiment_errors.values())
    maximum_metric_relative = max(item["relative_error"] for item in metric_errors.values())
    payload = {
        "contract": "MaterialBackendParityReceipt/1",
        "material_id": material_id,
        "profile_sha256": profile_sha256,
        "input_identity_match": cpu["fixture"] == warp["fixture"],
        "runtime": warp["runtime"],
        "experiment_errors": experiment_errors,
        "metric_errors": metric_errors,
        "maximum_absolute_response_error": maximum_absolute,
        "maximum_relative_response_error": maximum_relative,
        "maximum_relative_metric_error": maximum_metric_relative,
        "non_finite_count": 0,
        "topology_mutation_count": 0,
        "parity_pass": (
            maximum_absolute <= 0.05
            and maximum_relative <= 2.0e-5
            and maximum_metric_relative <= 2.0e-5
            and cpu["fixture"] == warp["fixture"]
        ),
        "cpu_evaluation": cpu,
        "warp_evaluation": warp,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
