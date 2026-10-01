"""Deterministic project reference lab series for CP4 calibration."""
from __future__ import annotations

import math

from .model import MaterialMeasurementSet, MeasurementSeries


_NOISE = (0.0, 0.012, -0.009, 0.006, -0.004, 0.003)


def _observed(values: tuple[float, ...], floor: float) -> tuple[float, ...]:
    return tuple(value * (1.0 + _NOISE[index % len(_NOISE)]) for index, value in enumerate(values))


def _sigma(values: tuple[float, ...], relative: float, floor: float) -> tuple[float, ...]:
    return tuple(max(abs(value) * relative, floor) for value in values)


def _series(
    material_id: str,
    experiment: str,
    x_name: str,
    x_unit: str,
    y_name: str,
    y_unit: str,
    x: tuple[float, ...],
    ideal: tuple[float, ...],
    sigma_relative: float,
    sigma_floor: float,
) -> MeasurementSeries:
    measured = _observed(ideal, sigma_floor)
    return MeasurementSeries(
        f"{material_id}.{experiment}",
        experiment,
        x_name,
        x_unit,
        y_name,
        y_unit,
        x,
        measured,
        _sigma(measured, sigma_relative, sigma_floor),
        5,
        f"CP4_REFERENCE_METHOD_{experiment}_V1",
    )


def _tensile(material_id: str, axis: str, linear: float, cubic: float) -> MeasurementSeries:
    x = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08)
    ideal = tuple(linear * value + cubic * value**3 for value in x)
    return _series(material_id, f"{axis}_TENSILE", "engineering_strain", "ratio", "force_per_width", "N/m", x, ideal, 0.025, 3.0)


def _shear(material_id: str, linear: float, cubic: float) -> MeasurementSeries:
    x = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
    ideal = tuple(linear * value + cubic * value**3 for value in x)
    return _series(material_id, "BIAS_SHEAR", "shear_angle", "rad", "shear_force_per_width", "N/m", x, ideal, 0.035, 0.8)


def _bending(material_id: str, axis: str, rigidity: float) -> MeasurementSeries:
    x = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)
    ideal = tuple(rigidity * value for value in x)
    return _series(material_id, f"{axis}_BENDING", "curvature", "1/m", "bending_moment_per_width", "N", x, ideal, 0.04, 1.0e-5)


def _compression(material_id: str, scale: float, exponent: float) -> MeasurementSeries:
    x = (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)
    ideal = tuple(scale * (math.exp(exponent * value) - 1.0) for value in x)
    return _series(material_id, "THICKNESS_COMPRESSION", "thickness_strain", "ratio", "pressure", "Pa", x, ideal, 0.03, 40.0)


def _friction(material_id: str, kind: str, coefficient: float) -> MeasurementSeries:
    x = (1.0, 2.0, 4.0, 8.0, 12.0)
    ideal = tuple(coefficient * (1.0 + 0.008 * math.log1p(value)) for value in x)
    return _series(material_id, kind, "normal_pressure", "kPa", "friction_coefficient", "ratio", x, ideal, 0.025, 0.006)


def _shrinkage(material_id: str, axis: str, ratio: float) -> MeasurementSeries:
    x = (1.0, 2.0, 3.0, 4.0, 5.0)
    ideal = tuple(ratio for _ in x)
    return _series(material_id, f"{axis}_SHRINKAGE", "specimen_index", "index", "post_pre_length_ratio", "ratio", x, ideal, 0.015, 0.004)


def _material(
    material_id: str,
    display_name: str,
    density: float,
    thickness: float,
    mesh_edge: float,
    parameters: dict[str, float],
) -> MaterialMeasurementSet:
    series = (
        _tensile(material_id, "WARP", parameters["warp_linear"], parameters["warp_cubic"]),
        _tensile(material_id, "WEFT", parameters["weft_linear"], parameters["weft_cubic"]),
        _shear(material_id, parameters["shear_linear"], parameters["shear_cubic"]),
        _bending(material_id, "WARP", parameters["bend_warp"]),
        _bending(material_id, "WEFT", parameters["bend_weft"]),
        _compression(material_id, parameters["compression_scale"], parameters["compression_exponent"]),
        _friction(material_id, "STATIC_FRICTION", parameters["friction_static"]),
        _friction(material_id, "DYNAMIC_FRICTION", parameters["friction_dynamic"]),
        _shrinkage(material_id, "WARP", parameters["shrink_warp"]),
        _shrinkage(material_id, "WEFT", parameters["shrink_weft"]),
    )
    result = MaterialMeasurementSet(
        material_id,
        display_name,
        f"CP4_{material_id}_BATCH_001",
        "PROJECT_REFERENCE_LAB_SERIES",
        "NOT_EXTERNAL_LAB_CERTIFIED",
        20.0,
        0.65,
        density,
        thickness,
        mesh_edge,
        series,
        "Deterministic project reference measurements for pipeline qualification; not a supplier certificate.",
    )
    result.validate()
    return result


def reference_material_sets() -> tuple[MaterialMeasurementSet, ...]:
    return (
        _material(
            "LINEN_LIGHT_REFERENCE",
            "Light plain-weave linen reference",
            0.145,
            0.00042,
            0.010,
            {
                "warp_linear": 8500.0, "warp_cubic": 210000.0,
                "weft_linear": 6200.0, "weft_cubic": 175000.0,
                "shear_linear": 230.0, "shear_cubic": 2200.0,
                "bend_warp": 4.2e-5, "bend_weft": 3.2e-5,
                "compression_scale": 1800.0, "compression_exponent": 4.0,
                "friction_static": 0.43, "friction_dynamic": 0.36,
                "shrink_warp": 0.982, "shrink_weft": 0.976,
            },
        ),
        _material(
            "WOOL_TWILL_MEDIUM_REFERENCE",
            "Medium wool twill reference",
            0.265,
            0.00078,
            0.012,
            {
                "warp_linear": 6100.0, "warp_cubic": 145000.0,
                "weft_linear": 4900.0, "weft_cubic": 125000.0,
                "shear_linear": 175.0, "shear_cubic": 1650.0,
                "bend_warp": 7.5e-5, "bend_weft": 6.4e-5,
                "compression_scale": 1250.0, "compression_exponent": 3.6,
                "friction_static": 0.58, "friction_dynamic": 0.49,
                "shrink_warp": 0.971, "shrink_weft": 0.965,
            },
        ),
        _material(
            "COTTON_CANVAS_HEAVY_REFERENCE",
            "Heavy cotton canvas reference",
            0.410,
            0.00125,
            0.016,
            {
                "warp_linear": 15500.0, "warp_cubic": 410000.0,
                "weft_linear": 12800.0, "weft_cubic": 360000.0,
                "shear_linear": 520.0, "shear_cubic": 5200.0,
                "bend_warp": 1.25e-3, "bend_weft": 9.8e-4,
                "compression_scale": 4300.0, "compression_exponent": 5.0,
                "friction_static": 0.51, "friction_dynamic": 0.42,
                "shrink_warp": 0.988, "shrink_weft": 0.984,
            },
        ),
    )
