"""Compile bounded secondary-motion morph targets into accepted CP5 products."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wuxia_garment_oss.rig.game_product.glb_skin import (
    _accessor_array,
    _append_accessor,
    read_glb,
    write_glb,
)


SECONDARY_TARGETS = (
    "SECONDARY_L_SLEEVE",
    "SECONDARY_R_SLEEVE",
    "SECONDARY_ROBE_HEM",
)
CP5_MESH_SUFFIX = "_CP5_COMPONENTS"


def _component_name(primitive: dict) -> str:
    return str(primitive.get("extras", {}).get("componentName", ""))


def _sleeve_longitudinal(positions: np.ndarray) -> np.ndarray:
    lateral = np.abs(positions[:, 0].astype(np.float64))
    low, high = float(np.min(lateral)), float(np.max(lateral))
    return np.clip((lateral - low) / max(high - low, 1.0e-9), 0.0, 1.0)


def _left_sleeve_delta(positions: np.ndarray, component: str) -> np.ndarray:
    if not component.startswith("LEFT_") or "SLEEVE" not in component:
        return np.zeros_like(positions, dtype=np.float32)
    t = _sleeve_longitudinal(positions)
    delta = np.zeros_like(positions, dtype=np.float64)
    delta[:, 1] = 0.010 * np.sin(np.pi * t) * t
    delta[:, 2] = 0.040 * np.square(t)
    delta[:, 0] = -0.006 * np.square(t)
    return delta.astype(np.float32)


def _right_sleeve_delta(positions: np.ndarray, component: str) -> np.ndarray:
    if not component.startswith("RIGHT_") or "SLEEVE" not in component:
        return np.zeros_like(positions, dtype=np.float32)
    t = _sleeve_longitudinal(positions)
    delta = np.zeros_like(positions, dtype=np.float64)
    delta[:, 1] = 0.010 * np.sin(np.pi * t) * t
    delta[:, 2] = -0.040 * np.square(t)
    delta[:, 0] = 0.006 * np.square(t)
    return delta.astype(np.float32)


def _robe_hem_delta(positions: np.ndarray, component: str) -> np.ndarray:
    if component != "STRAIGHT_ROBE_SKIRT":
        return np.zeros_like(positions, dtype=np.float32)
    y = positions[:, 1].astype(np.float64)
    top, bottom = float(np.max(y)), float(np.min(y))
    t = np.clip((top - y) / max(top - bottom, 1.0e-9), 0.0, 1.0)
    angle = np.arctan2(positions[:, 2].astype(np.float64), positions[:, 0].astype(np.float64))
    delta = np.zeros_like(positions, dtype=np.float64)
    delta[:, 0] = 0.025 * np.square(t) * np.cos(angle + 0.35)
    delta[:, 2] = 0.060 * np.square(t) * np.sin(angle + 0.35)
    delta[:, 1] = 0.008 * np.square(t) * np.cos(2.0 * angle)
    return delta.astype(np.float32)


def _target_arrays(positions: np.ndarray, component: str) -> tuple[np.ndarray, ...]:
    return (
        _left_sleeve_delta(positions, component),
        _right_sleeve_delta(positions, component),
        _robe_hem_delta(positions, component),
    )


def _append_targets(glb, mesh: dict) -> list[dict]:
    component_receipts = []
    for primitive in mesh.get("primitives", []):
        component = _component_name(primitive)
        accessor = primitive["attributes"]["POSITION"]
        positions = _accessor_array(glb, accessor).astype(np.float32)
        arrays = _target_arrays(positions, component)
        targets = primitive.setdefault("targets", [])
        targets.extend(
            {"POSITION": _append_accessor(glb, values, 5126, "VEC3", 34962)}
            for values in arrays
        )
        maximum = {
            name: float(np.max(np.linalg.norm(values.astype(np.float64), axis=1)))
            for name, values in zip(SECONDARY_TARGETS, arrays, strict=True)
        }
        component_receipts.append({
            "component_id": component,
            "vertex_count": int(len(positions)),
            "maximum_displacement_m": maximum,
        })
    return component_receipts


def _extend_target_names(mesh: dict) -> tuple[str, ...]:
    extras = mesh.setdefault("extras", {})
    names = [str(item) for item in extras.get("targetNames", [])]
    for name in SECONDARY_TARGETS:
        if name not in names:
            names.append(name)
    extras["targetNames"] = names
    weights = list(mesh.get("weights", []))
    while len(weights) < len(names):
        weights.append(0.0)
    mesh["weights"] = weights
    return tuple(names)


def compile_secondary_motion_glb(source: Path, target: Path, product_kind: str) -> dict:
    glb = read_glb(source)
    meshes = [
        mesh for mesh in glb.document.get("meshes", [])
        if str(mesh.get("name", "")).endswith(CP5_MESH_SUFFIX)
    ]
    if len(meshes) != 1:
        raise ValueError(f"expected one CP5 component mesh, found {len(meshes)}")
    mesh = meshes[0]
    original_names = tuple(str(item) for item in mesh.get("extras", {}).get("targetNames", []))
    components = _append_targets(glb, mesh)
    final_names = _extend_target_names(mesh)
    if final_names[: len(original_names)] != original_names:
        raise ValueError("existing corrective target order changed")
    glb.document.setdefault("asset", {})["generator"] = "GARMENT-CAD-PRO-R1B CP6"
    glb.document.setdefault("extras", {})["r1bCp6SecondaryMotion"] = {
        "contract": "SecondaryMotionMorphSet/1",
        "productKind": product_kind,
        "targetNames": list(SECONDARY_TARGETS),
        "correctivePrefixPreserved": True,
    }
    write_glb(target, glb)
    return {
        "contract": "SecondaryMotionMorphSet/1",
        "product_kind": product_kind,
        "source_path": source.name,
        "target_path": target.name,
        "original_target_names": list(original_names),
        "final_target_names": list(final_names),
        "secondary_target_names": list(SECONDARY_TARGETS),
        "components": components,
        "corrective_prefix_preserved": True,
    }
