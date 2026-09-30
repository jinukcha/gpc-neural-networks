"""Compile and publish the CP2B Warp static garment model package."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from ..authority import AREAL_DENSITY_KG_M2, authority_summary
from .bending import BendingModel, build_bending_model
from .constraints import StaticConstraints, build_static_constraints
from .contract import validate_model_arrays
from .grain import GrainModel, build_grain_model
from .mass import MassModel, build_dual_area_mass
from .native_input import NativeInput, build_native_input


@dataclass(frozen=True)
class CompiledModel:
    native: NativeInput
    mass: MassModel
    grain: GrainModel
    bending: BendingModel
    constraints: StaticConstraints
    arrays: Mapping[str, np.ndarray]
    metadata: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_arrays(native: NativeInput) -> Dict[str, np.ndarray]:
    return {
        "positions": native.positions,
        "velocities": native.velocities,
        "triangles": native.triangles,
        "uv": native.uv,
        "panel_ids": native.panel_ids,
        "face_panel_ids": native.face_panel_ids,
        "seam_pairs": native.seam_pairs,
        "seam_ids": native.seam_ids,
        "attachment_indices": native.attachment_indices,
        "attachment_targets": native.attachment_targets,
        "attachment_ids": native.attachment_ids,
    }


def _model_arrays(
    native: NativeInput,
    mass: MassModel,
    grain: GrainModel,
    bending: BendingModel,
    constraints: StaticConstraints,
) -> Dict[str, np.ndarray]:
    return {
        "positions_initial": native.positions.astype(np.float32, copy=False),
        "velocities_initial": native.velocities.astype(np.float32, copy=False),
        "triangles": native.triangles.astype(np.int32, copy=False),
        "uv": native.uv.astype(np.float32, copy=False),
        "panel_ids": native.panel_ids.astype(np.int32, copy=False),
        "face_panel_ids": native.face_panel_ids.astype(np.int32, copy=False),
        "triangle_area": mass.triangle_area,
        "vertex_dual_area": mass.vertex_dual_area,
        "vertex_mass": mass.vertex_mass,
        "inverse_mass": mass.inverse_mass,
        "rest_uv_inverse": grain.rest_uv_inverse,
        "rest_triangle_basis": grain.rest_triangle_basis,
        "grain_axes": grain.grain_axes,
        "uv_determinant": grain.uv_determinant,
        "interior_edges": bending.interior_edges,
        "interior_edge_faces": bending.interior_edge_faces,
        "rest_edge_length": bending.rest_edge_length,
        "rest_dihedral": bending.rest_dihedral,
        "seam_pairs": constraints.seam_pairs,
        "seam_ids": constraints.seam_ids,
        "seam_rest_length": constraints.seam_rest_length,
        "seam_group_offsets": constraints.seam_group_offsets,
        "attachment_indices": constraints.attachment_indices,
        "attachment_targets": constraints.attachment_targets,
        "attachment_ids": constraints.attachment_ids,
        "attachment_rest_distance": constraints.attachment_rest_distance,
    }


def compile_model() -> CompiledModel:
    native = build_native_input()
    mass = build_dual_area_mass(native.positions, native.triangles, AREAL_DENSITY_KG_M2)
    grain = build_grain_model(native.positions, native.uv, native.triangles)
    bending = build_bending_model(native.positions, native.triangles)
    constraints = build_static_constraints(native)
    arrays = _model_arrays(native, mass, grain, bending, constraints)
    contract = validate_model_arrays(arrays)
    metadata = {
        "schema_version": 1,
        "contract": "WarpGarmentModelPackage/1",
        "fixture_id": authority_summary()["fixture_id"],
        "authority": authority_summary(),
        "seam_names": list(native.seam_names),
        "density_kg_m2": AREAL_DENSITY_KG_M2,
        "total_area_m2": mass.total_area_m2,
        "total_mass_kg": mass.total_mass_kg,
        "boundary_edges": bending.boundary_edge_count,
        "interior_edges": int(len(bending.interior_edges)),
        "contact_model_included": False,
        "simulation_frames_executed": 0,
        "contract_validation": contract,
    }
    return CompiledModel(native, mass, grain, bending, constraints, arrays, metadata)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_model(root: Path) -> dict:
    compiled = compile_model()
    build = root / "build/tunic_pilot"
    cp1 = build / "warp_cp1"
    fixture = root / "fixtures/pilot/tunic"
    cp1.mkdir(parents=True, exist_ok=True)
    fixture.mkdir(parents=True, exist_ok=True)
    native_path = build / "native_input_mesh.npz"
    model_path = cp1 / "warp_model_package.npz"
    np.savez_compressed(native_path, **_native_arrays(compiled.native))
    np.savez_compressed(model_path, **compiled.arrays)
    metadata = dict(compiled.metadata)
    metadata["native_input"] = {
        "path": native_path.relative_to(root).as_posix(),
        "sha256": _sha256(native_path),
    }
    metadata["model_package"] = {
        "path": model_path.relative_to(root).as_posix(),
        "sha256": _sha256(model_path),
    }
    _write_json(fixture / "warp_model_package.json", metadata)
    _write_json(cp1 / "warp_model_receipt.json", metadata)
    return metadata
