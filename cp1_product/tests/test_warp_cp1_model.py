from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wuxia_garment_oss.drape.warp_backend.authority import (
    EXPECTED_ATTACHMENTS,
    EXPECTED_SEAM_PAIRS,
    EXPECTED_TRIANGLES,
    EXPECTED_VERTICES,
    SEAM_COUNTS,
)
from wuxia_garment_oss.drape.warp_backend.model.contract import validate_model_arrays
from wuxia_garment_oss.drape.warp_backend.model.package import compile_model, publish_model


@pytest.fixture(scope="module")
def compiled():
    return compile_model()


def test_exact_static_authority_counts(compiled) -> None:
    arrays = compiled.arrays
    assert arrays["positions_initial"].shape == (EXPECTED_VERTICES, 3)
    assert arrays["triangles"].shape == (EXPECTED_TRIANGLES, 3)
    assert arrays["seam_pairs"].shape == (EXPECTED_SEAM_PAIRS, 2)
    assert arrays["attachment_indices"].shape == (EXPECTED_ATTACHMENTS,)
    assert len(compiled.native.seam_names) == 8


def test_dual_area_mass_is_conservative(compiled) -> None:
    arrays = compiled.arrays
    assert np.all(arrays["vertex_mass"] > 0.0)
    expected = compiled.mass.total_area_m2 * compiled.metadata["density_kg_m2"]
    assert compiled.mass.total_mass_kg == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
    assert float(np.sum(arrays["vertex_mass"], dtype=np.float64)) == pytest.approx(expected, rel=2.0e-7)


def test_grain_basis_is_finite_and_normalized(compiled) -> None:
    axes = compiled.arrays["grain_axes"]
    assert np.isfinite(compiled.arrays["rest_triangle_basis"]).all()
    assert np.isfinite(compiled.arrays["rest_uv_inverse"]).all()
    assert np.allclose(np.linalg.norm(axes, axis=2), 1.0, atol=2.0e-6)
    assert np.all(np.abs(compiled.arrays["uv_determinant"]) > 1.0e-12)


def test_bending_owns_all_manifold_interior_edges(compiled) -> None:
    assert compiled.bending.boundary_edge_count == 767
    assert len(compiled.arrays["interior_edges"]) == 39_860
    assert np.all(compiled.arrays["rest_edge_length"] > 0.0)
    assert np.isfinite(compiled.arrays["rest_dihedral"]).all()


def test_named_seams_and_attachments_are_exact(compiled) -> None:
    arrays = compiled.arrays
    expected_counts = np.asarray(
        [SEAM_COUNTS[name] for name in compiled.native.seam_names], dtype=np.int64
    )
    assert np.array_equal(np.bincount(arrays["seam_ids"], minlength=8), expected_counts)
    assert len(np.unique(arrays["attachment_indices"])) == EXPECTED_ATTACHMENTS
    assert float(np.max(arrays["seam_rest_length"])) <= 0.0135
    assert float(np.max(arrays["attachment_rest_distance"])) <= 0.00601


def test_model_contract_and_publication(tmp_path: Path, compiled) -> None:
    expected_hash = compiled.metadata["contract_validation"]["canonical_array_sha256"]
    assert validate_model_arrays(compiled.arrays)["canonical_array_sha256"] == expected_hash
    metadata = publish_model(tmp_path)
    model_path = tmp_path / "build/tunic_pilot/warp_cp1/warp_model_package.npz"
    native_path = tmp_path / "build/tunic_pilot/native_input_mesh.npz"
    assert model_path.is_file() and native_path.is_file()
    with np.load(model_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    assert validate_model_arrays(arrays)["canonical_array_sha256"] == expected_hash
    written = json.loads(
        (tmp_path / "fixtures/pilot/tunic/warp_model_package.json").read_text(encoding="utf-8")
    )
    assert written == metadata
    assert written["contact_model_included"] is False
    assert written["simulation_frames_executed"] == 0
