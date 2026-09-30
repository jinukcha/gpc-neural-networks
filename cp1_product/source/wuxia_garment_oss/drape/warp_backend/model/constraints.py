"""Static seam and shoulder-attachment constraint ownership."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

from ..authority import EXPECTED_ATTACHMENTS, EXPECTED_SEAM_PAIRS, SEAM_COUNTS
from .native_input import NativeInput


@dataclass(frozen=True)
class StaticConstraints:
    seam_pairs: np.ndarray
    seam_ids: np.ndarray
    seam_rest_length: np.ndarray
    seam_group_offsets: np.ndarray
    attachment_indices: np.ndarray
    attachment_targets: np.ndarray
    attachment_ids: np.ndarray
    attachment_rest_distance: np.ndarray


_EXPECTED_PANEL_PAIRS: Mapping[int, Tuple[int, int]] = {
    0: (0, 1),
    1: (0, 1),
    2: (0, 1),
    3: (0, 1),
    4: (0, 2),
    5: (1, 3),
    6: (2, 3),
    7: (2, 3),
}


def _validate_seam_ownership(native: NativeInput) -> None:
    if len(native.seam_pairs) != EXPECTED_SEAM_PAIRS:
        raise ValueError("seam pair count mismatch")
    for pair, seam_id in zip(native.seam_pairs, native.seam_ids):
        actual = (int(native.panel_ids[pair[0]]), int(native.panel_ids[pair[1]]))
        if actual != _EXPECTED_PANEL_PAIRS[int(seam_id)]:
            raise ValueError(f"seam {int(seam_id)} crosses wrong panel owners: {actual}")
    counts = np.bincount(native.seam_ids, minlength=len(native.seam_names))
    expected = np.asarray([SEAM_COUNTS[name] for name in native.seam_names], dtype=np.int64)
    if not np.array_equal(counts, expected):
        raise ValueError(f"named seam counts differ: {counts.tolist()} != {expected.tolist()}")


def _group_offsets(ids: np.ndarray, group_count: int) -> np.ndarray:
    counts = np.bincount(ids, minlength=group_count).astype(np.int32)
    return np.concatenate((np.asarray([0], dtype=np.int32), np.cumsum(counts, dtype=np.int32)))


def build_static_constraints(native: NativeInput) -> StaticConstraints:
    _validate_seam_ownership(native)
    seam_delta = native.positions[native.seam_pairs[:, 1]] - native.positions[native.seam_pairs[:, 0]]
    seam_rest = np.linalg.norm(seam_delta.astype(np.float64), axis=1)
    if np.any(seam_rest > 0.0135):
        raise ValueError(f"initial sewing gap exceeds admission: {float(np.max(seam_rest)):.9f} m")
    if len(native.attachment_indices) != EXPECTED_ATTACHMENTS:
        raise ValueError("attachment count mismatch")
    if len(np.unique(native.attachment_indices)) != EXPECTED_ATTACHMENTS:
        raise ValueError("attachment vertices are not uniquely owned")
    attachment_delta = native.positions[native.attachment_indices] - native.attachment_targets
    attachment_rest = np.linalg.norm(attachment_delta.astype(np.float64), axis=1)
    if np.any(attachment_rest > 0.00601):
        raise ValueError("shoulder attachment target exceeds half-gap admission")
    return StaticConstraints(
        seam_pairs=native.seam_pairs.astype(np.int32, copy=True),
        seam_ids=native.seam_ids.astype(np.int32, copy=True),
        seam_rest_length=seam_rest.astype(np.float32),
        seam_group_offsets=_group_offsets(native.seam_ids, len(native.seam_names)),
        attachment_indices=native.attachment_indices.astype(np.int32, copy=True),
        attachment_targets=native.attachment_targets.astype(np.float32, copy=True),
        attachment_ids=native.attachment_ids.astype(np.int32, copy=True),
        attachment_rest_distance=attachment_rest.astype(np.float32),
    )
