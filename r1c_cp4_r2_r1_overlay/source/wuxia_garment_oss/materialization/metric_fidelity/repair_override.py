"""Correct the sign of the bounded edge-length projection used by CP4-R2-R1."""
from __future__ import annotations

import numpy as np

from . import repair as _base


def _corrected_iteration(
    positions,
    baseline,
    arrays,
    edge_mask,
    movable,
    relaxation,
    anchor,
):
    edges = arrays["edges"][edge_mask]
    target = arrays["pattern_rest_lengths"][edge_mask]
    first, second = edges[:, 0], edges[:, 1]
    delta = positions[second] - positions[first]
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1.0e-10
    correction = np.zeros_like(delta)
    correction[valid] = (
        relaxation
        * ((length[valid] - target[valid]) / length[valid])[:, None]
        * delta[valid]
    )
    accumulation = np.zeros_like(positions)
    counts = np.zeros(len(positions), dtype=np.float64)
    first_move, second_move = movable[first], movable[second]
    both = first_move & second_move
    np.add.at(accumulation, first[both], 0.5 * correction[both])
    np.add.at(accumulation, second[both], -0.5 * correction[both])
    np.add.at(counts, first[both], 1.0)
    np.add.at(counts, second[both], 1.0)
    only_first = first_move & ~second_move
    only_second = second_move & ~first_move
    np.add.at(accumulation, first[only_first], correction[only_first])
    np.add.at(accumulation, second[only_second], -correction[only_second])
    np.add.at(counts, first[only_first], 1.0)
    np.add.at(counts, second[only_second], 1.0)
    active = movable & (counts > 0.0)
    positions[active] += accumulation[active] / counts[active, None]
    positions[movable] += anchor * (baseline[movable] - positions[movable])


def repair_arrangement(arrays, profile):
    original = _base._iteration
    _base._iteration = _corrected_iteration
    try:
        return _base.repair_arrangement(arrays, profile)
    finally:
        _base._iteration = original
