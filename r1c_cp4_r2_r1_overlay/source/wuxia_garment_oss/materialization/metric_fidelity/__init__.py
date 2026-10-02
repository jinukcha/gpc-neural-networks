"""Component-aware 2D-to-3D metric fidelity analysis and repair."""

from .classify import decompose_metric
from .repair import repair_arrangement

__all__ = ["decompose_metric", "repair_arrangement"]
