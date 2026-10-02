"""Bounded secondary-motion product kernel."""

from .contracts import SecondaryMotionDomain, SecondaryMotionProfile, product_profile
from .morph import SECONDARY_TARGETS, compile_secondary_motion_glb

__all__ = [
    "SECONDARY_TARGETS",
    "SecondaryMotionDomain",
    "SecondaryMotionProfile",
    "compile_secondary_motion_glb",
    "product_profile",
]
