"""R1C visual product acceptance authority."""

from .evaluator import evaluate_visual_review
from .model import VisualAcceptanceProfile, VisualGateSpec, ViewRequirement

__all__ = ["VisualAcceptanceProfile", "VisualGateSpec", "ViewRequirement", "evaluate_visual_review"]
