"""Canonical 2D pattern-parameter package shared by garment families."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..instance.model import canonical_sha256


PUBLISHABLE_ADMISSIONS = {"NORMAL_GRADE", "CUSTOM_ALTERATION"}


@dataclass(frozen=True)
class PatternParameterPackage:
    request_id: str
    garment_design_id: str
    topology_class: str
    sizing_mode: str
    selected_size_id: str
    shape_block: str
    height_block: str
    admission: str
    selection_receipt_sha256: str
    poms: dict[str, float]
    panels: list[dict]
    seam_pairs: list[dict]
    grade_plan: list[dict]
    custom_alteration_plan: list[dict]
    provenance: dict
    warnings: list[str]

    def validate(self) -> None:
        if self.admission not in PUBLISHABLE_ADMISSIONS:
            raise ValueError(f"admission is not publishable: {self.admission}")
        identities = (
            self.request_id,
            self.garment_design_id,
            self.topology_class,
            self.selected_size_id,
            self.selection_receipt_sha256,
        )
        if any(not value for value in identities):
            raise ValueError("pattern package identities are required")
        invalid_poms = [
            name for name, value in self.poms.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid_poms:
            raise ValueError(f"invalid POM values: {invalid_poms}")
        panel_ids = [str(panel.get("panel_id", "")) for panel in self.panels]
        if len(panel_ids) != len(set(panel_ids)) or any(not item for item in panel_ids):
            raise ValueError("panel IDs must be unique and non-empty")
        for panel in self.panels:
            landmarks = panel.get("landmarks", {})
            if not landmarks or not panel.get("boundaries"):
                raise ValueError(f"panel lacks parameter authority: {panel['panel_id']}")
            for point in landmarks.values():
                if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
                    raise ValueError(f"invalid landmark in {panel['panel_id']}")
        seam_ids = [str(pair.get("seam_id", "")) for pair in self.seam_pairs]
        if len(seam_ids) != len(set(seam_ids)) or any(not item for item in seam_ids):
            raise ValueError("seam IDs must be unique and non-empty")

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "GarmentPatternParameterPackage/1",
            "request_id": self.request_id,
            "garment_design_id": self.garment_design_id,
            "topology_class": self.topology_class,
            "units": "m",
            "sizing_mode": self.sizing_mode,
            "selection": {
                "selected_size_id": self.selected_size_id,
                "shape_block": self.shape_block,
                "height_block": self.height_block,
                "admission": self.admission,
                "selection_receipt_sha256": self.selection_receipt_sha256,
            },
            "poms": {name: float(value) for name, value in sorted(self.poms.items())},
            "panels": self.panels,
            "seam_pairs": self.seam_pairs,
            "grade_plan": self.grade_plan,
            "custom_alteration_plan": self.custom_alteration_plan,
            "provenance": self.provenance,
            "warnings": self.warnings,
            "pattern_compile": "PARAMETRIC_POM_AND_LANDMARKS_ONLY",
            "triangulation_executed": False,
            "warp_simulation_executed": False,
            "mesh_scaling": "FORBIDDEN",
        }
        payload["package_id"] = canonical_sha256(payload)
        return payload
