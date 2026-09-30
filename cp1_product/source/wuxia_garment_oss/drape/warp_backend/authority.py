"""Static authority for the CP2B Warp garment pilot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class PanelAuthority:
    name: str
    vertex_count: int
    boundary_count: int
    panel_id: int
    front: bool


FIXTURE_ID: Final = "cp2b_sleeveless_long_tunic_v1"
AUTHORITY_MODE: Final = "BOUNDED_R2_R3_CONTRACT_REIMPLEMENTATION"
SOURCE_CP0_SHA256: Final = (
    "92c2f5be6745083466f2ac547f23334493ce1ccbbd8f6d35860142d4283743a2"
)
SOURCE_CP0_AVAILABLE: Final = False
WARP_VERSION: Final = "1.17.0"
WARP_WHEEL_SHA256: Final = (
    "47cd93636828dd16e55eeb4e77a0e3547b5fb0f383000a3d228629df68f28fd8"
)
AREAL_DENSITY_KG_M2: Final = 0.22
INITIAL_SEWING_GAP_M: Final = 0.012

PANELS: Final = (
    PanelAuthority("bodice_front", 2500, 150, 0, True),
    PanelAuthority("bodice_back", 2500, 150, 1, False),
    PanelAuthority("skirt_front", 4401, 234, 2, True),
    PanelAuthority("skirt_back", 4401, 233, 3, False),
)

SEAM_COUNTS: Final = {
    "shoulder_left": 13,
    "shoulder_right": 13,
    "bodice_side_left": 22,
    "bodice_side_right": 22,
    "waist_front": 35,
    "waist_back": 35,
    "skirt_side_left": 66,
    "skirt_side_right": 67,
}

EXPECTED_VERTICES: Final = 13_802
EXPECTED_TRIANGLES: Final = 26_829
EXPECTED_SEAM_PAIRS: Final = 273
EXPECTED_ATTACHMENTS: Final = 52
EXPECTED_PANEL_COUNT: Final = 4


def authority_summary() -> dict:
    return {
        "fixture_id": FIXTURE_ID,
        "mode": AUTHORITY_MODE,
        "source_cp0_sha256": SOURCE_CP0_SHA256,
        "source_cp0_available": SOURCE_CP0_AVAILABLE,
        "warp_version": WARP_VERSION,
        "warp_wheel_sha256": WARP_WHEEL_SHA256,
        "expected": {
            "panels": EXPECTED_PANEL_COUNT,
            "vertices": EXPECTED_VERTICES,
            "triangles": EXPECTED_TRIANGLES,
            "seam_pairs": EXPECTED_SEAM_PAIRS,
            "attachments": EXPECTED_ATTACHMENTS,
        },
        "seam_counts": dict(SEAM_COUNTS),
    }
