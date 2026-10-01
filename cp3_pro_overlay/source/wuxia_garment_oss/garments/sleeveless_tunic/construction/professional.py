"""Professional construction authority for the CP2 feature-complete tunic."""
from __future__ import annotations

from .....construction.model import (
    AssemblyOperation,
    BoundaryInterval,
    ClosureSpec,
    ConstructionGraph,
    EdgeFinishSpec,
    FacingSpec,
    LayerPieceSpec,
    NotchMatch,
    SeamSpecV2,
    TurnOfClothSpec,
)


GUSSET_PREFIX = "gusset_underarm.feature.GUSSET_UNDERARM"


def _boundary(panel: str, curve: str, start: float = 0.0, end: float = 1.0) -> BoundaryInterval:
    return BoundaryInterval(panel, curve, start, end)


def _seams() -> tuple[SeamSpecV2, ...]:
    plain = "ISO_301_LOCKSTITCH"
    return (
        SeamSpecV2("shoulder_left", "PLAIN_SEAM", _boundary("bodice_front", "bodice_front.shoulder_left"), _boundary("bodice_back", "bodice_back.shoulder_left"), 0.012, 0.012, plain, notch_pair_ids=("NP_SHOULDER_LEFT",)),
        SeamSpecV2("shoulder_right", "PLAIN_SEAM", _boundary("bodice_front", "bodice_front.shoulder_right"), _boundary("bodice_back", "bodice_back.shoulder_right"), 0.012, 0.012, plain, notch_pair_ids=("NP_SHOULDER_RIGHT",)),
        SeamSpecV2("bodice_side_left", "PLAIN_SEAM", _boundary("bodice_front", "bodice_front.side_left", 0.22, 1.0), _boundary("bodice_back", "bodice_back.side_left", 0.22, 1.0), 0.015, 0.015, plain, notch_pair_ids=("NP_BODICE_SIDE_LEFT",)),
        SeamSpecV2("bodice_side_right", "PLAIN_SEAM", _boundary("bodice_front", "bodice_front.side_right", 0.0, 0.78), _boundary("bodice_back", "bodice_back.side_right", 0.0, 0.78), 0.015, 0.015, plain, notch_pair_ids=("NP_BODICE_SIDE_RIGHT",)),
        SeamSpecV2("waist_front", "WAIST_JOIN", _boundary("bodice_front", "bodice_front.waist"), _boundary("skirt_front", "skirt_front.waist"), 0.012, 0.012, plain, notch_pair_ids=("NP_WAIST_FRONT",)),
        SeamSpecV2("waist_back", "GATHERED_SEAM", _boundary("bodice_back", "bodice_back.waist"), _boundary("skirt_back", "skirt_back.waist"), 0.012, 0.012, plain, gather_ratio=1.12, notch_pair_ids=("NP_WAIST_BACK",)),
        SeamSpecV2("skirt_side_left", "PLAIN_SEAM", _boundary("skirt_front", "skirt_front.side_left"), _boundary("skirt_back", "skirt_back.side_left"), 0.015, 0.015, plain, notch_pair_ids=("NP_SKIRT_SIDE_LEFT",)),
        SeamSpecV2("skirt_side_right", "PLAIN_SEAM", _boundary("skirt_front", "skirt_front.side_right"), _boundary("skirt_back", "skirt_back.side_right"), 0.015, 0.015, plain, notch_pair_ids=("NP_SKIRT_SIDE_RIGHT",)),
        SeamSpecV2("gusset_front_left", "GUSSET_INSERTION", _boundary("bodice_front", "bodice_front.side_left", 0.0, 0.22), _boundary("gusset_underarm", f"{GUSSET_PREFIX}.edge_0"), 0.012, 0.012, plain, notch_pair_ids=("NP_GUSSET_FRONT_LEFT",)),
        SeamSpecV2("gusset_back_left", "GUSSET_INSERTION", _boundary("bodice_back", "bodice_back.side_left", 0.0, 0.22), _boundary("gusset_underarm", f"{GUSSET_PREFIX}.edge_1"), 0.012, 0.012, plain, notch_pair_ids=("NP_GUSSET_BACK_LEFT",)),
        SeamSpecV2("gusset_front_right", "GUSSET_INSERTION", _boundary("bodice_front", "bodice_front.side_right", 0.78, 1.0), _boundary("gusset_underarm", f"{GUSSET_PREFIX}.edge_2"), 0.012, 0.012, plain, notch_pair_ids=("NP_GUSSET_FRONT_RIGHT",)),
        SeamSpecV2("gusset_back_right", "GUSSET_INSERTION", _boundary("bodice_back", "bodice_back.side_right", 0.78, 1.0), _boundary("gusset_underarm", f"{GUSSET_PREFIX}.edge_3"), 0.012, 0.012, plain, notch_pair_ids=("NP_GUSSET_BACK_RIGHT",)),
    )


def _notches() -> tuple[NotchMatch, ...]:
    return (
        NotchMatch("NP_SHOULDER_LEFT", "shoulder_left", "SHOULDER_MATCH", "N_SHOULDER_L_FRONT", "N_SHOULDER_L_BACK", 0.55, 0.55),
        NotchMatch("NP_SHOULDER_RIGHT", "shoulder_right", "SHOULDER_MATCH", "N_SHOULDER_R_FRONT", "N_SHOULDER_R_BACK", 0.45, 0.45),
        NotchMatch("NP_BODICE_SIDE_LEFT", "bodice_side_left", "SIDE_MATCH", "N_SIDE_L_FRONT", "N_SIDE_L_BACK", 0.40, 0.40),
        NotchMatch("NP_BODICE_SIDE_RIGHT", "bodice_side_right", "SIDE_MATCH", "N_SIDE_R_FRONT", "N_SIDE_R_BACK", 0.60, 0.60),
        NotchMatch("NP_WAIST_FRONT", "waist_front", "WAIST_CENTER", "N_WAIST_FRONT_BODICE", "N_WAIST_FRONT_SKIRT", 0.50, 0.50),
        NotchMatch("NP_WAIST_BACK", "waist_back", "WAIST_CENTER", "N_WAIST_BACK_BODICE", "N_WAIST_BACK_SKIRT", 0.50, 0.50),
        NotchMatch("NP_SKIRT_SIDE_LEFT", "skirt_side_left", "SKIRT_SIDE_MATCH", "CP3_N_SKIRT_L_FRONT", "CP3_N_SKIRT_L_BACK", 0.50, 0.50),
        NotchMatch("NP_SKIRT_SIDE_RIGHT", "skirt_side_right", "SKIRT_SIDE_MATCH", "CP3_N_SKIRT_R_FRONT", "CP3_N_SKIRT_R_BACK", 0.50, 0.50),
        NotchMatch("NP_GUSSET_FRONT_LEFT", "gusset_front_left", "GUSSET_MATCH", "CP3_N_GFL_BODY", "CP3_N_GFL_GUSSET", 0.11, 0.50),
        NotchMatch("NP_GUSSET_BACK_LEFT", "gusset_back_left", "GUSSET_MATCH", "CP3_N_GBL_BODY", "CP3_N_GBL_GUSSET", 0.11, 0.50),
        NotchMatch("NP_GUSSET_FRONT_RIGHT", "gusset_front_right", "GUSSET_MATCH", "CP3_N_GFR_BODY", "CP3_N_GFR_GUSSET", 0.89, 0.50),
        NotchMatch("NP_GUSSET_BACK_RIGHT", "gusset_back_right", "GUSSET_MATCH", "CP3_N_GBR_BODY", "CP3_N_GBR_GUSSET", 0.89, 0.50),
    )


def _finishes() -> tuple[EdgeFinishSpec, ...]:
    return (
        EdgeFinishSpec("FINISH_NECK_FRONT", _boundary("bodice_front", "bodice_front.neckline"), "BOUND_FACING", 0.006, "FACING_NECK_FRONT", 0.003),
        EdgeFinishSpec("FINISH_NECK_BACK", _boundary("bodice_back", "bodice_back.neckline"), "BOUND_FACING", 0.006, "FACING_NECK_BACK", 0.003),
        EdgeFinishSpec("FINISH_ARMHOLE_FRONT_LEFT", _boundary("bodice_front", "bodice_front.armhole_left"), "BIAS_BINDING", 0.008, topstitch_offset_m=0.003),
        EdgeFinishSpec("FINISH_ARMHOLE_FRONT_RIGHT", _boundary("bodice_front", "bodice_front.armhole_right"), "BIAS_BINDING", 0.008, topstitch_offset_m=0.003),
        EdgeFinishSpec("FINISH_ARMHOLE_BACK_LEFT", _boundary("bodice_back", "bodice_back.armhole_left"), "BIAS_BINDING", 0.008, topstitch_offset_m=0.003),
        EdgeFinishSpec("FINISH_ARMHOLE_BACK_RIGHT", _boundary("bodice_back", "bodice_back.armhole_right"), "BIAS_BINDING", 0.008, topstitch_offset_m=0.003),
        EdgeFinishSpec("FINISH_HEM_FRONT", _boundary("skirt_front", "skirt_front.hem"), "DOUBLE_TURN_HEM", 0.030, topstitch_offset_m=0.020),
        EdgeFinishSpec("FINISH_HEM_BACK", _boundary("skirt_back", "skirt_back.hem"), "DOUBLE_TURN_HEM", 0.030, topstitch_offset_m=0.020),
    )


def _operations() -> tuple[AssemblyOperation, ...]:
    return (
        AssemblyOperation("OP_010_CLOSE_DART", "CLOSE_DART", ("DART_FRONT_WAIST",)),
        AssemblyOperation("OP_020_FORM_PLEAT", "FORM_PLEAT", ("PLEAT_FRONT_CENTER",)),
        AssemblyOperation("OP_030_PREPARE_GATHER", "PREPARE_GATHER", ("GATHER_BACK_WAIST",)),
        AssemblyOperation("OP_040_PREPARE_GUSSET", "PREPARE_GUSSET", ("GUSSET_UNDERARM",)),
        AssemblyOperation("OP_050_PREPARE_CLOSURE", "PREPARE_CLOSURE", ("CENTER_BACK_NECK_LOOP_BUTTON",)),
        AssemblyOperation("OP_060_PREPARE_FACINGS", "PREPARE_FACING", ("FACING_NECK_FRONT", "FACING_NECK_BACK")),
        AssemblyOperation("OP_070_SEW_SHOULDERS", "SEW_SEAMS", ("shoulder_left", "shoulder_right"), ("OP_010_CLOSE_DART",)),
        AssemblyOperation("OP_080_INSERT_GUSSET_LEFT", "INSERT_GUSSET", ("gusset_front_left", "gusset_back_left"), ("OP_040_PREPARE_GUSSET", "OP_070_SEW_SHOULDERS")),
        AssemblyOperation("OP_090_INSERT_GUSSET_RIGHT", "INSERT_GUSSET", ("gusset_front_right", "gusset_back_right"), ("OP_040_PREPARE_GUSSET", "OP_070_SEW_SHOULDERS")),
        AssemblyOperation("OP_100_SEW_BODICE_SIDES", "SEW_SEAMS", ("bodice_side_left", "bodice_side_right"), ("OP_080_INSERT_GUSSET_LEFT", "OP_090_INSERT_GUSSET_RIGHT")),
        AssemblyOperation("OP_110_JOIN_FRONT_WAIST", "JOIN_WAIST", ("waist_front",), ("OP_020_FORM_PLEAT", "OP_100_SEW_BODICE_SIDES")),
        AssemblyOperation("OP_120_JOIN_BACK_WAIST", "JOIN_GATHERED_WAIST", ("waist_back",), ("OP_030_PREPARE_GATHER", "OP_100_SEW_BODICE_SIDES")),
        AssemblyOperation("OP_130_SEW_SKIRT_SIDES", "SEW_SEAMS", ("skirt_side_left", "skirt_side_right"), ("OP_110_JOIN_FRONT_WAIST", "OP_120_JOIN_BACK_WAIST")),
        AssemblyOperation("OP_140_INSTALL_CLOSURE", "INSTALL_CLOSURE", ("CENTER_BACK_NECK_LOOP_BUTTON",), ("OP_050_PREPARE_CLOSURE", "OP_070_SEW_SHOULDERS")),
        AssemblyOperation("OP_150_ATTACH_FACINGS", "ATTACH_FACING", ("FACING_NECK_FRONT", "FACING_NECK_BACK"), ("OP_060_PREPARE_FACINGS", "OP_140_INSTALL_CLOSURE")),
        AssemblyOperation("OP_160_FINISH_ARMHOLES", "FINISH_OPEN_EDGE", ("FINISH_ARMHOLE_FRONT_LEFT", "FINISH_ARMHOLE_FRONT_RIGHT", "FINISH_ARMHOLE_BACK_LEFT", "FINISH_ARMHOLE_BACK_RIGHT"), ("OP_100_SEW_BODICE_SIDES", "OP_150_ATTACH_FACINGS")),
        AssemblyOperation("OP_170_PREPARE_LINING", "PREPARE_LINING", ("LINING_BODICE_FRONT", "LINING_BODICE_BACK", "LINING_SKIRT_FRONT", "LINING_SKIRT_BACK")),
        AssemblyOperation("OP_180_INSTALL_LINING", "INSTALL_LINING", ("LINING_BODICE_FRONT", "LINING_BODICE_BACK", "LINING_SKIRT_FRONT", "LINING_SKIRT_BACK"), ("OP_130_SEW_SKIRT_SIDES", "OP_160_FINISH_ARMHOLES", "OP_170_PREPARE_LINING")),
        AssemblyOperation("OP_190_HEM_SHELL", "HEM_SHELL", ("FINISH_HEM_FRONT", "FINISH_HEM_BACK"), ("OP_130_SEW_SKIRT_SIDES",)),
        AssemblyOperation("OP_200_HEM_LINING", "HEM_LINING", ("LINING_SKIRT_FRONT", "LINING_SKIRT_BACK"), ("OP_180_INSTALL_LINING",)),
        AssemblyOperation("OP_210_FINAL_TOPSTITCH", "FINAL_TOPSTITCH", ("CENTER_BACK_NECK_LOOP_BUTTON", "FACING_NECK_FRONT", "FACING_NECK_BACK"), ("OP_150_ATTACH_FACINGS", "OP_190_HEM_SHELL", "OP_200_HEM_LINING")),
    )


def tunic_construction_authority() -> dict:
    return {
        "seams": _seams(),
        "notches": _notches(),
        "finishes": _finishes(),
        "closures": (
            ClosureSpec("CENTER_BACK_NECK_LOOP_BUTTON", "LOOP_BUTTON_SLIT", "bodice_back", "bodice_back.neck_center", 0.120, (0.0, -1.0), ("LOOP_TAPE_001", "BUTTON_12MM_001"), "FACING_NECK_BACK"),
        ),
        "facings": (
            FacingSpec("FACING_NECK_FRONT", "bodice_front", ("bodice_front.neckline",), 0.045, 0.006, 0.0012),
            FacingSpec("FACING_NECK_BACK", "bodice_back", ("bodice_back.neckline",), 0.055, 0.006, 0.0012),
        ),
        "layers": (
            LayerPieceSpec("LINING_BODICE_FRONT", "LINING", "bodice_front", 0.006, 0.0),
            LayerPieceSpec("LINING_BODICE_BACK", "LINING", "bodice_back", 0.006, 0.0),
            LayerPieceSpec("LINING_SKIRT_FRONT", "LINING", "skirt_front", 0.006, 0.020),
            LayerPieceSpec("LINING_SKIRT_BACK", "LINING", "skirt_back", 0.006, 0.020),
            LayerPieceSpec("INTERFACE_FRONT_NECK", "INTERFACING", "bodice_front.neckline", bonded=True),
            LayerPieceSpec("INTERFACE_BACK_NECK", "INTERFACING", "bodice_back.neckline", bonded=True),
            LayerPieceSpec("INTERFACE_BACK_CLOSURE", "INTERFACING", "bodice_back", bonded=True),
        ),
        "turns": (
            TurnOfClothSpec("TURN_FRONT_NECK", "FACING_NECK_FRONT", 0.0012, "UNDER_ROLL"),
            TurnOfClothSpec("TURN_BACK_NECK", "FACING_NECK_BACK", 0.0012, "UNDER_ROLL"),
            TurnOfClothSpec("TURN_BACK_CLOSURE", "CENTER_BACK_NECK_LOOP_BUTTON", 0.0008, "INWARD"),
        ),
        "graph": ConstructionGraph("CP2B_TUNIC_CONSTRUCTION_GRAPH_V1", _operations()),
        "bill_of_materials": (
            {"item_id": "SHELL_WOVEN", "category": "FABRIC", "unit": "m2", "quantity": 2.40},
            {"item_id": "LIGHTWEIGHT_LINING", "category": "FABRIC", "unit": "m2", "quantity": 2.10},
            {"item_id": "FUSIBLE_INTERFACING", "category": "INTERFACING", "unit": "m2", "quantity": 0.18},
            {"item_id": "BUTTON_12MM_001", "category": "CLOSURE", "unit": "each", "quantity": 1},
            {"item_id": "LOOP_TAPE_001", "category": "CLOSURE", "unit": "m", "quantity": 0.08},
            {"item_id": "POLY_CORE_THREAD", "category": "THREAD", "unit": "m", "quantity": 68.0},
        ),
    }
