"""Deterministic safe, guided, HOLD, and rollback fixtures for CP3."""
from __future__ import annotations

from .snapshot import clone_snapshot, refresh_snapshot


def _remove_component(snapshot: dict, instance_id: str) -> None:
    snapshot["geometry_by_instance"].pop(instance_id, None)
    snapshot["assembled_package"]["component_instances"] = [
        item for item in snapshot["assembled_package"]["component_instances"]
        if item["instance_id"] != instance_id
    ]


def _remove_seams(snapshot: dict, interface_ids: set[str]) -> None:
    snapshot["assembled_package"]["seams"] = [
        item for item in snapshot["assembled_package"]["seams"]
        if item["interface_id"] not in interface_ids
    ]


def _seam(snapshot: dict, interface_id: str) -> dict:
    return next(
        item for item in snapshot["assembled_package"]["seams"]
        if item["interface_id"] == interface_id
    )


def _segment(snapshot: dict, instance_id: str, segment_id: str) -> dict:
    return next(
        item for item in snapshot["geometry_by_instance"][instance_id]["segments"]
        if item["segment_id"] == segment_id
    )


def safe_auto_candidate(canonical: dict) -> dict:
    candidate = clone_snapshot(canonical, "R1C_CP3_SAFE_AUTO_CANDIDATE")
    _remove_component(candidate, "cuff_right")
    _remove_seams(candidate, {"cuff_right_attach", "cuff_right_end", "collar_back"})
    _seam(candidate, "left_underarm").pop("seam_allowance_b_m", None)
    sleeve_notches = candidate["geometry_by_instance"]["sleeve_right"]["notches"]
    candidate["geometry_by_instance"]["sleeve_right"]["notches"] = [
        item for item in sleeve_notches if item["notch_id"] != "FRONT_PITCH"
    ]
    armhole = _segment(candidate, "bodice_front", "armhole_left_curve")
    armhole["points_m"][1][0] += 0.0015
    cuff_attach = _segment(candidate, "cuff_left", "sleeve_attach_line")
    cuff_attach["points_m"][1][0] += 0.0012
    return refresh_snapshot(candidate, canonical)


def guided_candidate(canonical: dict) -> dict:
    candidate = clone_snapshot(canonical, "R1C_CP3_GUIDED_CANDIDATE")
    armhole = _segment(candidate, "bodice_front", "armhole_left_curve")
    armhole["points_m"][1][0] += 0.008
    return refresh_snapshot(candidate, canonical)


def hold_candidate(canonical: dict) -> dict:
    candidate = clone_snapshot(canonical, "R1C_CP3_HOLD_CANDIDATE")
    _remove_component(candidate, "collar")
    _remove_seams(candidate, {"collar_front", "collar_back"})
    return refresh_snapshot(candidate, canonical)
