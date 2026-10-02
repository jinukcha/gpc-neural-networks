"""Apply bounded source-pattern repair operations to private working snapshots."""
from __future__ import annotations

from copy import deepcopy

from .snapshot import clone_snapshot, refresh_snapshot


def _canonical_component(canonical: dict, instance_id: str) -> dict:
    for item in canonical["assembled_package"]["component_instances"]:
        if item["instance_id"] == instance_id:
            return deepcopy(item)
    raise KeyError(instance_id)


def _canonical_seam(canonical: dict, interface_id: str) -> dict:
    for item in canonical["assembled_package"]["seams"]:
        if item["interface_id"] == interface_id:
            return deepcopy(item)
    raise KeyError(interface_id)


def _canonical_segment(canonical: dict, instance_id: str, segment_id: str) -> dict:
    geometry = canonical["geometry_by_instance"][instance_id]
    for item in geometry["segments"]:
        if item["segment_id"] == segment_id:
            return deepcopy(item)
    raise KeyError(f"{instance_id}.{segment_id}")


def _canonical_notch(canonical: dict, instance_id: str, notch_id: str) -> dict:
    geometry = canonical["geometry_by_instance"][instance_id]
    for item in geometry["notches"]:
        if item["notch_id"] == notch_id:
            return deepcopy(item)
    raise KeyError(f"{instance_id}.{notch_id}")


def _restore_component(snapshot: dict, canonical: dict, instance_id: str) -> None:
    snapshot["geometry_by_instance"][instance_id] = deepcopy(
        canonical["geometry_by_instance"][instance_id]
    )
    snapshot["assembled_package"]["component_instances"].append(
        _canonical_component(canonical, instance_id)
    )


def _restore_seam(snapshot: dict, canonical: dict, interface_id: str) -> None:
    snapshot["assembled_package"]["seams"].append(
        _canonical_seam(canonical, interface_id)
    )


def _restore_seam_field(snapshot: dict, canonical: dict, interface_id: str, field: str) -> None:
    expected = _canonical_seam(canonical, interface_id)
    for item in snapshot["assembled_package"]["seams"]:
        if item["interface_id"] == interface_id:
            item[field] = deepcopy(expected[field])
            return
    raise KeyError(interface_id)


def _restore_notch(snapshot: dict, canonical: dict, instance_id: str, notch_id: str) -> None:
    snapshot["geometry_by_instance"][instance_id]["notches"].append(
        _canonical_notch(canonical, instance_id, notch_id)
    )


def _restore_segment(snapshot: dict, canonical: dict, instance_id: str, segment_id: str) -> None:
    replacement = _canonical_segment(canonical, instance_id, segment_id)
    segments = snapshot["geometry_by_instance"][instance_id]["segments"]
    for index, item in enumerate(segments):
        if item["segment_id"] == segment_id:
            segments[index] = replacement
            return
    segments.append(replacement)


def apply_operation(snapshot: dict, canonical: dict, operation: dict) -> None:
    parts = operation["target_path"].split("/")
    kind = operation["operation_kind"]
    if kind == "RESTORE_COMPONENT":
        _restore_component(snapshot, canonical, parts[1])
        return
    if kind == "RESTORE_SEAM":
        _restore_seam(snapshot, canonical, parts[1])
        return
    if kind == "RESTORE_SEAM_FIELD":
        _restore_seam_field(snapshot, canonical, parts[1], parts[3])
        return
    if kind == "RESTORE_NOTCH":
        _restore_notch(snapshot, canonical, parts[1], parts[3])
        return
    if kind == "RESTORE_SEGMENT":
        _restore_segment(snapshot, canonical, parts[1], parts[3])
        return
    raise ValueError(f"unsupported repair operation: {kind}")


def apply_operations(candidate: dict, canonical: dict, operations: list[dict], snapshot_id: str) -> dict:
    working = clone_snapshot(candidate, snapshot_id)
    for operation in operations:
        apply_operation(working, canonical, operation)
    return refresh_snapshot(working, canonical)
