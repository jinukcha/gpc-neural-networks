"""Immutable completion snapshots derived from CP2 pattern products."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .model import canonical_sha256


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(payload: dict, key: str) -> dict:
    result = deepcopy(payload)
    result.pop(key, None)
    result[key] = canonical_sha256(result)
    return result


def _component_order(package: dict) -> list[str]:
    return [item["instance_id"] for item in package["component_instances"]]


def _seam_order(package: dict) -> list[str]:
    return [item["interface_id"] for item in package["seams"]]


def load_cp2_snapshot(root: Path, snapshot_id: str = "R1C_CP2_CANONICAL") -> dict:
    build = root / "build/r1c_cp2"
    geometry = {
        path.stem: _read_json(path)
        for path in sorted((build / "geometry").glob("*.json"))
    }
    package = _read_json(build / "assembled_pattern_package.json")
    parameters = _read_json(root / "build/r1c_cp1/resolved_parameter_set.json")
    return make_snapshot(snapshot_id, geometry, package, parameters)


def make_snapshot(
    snapshot_id: str,
    geometry_by_instance: dict[str, dict],
    assembled_package: dict,
    resolved_parameter_set: dict,
) -> dict:
    payload = {
        "contract": "PatternCompletionSnapshot/1",
        "snapshot_id": snapshot_id,
        "geometry_by_instance": deepcopy(geometry_by_instance),
        "assembled_package": deepcopy(assembled_package),
        "resolved_parameter_set": deepcopy(resolved_parameter_set),
    }
    return refresh_snapshot(payload)


def clone_snapshot(snapshot: dict, snapshot_id: str) -> dict:
    result = deepcopy(snapshot)
    result["snapshot_id"] = snapshot_id
    return refresh_snapshot(result)


def _sort_owner_list(payload: dict, canonical_payload: dict, field: str, identity: str) -> None:
    rank = {
        item[identity]: index
        for index, item in enumerate(canonical_payload.get(field, []))
    }
    payload[field].sort(key=lambda item: rank.get(item[identity], 10_000))


def _normalise_geometry(snapshot: dict, canonical: dict | None) -> None:
    geometry = snapshot["geometry_by_instance"]
    if canonical is not None:
        for instance_id, payload in geometry.items():
            authority = canonical["geometry_by_instance"].get(instance_id)
            if authority is None:
                continue
            _sort_owner_list(payload, authority, "segments", "segment_id")
            _sort_owner_list(payload, authority, "boundaries", "boundary_id")
            _sort_owner_list(payload, authority, "notches", "notch_id")
    for instance_id, payload in list(geometry.items()):
        geometry[instance_id] = _rehash(payload, "geometry_sha256")
    if canonical is None:
        return
    ordered = {}
    for instance_id in canonical["geometry_by_instance"]:
        if instance_id in geometry:
            ordered[instance_id] = geometry[instance_id]
    for instance_id in sorted(set(geometry) - set(ordered)):
        ordered[instance_id] = geometry[instance_id]
    snapshot["geometry_by_instance"] = ordered


def _normalise_package(snapshot: dict, canonical: dict | None) -> None:
    package = snapshot["assembled_package"]
    geometry = snapshot["geometry_by_instance"]
    for item in package.get("component_instances", []):
        instance_id = item["instance_id"]
        if instance_id in geometry:
            item["geometry_sha256"] = geometry[instance_id]["geometry_sha256"]
    if canonical is not None:
        component_rank = {
            value: index for index, value in enumerate(_component_order(canonical["assembled_package"]))
        }
        seam_rank = {
            value: index for index, value in enumerate(_seam_order(canonical["assembled_package"]))
        }
        package["component_instances"].sort(
            key=lambda item: component_rank.get(item["instance_id"], 10_000)
        )
        package["seams"].sort(
            key=lambda item: seam_rank.get(item["interface_id"], 10_000)
        )
    package["component_instance_count"] = len(package.get("component_instances", []))
    package["seam_count"] = len(package.get("seams", []))
    package["notch_pair_count"] = sum(
        len(item.get("notch_correspondence", [])) for item in package.get("seams", [])
    )
    snapshot["assembled_package"] = _rehash(package, "assembled_package_sha256")


def _state_payload(snapshot: dict) -> dict:
    return {
        "geometry_by_instance": snapshot["geometry_by_instance"],
        "assembled_package": snapshot["assembled_package"],
        "resolved_parameter_set": snapshot["resolved_parameter_set"],
    }


def refresh_snapshot(snapshot: dict, canonical: dict | None = None) -> dict:
    result = deepcopy(snapshot)
    _normalise_geometry(result, canonical)
    _normalise_package(result, canonical)
    result["state_sha256"] = canonical_sha256(_state_payload(result))
    unhashed = deepcopy(result)
    unhashed.pop("snapshot_sha256", None)
    result["snapshot_sha256"] = canonical_sha256(unhashed)
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
