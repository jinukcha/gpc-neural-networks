"""Atomic calibration checkpoint and fresh-process resume admission."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ...pattern_cad.document.model import canonical_sha256
from .fit import CalibratedMaterialProfile


def _profile_record(profile: CalibratedMaterialProfile) -> dict:
    return {
        "material_id": profile.material_id,
        "measurement_set_sha256": profile.measurement_set_sha256,
        "sizing_profile": profile.sizing_profile_dict(),
        "warp_profile": profile.warp_profile_dict(),
        "calibration_receipt": profile.calibration_receipt_dict(),
    }


def write_checkpoint(
    path: Path,
    profiles: tuple[CalibratedMaterialProfile, ...],
    all_measurement_hashes: dict[str, str],
    phase: str,
) -> dict:
    payload = {
        "contract": "MaterialCalibrationCheckpoint/1",
        "phase": phase,
        "completed_material_ids": [item.material_id for item in profiles],
        "all_measurement_hashes": dict(sorted(all_measurement_hashes.items())),
        "profiles": [_profile_record(item) for item in profiles],
    }
    payload["checkpoint_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def load_checkpoint(path: Path, expected_measurement_hashes: dict[str, str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("checkpoint_sha256")
    actual = canonical_sha256(payload)
    payload["checkpoint_sha256"] = recorded
    if recorded != actual:
        raise ValueError("calibration checkpoint hash mismatch")
    if payload["all_measurement_hashes"] != dict(sorted(expected_measurement_hashes.items())):
        raise ValueError("calibration checkpoint measurement identity mismatch")
    if len(payload["completed_material_ids"]) != len(set(payload["completed_material_ids"])):
        raise ValueError("duplicate material in checkpoint")
    return payload


def profile_ids(payload: dict) -> tuple[str, ...]:
    return tuple(item["material_id"] for item in payload["profiles"])
