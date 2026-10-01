#!/usr/bin/env python3
"""Fresh-process verification for CP6 GLB products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.export.glb import verify_glb
from wuxia_garment_oss.garments.trousers.fit import POSE_IDS


TUNIC_POSES = (
    "NEUTRAL_A", "ARMS_FORWARD", "ARMS_OVERHEAD", "CROSS_BODY_REACH",
    "DEEP_ELBOW_BEND", "TORSO_TWIST", "FORWARD_BEND", "SEATED", "SQUAT", "WALK_STRIDE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    root = parse_args().root.resolve()
    directory = root / "build/garment_cad_pro_cp6/game_products"
    tunic = verify_glb(directory / "sleeveless_tunic.glb", TUNIC_POSES)
    trousers = verify_glb(directory / "trousers.glb", POSE_IDS)
    write_json(directory / "sleeveless_tunic_fresh_reopen.json", tunic)
    write_json(directory / "trousers_fresh_reopen.json", trousers)
    result = {
        "contract": "CP6GLBFreshProcessReceipt/1",
        "tunic": tunic,
        "trousers": trousers,
        "fresh_process_pass": tunic["fresh_reopen_pass"] and trousers["fresh_reopen_pass"],
    }
    write_json(directory / "fresh_process_receipt.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["fresh_process_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
