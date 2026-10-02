#!/usr/bin/env python3
"""Compose same-camera Godot before/after evidence and reviewability metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VIEW_IDS = (
    "front", "left", "right", "back", "front_three_quarter", "back_three_quarter",
    "garment_only", "shoulder_cap_detail", "underarm_detail", "neck_collar_detail",
    "wireframe_front", "seam_notch_overlay",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--godot-receipt", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def image_metrics(path: Path) -> dict:
    image = Image.open(path).convert("RGB")
    values = np.asarray(image, dtype=np.float32)
    luminance = values.mean(axis=2)
    background = np.median(values[:96, :96], axis=(0, 1))
    distance = np.linalg.norm(values - background[None, None, :], axis=2)
    foreground = distance > 18.0
    h, w = foreground.shape
    centre = foreground[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4]
    return {
        "width": w,
        "height": h,
        "foreground_fraction": float(foreground.mean()),
        "centre_foreground_fraction": float(centre.mean()),
        "luminance_variance": float(luminance.var()),
        "reviewable": bool(w == 2048 and h == 2048 and foreground.mean() > 0.02 and luminance.var() > 25.0),
    }


def silhouette_iou(before: Path, after: Path) -> float:
    first = _mask(before)
    second = _mask(after)
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / max(union, 1))


def _mask(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    background = np.median(values[:96, :96], axis=(0, 1))
    return np.linalg.norm(values - background[None, None, :], axis=2) > 18.0


def compose_sheet(root: Path) -> Path:
    capture_root = root / "build/r1c_cp4_r2_rev1/godot_product/captures"
    tile_size = 512
    label_height = 30
    sheet = Image.new("RGB", (6 * tile_size, 4 * (tile_size + label_height)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for state_row, state_id in enumerate(("before", "after")):
        for index, view_id in enumerate(VIEW_IDS):
            column = index % 6
            local_row = index // 6
            row = state_row * 2 + local_row
            image = Image.open(capture_root / state_id / f"{view_id}.png").convert("RGB")
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            x, y = column * tile_size, row * (tile_size + label_height)
            sheet.paste(image, (x, y + label_height))
            draw.text((x + 8, y + 8), f"{state_id.upper()} · {view_id}", fill=(235, 235, 235), font=font)
    output = root / "build/r1c_cp4_r2_rev1/cp4_r2_before_after_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def build_receipt(root: Path, godot: dict, contact_sheet: Path) -> dict:
    capture_root = root / "build/r1c_cp4_r2_rev1/godot_product/captures"
    states = {}
    for state_id in ("before", "after"):
        states[state_id] = {
            view_id: image_metrics(capture_root / state_id / f"{view_id}.png")
            for view_id in VIEW_IDS
        }
    comparisons = {
        view_id: {"silhouette_iou": silhouette_iou(
            capture_root / "before" / f"{view_id}.png",
            capture_root / "after" / f"{view_id}.png",
        )}
        for view_id in VIEW_IDS[:6]
    }
    all_reviewable = all(item["reviewable"] for state in states.values() for item in state.values())
    return {
        "contract": "GodotSameCameraVisualEvidenceReceipt/1",
        "godot_consumer_pass": bool(godot["consumer_pass"]),
        "same_camera_fixture": True,
        "required_view_count_per_state": len(VIEW_IDS),
        "states": states,
        "primary_view_comparisons": comparisons,
        "all_views_reviewable": all_reviewable,
        "contact_sheet": {
            "path": contact_sheet.relative_to(root).as_posix(),
            "width": 3072,
            "height": 2168,
        },
        "direct_art_review": "PENDING_REVIEWER_DISPOSITION",
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    godot = load_json(args.godot_receipt)
    contact_sheet = compose_sheet(root)
    receipt = build_receipt(root, godot, contact_sheet)
    build = root / "build/r1c_cp4_r2_rev1"
    (build / "godot_visual_evidence_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (build / "godot_consumer_receipt.json").write_text(json.dumps(godot, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
