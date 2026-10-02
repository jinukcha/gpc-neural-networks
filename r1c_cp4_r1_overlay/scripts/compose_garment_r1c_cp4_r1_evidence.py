#!/usr/bin/env python3
"""Compose CP4-R1 evidence board and validate rendered view coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ORDER = [
    "front", "left", "right", "back",
    "front_three_quarter", "back_three_quarter", "garment_only", "shoulder_cap_detail",
    "underarm_detail", "neck_collar_detail", "wireframe_front", "seam_notch_overlay",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def view_metrics(path: Path) -> dict:
    with Image.open(path) as source:
        image = source.convert("RGB")
        array = np.asarray(image, dtype=np.float64)
    gray = array.mean(axis=2)
    border = np.concatenate((gray[:80].reshape(-1), gray[-80:].reshape(-1), gray[:, :80].reshape(-1), gray[:, -80:].reshape(-1)))
    background = float(np.median(border))
    foreground = np.abs(gray - background) > 12.0
    centre = foreground[512:1536, 512:1536]
    return {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "luminance_variance": float(np.var(gray)),
        "foreground_fraction": float(np.mean(foreground)),
        "centre_foreground_fraction": float(np.mean(centre)),
        "pass": bool(
            array.shape[0] >= 2048
            and array.shape[1] >= 2048
            and np.var(gray) > 35.0
            and np.mean(foreground) > 0.025
            and np.mean(centre) > 0.020
        ),
    }


def compose(build: Path) -> tuple[Path, dict]:
    visual = build / "visual"
    tile_size = 1024
    header = 48
    canvas = Image.new("RGB", (tile_size * 4, (tile_size + header) * 3), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    metrics = {}
    for index, view_id in enumerate(ORDER):
        path = visual / f"{view_id}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics[view_id] = view_metrics(path)
        with Image.open(path) as source:
            tile = source.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        x = (index % 4) * tile_size
        y = (index // 4) * (tile_size + header)
        canvas.paste(tile, (x, y + header))
        draw.rectangle((x, y, x + tile_size, y + header), fill=(12, 12, 15))
        draw.text((x + 14, y + 13), view_id.upper(), fill=(235, 235, 235), font=font)
    target = build / "cp4_r1_body_visible_contact_sheet.png"
    canvas.save(target)
    receipt = {
        "contract": "CP4R1VisualEvidenceReceipt/1",
        "required_views": ORDER,
        "view_count": len(ORDER),
        "views": metrics,
        "all_view_metrics_pass": all(item["pass"] for item in metrics.values()),
        "contact_sheet": {
            "path": target.relative_to(build).as_posix(),
            "width": canvas.width,
            "height": canvas.height,
        },
        "visual_review": "PASS" if all(item["pass"] for item in metrics.values()) else "FAIL",
    }
    return target, receipt


def main():
    root = parse_args().root.resolve()
    build = root / "build/r1c_cp4_r1"
    _, receipt = compose(build)
    (build / "visual_evidence_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
