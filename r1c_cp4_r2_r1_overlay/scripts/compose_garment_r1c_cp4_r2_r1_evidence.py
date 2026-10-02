#!/usr/bin/env python3
"""Compose identical-camera before/after boards and image continuity metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


VIEWS = (
    "front", "left", "right", "back",
    "front_three_quarter", "back_three_quarter", "garment_only",
    "shoulder_cap_detail", "underarm_detail", "neck_collar_detail",
    "wireframe_front", "seam_notch_overlay",
)


def _arguments() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args().root.resolve()


def _mask(image: Image.Image) -> np.ndarray:
    values = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate((values[:8].reshape(-1, 3), values[-8:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(values - background, axis=2)
    return distance > 18.0


def _metrics(before: Image.Image, after: Image.Image) -> dict:
    left = np.asarray(before.convert("RGB"), dtype=np.float32)
    right = np.asarray(after.convert("RGB"), dtype=np.float32)
    before_mask, after_mask = _mask(before), _mask(after)
    union = np.count_nonzero(before_mask | after_mask)
    intersection = np.count_nonzero(before_mask & after_mask)
    return {
        "silhouette_iou": float(intersection / max(union, 1)),
        "mean_absolute_rgb_delta": float(np.mean(np.abs(left - right))),
        "foreground_fraction_before": float(np.mean(before_mask)),
        "foreground_fraction_after": float(np.mean(after_mask)),
    }


def _pair_image(before: Image.Image, after: Image.Image, view_id: str) -> Image.Image:
    width, height = before.size
    canvas = Image.new("RGB", (width * 2, height + 72), (24, 24, 28))
    canvas.paste(before, (0, 72))
    canvas.paste(after, (width, 72))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 22), f"{view_id} — CP4-R1 BEFORE", fill=(235, 235, 235))
    draw.text((width + 24, 22), f"{view_id} — CP4-R2-R1 AFTER", fill=(235, 235, 235))
    return canvas


def _contact_sheet(pairs: list[tuple[str, Image.Image]]) -> Image.Image:
    cell_width, cell_height = 1024, 548
    sheet = Image.new("RGB", (cell_width * 2, cell_height * 6), (20, 20, 24))
    for index, (name, image) in enumerate(pairs):
        thumbnail = image.copy()
        thumbnail.thumbnail((cell_width, cell_height - 28), Image.Resampling.LANCZOS)
        x = (index % 2) * cell_width
        y = (index // 2) * cell_height
        sheet.paste(thumbnail, (x, y + 28))
        ImageDraw.Draw(sheet).text((x + 12, y + 7), name, fill=(245, 245, 245))
    return sheet


def main() -> int:
    root = _arguments()
    before_dir = root / "build/r1c_cp4_r1/visual"
    after_dir = root / "build/r1c_cp4_r2_r1/visual/after"
    output = root / "build/r1c_cp4_r2_r1/visual/comparison"
    output.mkdir(parents=True, exist_ok=True)
    metrics, pairs = {}, []
    for view_id in VIEWS:
        before = Image.open(before_dir / f"{view_id}.png").convert("RGB")
        after = Image.open(after_dir / f"{view_id}.png").convert("RGB")
        if before.size != after.size:
            raise ValueError(f"camera output size mismatch: {view_id}")
        metrics[view_id] = _metrics(before, after)
        pair = _pair_image(before, after, view_id)
        pair.save(output / f"{view_id}_before_after.png")
        pairs.append((view_id, pair))
    sheet = _contact_sheet(pairs)
    sheet_path = root / "build/r1c_cp4_r2_r1/cp4_r2_r1_before_after_contact_sheet.png"
    sheet.save(sheet_path)
    receipt = {
        "contract": "CP4R2R1BeforeAfterEvidenceReceipt/1",
        "same_camera_definition": True,
        "view_count": len(VIEWS),
        "views": metrics,
        "minimum_silhouette_iou": min(item["silhouette_iou"] for item in metrics.values()),
        "contact_sheet": {
            "path": sheet_path.relative_to(root).as_posix(),
            "width": sheet.width,
            "height": sheet.height,
        },
        "direct_art_audit": "PENDING_MODEL_REVIEW",
    }
    target = root / "build/r1c_cp4_r2_r1/before_after_evidence_receipt.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
