"""Validate actual Godot captures and assemble the CP6 evidence board."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np


CAPTURE_ORDER = (
    ("sleeved_tunic", "near"),
    ("sleeved_tunic", "mid"),
    ("sleeved_tunic", "far"),
    ("straight_robe", "near"),
    ("straight_robe", "mid"),
    ("straight_robe", "far"),
)


def validate_capture(path: Path) -> dict:
    with Image.open(path) as source:
        image = source.convert("RGB")
        values = np.asarray(image, dtype=np.float64)
        width, height = image.size
    luminance = np.mean(values, axis=2)
    variance = float(np.var(luminance))
    centre = luminance[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    centre_variance = float(np.var(centre))
    accepted = width >= 768 and height >= 768 and variance >= 15.0 and centre_variance >= 8.0
    return {
        "path": path.name,
        "width": width,
        "height": height,
        "luminance_variance": variance,
        "centre_luminance_variance": centre_variance,
        "accepted": accepted,
    }


def _labelled_tile(path: Path, label: str, tile_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((tile_size[0], tile_size[1] - 48), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", tile_size, (28, 28, 30))
    x = (tile_size[0] - image.width) // 2
    y = 42 + (tile_size[1] - 48 - image.height) // 2
    tile.paste(image, (x, y))
    draw = ImageDraw.Draw(tile)
    draw.text((14, 12), label, fill=(238, 238, 238))
    return tile


def build_contact_sheet(capture_root: Path, target: Path) -> dict:
    tile_size = (640, 640)
    sheet = Image.new("RGB", (tile_size[0] * 3, tile_size[1] * 2), (18, 18, 20))
    validations = []
    for index, (product, distance) in enumerate(CAPTURE_ORDER):
        path = capture_root / f"{product}_{distance}.png"
        validation = validate_capture(path)
        validations.append(validation)
        label = f"{product.replace('_', ' ').upper()} — {distance.upper()}"
        tile = _labelled_tile(path, label, tile_size)
        sheet.paste(tile, ((index % 3) * tile_size[0], (index // 3) * tile_size[1]))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return {
        "contract": "MultiDistanceCaptureBoard/1",
        "capture_count": len(validations),
        "capture_root": capture_root.name,
        "captures": validations,
        "contact_sheet_path": target.name,
        "contact_sheet_width": sheet.width,
        "contact_sheet_height": sheet.height,
        "all_captures_accepted": all(item["accepted"] for item in validations),
    }
