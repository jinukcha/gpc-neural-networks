#!/usr/bin/env python3
"""Render CP4-R2-R1 with the exact CP4-R1 camera definitions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import bpy


def _arguments() -> tuple[Path, Path]:
    values = sys.argv[sys.argv.index("--") + 1 :]
    root = Path(values[values.index("--root") + 1]).resolve()
    return root, root / "build/r1c_cp4_r2_r1"


def _load_base(root: Path):
    path = root / "scripts/blender_render_garment_r1c_cp4_r1.py"
    spec = importlib.util.spec_from_file_location("cp4_r1_blender", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_glb(build: Path, body, garments, seams) -> Path:
    bpy.ops.object.select_all(action="DESELECT")
    body.hide_render = True
    for seam in seams:
        seam.select_set(False)
    for obj in garments:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = garments[0]
    target = build / "product/r1c_cp4_r2_r1_sleeved_tunic.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(target), export_format="GLB", use_selection=True, export_yup=True
    )
    return target


def main() -> None:
    root, build = _arguments()
    base = _load_base(root)
    product = base.load_product(build)
    scene = base.configure_scene()
    body, garments, seams = base.create_objects(product)
    camera = base.create_camera(scene)
    rendered = base.render_views(
        scene, camera, body, garments, seams, build / "visual/after"
    )
    glb = _export_glb(build, body, garments, seams)
    blend = build / "product/r1c_cp4_r2_r1_sleeved_tunic.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    receipt = {
        "contract": "CP4R2R1BlenderVisualProductReceipt/1",
        "blender_version": ".".join(str(value) for value in bpy.app.version),
        "same_camera_source": "scripts/blender_render_garment_r1c_cp4_r1.py",
        "render_engine": scene.render.engine,
        "neutral_gray": True,
        "body_visible": True,
        "views": rendered,
        "view_count": len(rendered),
        "glb_path": glb.relative_to(build).as_posix(),
        "blend_path": blend.relative_to(build).as_posix(),
    }
    path = build / "blender_visual_product_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
