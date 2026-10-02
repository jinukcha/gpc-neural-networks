#!/usr/bin/env python3
"""Blender 5.2.2 neutral-gray rendering and GLB export for CP4-R1."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def arguments() -> tuple[Path, Path]:
    values = sys.argv[sys.argv.index("--") + 1 :]
    root = Path(values[values.index("--root") + 1]).resolve()
    build = root / "build/r1c_cp4_r1"
    return root, build


def load_product(build: Path) -> dict:
    path = build / "product/materialized_product.json"
    return json.loads(path.read_text(encoding="utf-8"))


def transform(point):
    return (float(point[0]), float(point[2]), float(point[1]))


def material(name: str, value: float, roughness: float = 0.8):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (value, value, value, 1.0)
    result.roughness = roughness
    return result


def create_mesh(name: str, positions, triangles, assigned_material):
    mesh = bpy.data.meshes.new(f"{name}_MESH")
    mesh.from_pydata([transform(item) for item in positions], [], triangles)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(assigned_material)
    return obj


def create_seam(name: str, positions, assigned_material):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.0014
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(positions) - 1)
    for point, source in zip(spline.points, positions, strict=True):
        x, y, z = transform(source)
        point.co = (x, y, z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(assigned_material)
    return obj


def configure_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.curvature_ridge_factor = 1.6
    scene.display.shading.curvature_valley_factor = 1.2
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.035, 0.045)
    return scene


def create_camera(scene):
    camera_data = bpy.data.cameras.new("CP4_R1_CAMERA")
    camera_data.lens = 58.0
    camera = bpy.data.objects.new("CP4_R1_CAMERA", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    return camera


def aim(camera, location, target, lens=58.0):
    camera.location = location
    camera.data.lens = lens
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_objects(product):
    materials = {
        "body": material("BODY_NEUTRAL_DARK", 0.23),
        "bodice": material("GARMENT_BODICE", 0.76),
        "sleeve": material("GARMENT_SLEEVE", 0.70),
        "collar": material("GARMENT_COLLAR", 0.84),
        "cuff": material("GARMENT_CUFF", 0.80),
        "seam": material("SEAM_DIAGNOSTIC", 0.05),
    }
    body = create_mesh("REFERENCE_BODY", product["body"]["positions_m"], product["body"]["triangles"], materials["body"])
    garments = []
    for component in product["components"]:
        instance = component["instance_id"]
        owner = "bodice" if instance.startswith("bodice") else "sleeve" if instance.startswith("sleeve") else "collar" if instance == "collar" else "cuff"
        garments.append(create_mesh(instance.upper(), component["positions_m"], component["triangles"], materials[owner]))
    seams = [create_seam(f"SEAM_{item['interface_id']}", item["positions_m"], materials["seam"]) for item in product["seam_lines"]]
    return body, garments, seams


def visibility(body, garments, seams, body_visible=True, seam_visible=False, wire=False):
    body.hide_render = not body_visible
    for obj in garments:
        obj.hide_render = False
        obj.show_wire = wire
        obj.show_all_edges = wire
    for obj in seams:
        obj.hide_render = not seam_visible


def render_views(scene, camera, body, garments, seams, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    views = [
        ("front", (0.0, 3.5, 1.20), (0.0, 0.0, 1.15), 62.0, True, False, False),
        ("left", (-3.4, 0.0, 1.20), (0.0, 0.0, 1.15), 62.0, True, False, False),
        ("right", (3.4, 0.0, 1.20), (0.0, 0.0, 1.15), 62.0, True, False, False),
        ("back", (0.0, -3.5, 1.20), (0.0, 0.0, 1.15), 62.0, True, False, False),
        ("front_three_quarter", (2.55, 2.55, 1.32), (0.0, 0.0, 1.17), 62.0, True, False, False),
        ("back_three_quarter", (-2.55, -2.55, 1.32), (0.0, 0.0, 1.17), 62.0, True, False, False),
        ("garment_only", (0.0, 3.5, 1.20), (0.0, 0.0, 1.15), 62.0, False, False, False),
        ("shoulder_cap_detail", (-0.78, 0.95, 1.43), (-0.23, 0.0, 1.36), 72.0, True, True, False),
        ("underarm_detail", (-0.90, 0.72, 1.25), (-0.26, 0.0, 1.23), 75.0, True, True, False),
        ("neck_collar_detail", (0.0, 1.05, 1.55), (0.0, 0.0, 1.47), 78.0, True, True, False),
        ("wireframe_front", (0.0, 3.5, 1.20), (0.0, 0.0, 1.15), 62.0, False, False, True),
        ("seam_notch_overlay", (0.0, 3.5, 1.20), (0.0, 0.0, 1.15), 62.0, False, True, False),
    ]
    rendered = []
    for name, location, target, lens, body_visible, seam_visible, wire in views:
        visibility(body, garments, seams, body_visible, seam_visible, wire)
        aim(camera, location, target, lens)
        path = output / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered.append({"view_id": name, "path": path.name, "width": 2048, "height": 2048})
    return rendered


def export_glb(build: Path, body, garments, seams):
    bpy.ops.object.select_all(action="DESELECT")
    body.hide_render = True
    body.select_set(False)
    for seam in seams:
        seam.select_set(False)
    for obj in garments:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = garments[0]
    target = build / "product/r1c_cp4_r1_sleeved_tunic.glb"
    bpy.ops.export_scene.gltf(filepath=str(target), export_format="GLB", use_selection=True, export_yup=True)
    return target


def main():
    _, build = arguments()
    product = load_product(build)
    scene = configure_scene()
    body, garments, seams = create_objects(product)
    camera = create_camera(scene)
    rendered = render_views(scene, camera, body, garments, seams, build / "visual")
    glb = export_glb(build, body, garments, seams)
    blend = build / "product/r1c_cp4_r1_sleeved_tunic.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    receipt = {
        "contract": "BlenderVisualProductReceipt/1",
        "blender_version": ".".join(str(value) for value in bpy.app.version),
        "render_engine": scene.render.engine,
        "neutral_gray": True,
        "body_visible": True,
        "views": rendered,
        "view_count": len(rendered),
        "glb_path": glb.relative_to(build).as_posix(),
        "blend_path": blend.relative_to(build).as_posix(),
    }
    (build / "blender_visual_product_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
