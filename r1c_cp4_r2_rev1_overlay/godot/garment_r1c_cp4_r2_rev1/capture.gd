extends SceneTree

const MARKER := "R1C_CP4_R2_GODOT_RECEIPT="
const VIEW_SIZE := 2048

var scene_root: Node3D
var camera: Camera3D
var active_product: Node
var wireframe_overlay: MeshInstance3D
var seam_overlay: MeshInstance3D
var garment_material: StandardMaterial3D
var body_material: StandardMaterial3D
var hidden_material: StandardMaterial3D
var line_material: StandardMaterial3D

func _initialize() -> void:
    call_deferred("_run")

func _run() -> void:
    _setup_scene()
    var products := {
        "before": "res://products/before_cp4_r1.glb",
        "after": "res://products/after_cp4_r2.glb",
    }
    var receipt := {
        "contract": "GodotGarmentVisualProduct/1",
        "godot_version": Engine.get_version_info(),
        "read_only_consumer": true,
        "states": {},
        "consumer_pass": true,
        "blender_used": false,
    }
    for state_id in ["before", "after"]:
        var state_receipt: Dictionary = await _capture_state(state_id, products[state_id])
        receipt["states"][state_id] = state_receipt
        receipt["consumer_pass"] = bool(receipt["consumer_pass"]) and bool(state_receipt["pass"])
    print(MARKER + JSON.stringify(receipt))
    quit(0 if receipt["consumer_pass"] else 1)

func _setup_scene() -> void:
    get_root().size = Vector2i(VIEW_SIZE, VIEW_SIZE)
    scene_root = Node3D.new()
    scene_root.name = "AuditScene"
    get_root().add_child(scene_root)
    camera = Camera3D.new()
    camera.projection = Camera3D.PROJECTION_ORTHOGONAL
    camera.current = true
    scene_root.add_child(camera)
    _add_light(Vector3(-0.35, -0.45, -1.0), 1.35)
    _add_light(Vector3(0.5, -0.2, 0.75), 0.75)
    var environment := WorldEnvironment.new()
    var resource := Environment.new()
    resource.background_mode = Environment.BG_COLOR
    resource.background_color = Color(0.07, 0.07, 0.07)
    resource.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    resource.ambient_light_color = Color(0.65, 0.65, 0.65)
    resource.ambient_light_energy = 0.55
    environment.environment = resource
    scene_root.add_child(environment)
    garment_material = _material(Color(0.62, 0.62, 0.62), false)
    body_material = _material(Color(0.29, 0.29, 0.29), false)
    hidden_material = _material(Color(0.0, 0.0, 0.0, 0.0), true)
    line_material = _material(Color(0.92, 0.92, 0.92), false)

func _add_light(direction: Vector3, energy: float) -> void:
    var light := DirectionalLight3D.new()
    light.rotation = Vector3(direction.x, direction.y, direction.z)
    light.light_energy = energy
    light.shadow_enabled = true
    scene_root.add_child(light)

func _material(color: Color, transparent: bool) -> StandardMaterial3D:
    var material := StandardMaterial3D.new()
    material.albedo_color = color
    material.metallic = 0.0
    material.roughness = 0.88
    material.cull_mode = BaseMaterial3D.CULL_DISABLED
    if transparent:
        material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
    return material

func _capture_state(state_id: String, path: String) -> Dictionary:
    _remove_active_product()
    var resource: Resource = ResourceLoader.load(path)
    if resource == null or not (resource is PackedScene):
        return {"pass": false, "reason": "LOAD_FAILED", "path": path}
    active_product = (resource as PackedScene).instantiate()
    active_product.name = state_id + "_product"
    scene_root.add_child(active_product)
    _apply_neutral_materials(active_product, state_id)
    await process_frame
    var stats := _mesh_stats(active_product)
    var captures := []
    for config in _view_configs():
        var capture: Dictionary = await _capture_view(state_id, config)
        captures.append(capture)
    var pass := captures.size() == 12
    for item in captures:
        pass = pass and bool(item["pass"])
    return {"pass": pass, "path": path, "mesh_stats": stats, "captures": captures}

func _remove_active_product() -> void:
    if wireframe_overlay != null:
        wireframe_overlay.queue_free()
        wireframe_overlay = null
    if seam_overlay != null:
        seam_overlay.queue_free()
        seam_overlay = null
    if active_product != null:
        active_product.queue_free()
        active_product = null
        await process_frame

func _capture_view(state_id: String, config: Dictionary) -> Dictionary:
    _clear_diagnostics()
    _set_body_visible(active_product, bool(config["body_visible"]), state_id)
    if config["diagnostic"] == "wireframe":
        wireframe_overlay = _build_wireframe(active_product, state_id)
        scene_root.add_child(wireframe_overlay)
    elif config["diagnostic"] == "seam":
        seam_overlay = _build_seam_overlay(state_id)
        scene_root.add_child(seam_overlay)
    camera.position = config["position"]
    camera.size = float(config["size"])
    camera.look_at(config["target"], Vector3.UP)
    await process_frame
    await process_frame
    await process_frame
    var directory := "res://captures/" + state_id
    DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(directory))
    var output := directory + "/" + String(config["id"]) + ".png"
    var image := get_root().get_texture().get_image()
    var error := image.save_png(ProjectSettings.globalize_path(output))
    return {
        "view_id": config["id"],
        "path": output,
        "width": image.get_width(),
        "height": image.get_height(),
        "pass": error == OK and image.get_width() == VIEW_SIZE and image.get_height() == VIEW_SIZE,
    }

func _clear_diagnostics() -> void:
    if wireframe_overlay != null:
        wireframe_overlay.queue_free()
        wireframe_overlay = null
    if seam_overlay != null:
        seam_overlay.queue_free()
        seam_overlay = null

func _view_configs() -> Array[Dictionary]:
    return [
        _view("front", Vector3(0.0, 1.08, 3.2), Vector3(0.0, 1.05, 0.0), 2.05, true, "none"),
        _view("left", Vector3(-3.2, 1.08, 0.0), Vector3(0.0, 1.05, 0.0), 2.05, true, "none"),
        _view("right", Vector3(3.2, 1.08, 0.0), Vector3(0.0, 1.05, 0.0), 2.05, true, "none"),
        _view("back", Vector3(0.0, 1.08, -3.2), Vector3(0.0, 1.05, 0.0), 2.05, true, "none"),
        _view("front_three_quarter", Vector3(2.35, 1.18, 2.35), Vector3(0.0, 1.05, 0.0), 2.05, true, "none"),
        _view("back_three_quarter", Vector3(-2.35, 1.18, -2.35), Vector3(0.0, 1.05, 0.0), 2.05, true, "none"),
        _view("garment_only", Vector3(0.0, 1.08, 3.2), Vector3(0.0, 1.05, 0.0), 2.05, false, "none"),
        _view("shoulder_cap_detail", Vector3(1.35, 1.42, 1.55), Vector3(0.36, 1.36, 0.0), 0.72, true, "none"),
        _view("underarm_detail", Vector3(1.45, 1.15, 1.55), Vector3(0.38, 1.13, 0.0), 0.72, true, "none"),
        _view("neck_collar_detail", Vector3(0.0, 1.62, 1.5), Vector3(0.0, 1.43, 0.0), 0.58, true, "none"),
        _view("wireframe_front", Vector3(0.0, 1.08, 3.2), Vector3(0.0, 1.05, 0.0), 2.05, false, "wireframe"),
        _view("seam_notch_overlay", Vector3(0.0, 1.08, 3.2), Vector3(0.0, 1.05, 0.0), 2.05, false, "seam"),
    ]

func _view(id: String, position: Vector3, target: Vector3, size: float, body_visible: bool, diagnostic: String) -> Dictionary:
    return {"id": id, "position": position, "target": target, "size": size, "body_visible": body_visible, "diagnostic": diagnostic}

func _apply_neutral_materials(node: Node, state_id: String) -> void:
    if node is MeshInstance3D and node.mesh != null:
        var mesh_instance := node as MeshInstance3D
        var surface_count := mesh_instance.mesh.get_surface_count()
        for surface in range(surface_count):
            var is_body := _surface_is_body(mesh_instance, surface, state_id)
            mesh_instance.set_surface_override_material(surface, body_material if is_body else garment_material)
    for child in node.get_children():
        _apply_neutral_materials(child, state_id)

func _set_body_visible(node: Node, visible: bool, state_id: String) -> void:
    if node is MeshInstance3D and node.mesh != null:
        var mesh_instance := node as MeshInstance3D
        for surface in range(mesh_instance.mesh.get_surface_count()):
            if _surface_is_body(mesh_instance, surface, state_id):
                mesh_instance.set_surface_override_material(surface, body_material if visible else hidden_material)
    for child in node.get_children():
        _set_body_visible(child, visible, state_id)

func _surface_is_body(mesh_instance: MeshInstance3D, surface: int, state_id: String) -> bool:
    var lowered := String(mesh_instance.name).to_lower()
    if "body" in lowered or "avatar" in lowered or "reference" in lowered:
        return true
    if state_id == "after" and mesh_instance.mesh.get_surface_count() >= 8:
        return surface == mesh_instance.mesh.get_surface_count() - 1
    return false

func _mesh_stats(node: Node) -> Dictionary:
    var stats := {"mesh_instance_count": 0, "surface_count": 0, "vertex_count": 0, "triangle_count": 0}
    _accumulate_stats(node, stats)
    return stats

func _accumulate_stats(node: Node, stats: Dictionary) -> void:
    if node is MeshInstance3D and node.mesh != null:
        var mesh_instance := node as MeshInstance3D
        stats["mesh_instance_count"] = int(stats["mesh_instance_count"]) + 1
        stats["surface_count"] = int(stats["surface_count"]) + mesh_instance.mesh.get_surface_count()
        for surface in range(mesh_instance.mesh.get_surface_count()):
            var arrays := mesh_instance.mesh.surface_get_arrays(surface)
            var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
            var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
            stats["vertex_count"] = int(stats["vertex_count"]) + vertices.size()
            stats["triangle_count"] = int(stats["triangle_count"]) + (indices.size() / 3 if indices.size() > 0 else vertices.size() / 3)
    for child in node.get_children():
        _accumulate_stats(child, stats)

func _build_wireframe(product: Node, state_id: String) -> MeshInstance3D:
    var immediate := ImmediateMesh.new()
    immediate.surface_begin(Mesh.PRIMITIVE_LINES, line_material)
    _append_wireframe(product, immediate, state_id)
    immediate.surface_end()
    var instance := MeshInstance3D.new()
    instance.name = "WireframeOverlay"
    instance.mesh = immediate
    return instance

func _append_wireframe(node: Node, immediate: ImmediateMesh, state_id: String) -> void:
    if node is MeshInstance3D and node.mesh != null:
        var mesh_instance := node as MeshInstance3D
        for surface in range(mesh_instance.mesh.get_surface_count()):
            if _surface_is_body(mesh_instance, surface, state_id):
                continue
            var arrays := mesh_instance.mesh.surface_get_arrays(surface)
            var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
            var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
            for triangle in range(indices.size() / 3):
                var a := vertices[indices[triangle * 3]]
                var b := vertices[indices[triangle * 3 + 1]]
                var c := vertices[indices[triangle * 3 + 2]]
                _line(immediate, a, b)
                _line(immediate, b, c)
                _line(immediate, c, a)
    for child in node.get_children():
        _append_wireframe(child, immediate, state_id)

func _build_seam_overlay(state_id: String) -> MeshInstance3D:
    var immediate := ImmediateMesh.new()
    immediate.surface_begin(Mesh.PRIMITIVE_LINES, line_material)
    var path := "res://data/" + state_id + "_seam_lines.json"
    var file := FileAccess.open(path, FileAccess.READ)
    if file != null:
        var payload: Variant = JSON.parse_string(file.get_as_text())
        if payload is Dictionary:
            for seam in payload.get("seam_lines", []):
                var points: Array = seam.get("positions_m", [])
                for index in range(points.size() - 1):
                    _line(immediate, _vec3(points[index]), _vec3(points[index + 1]))
    immediate.surface_end()
    var instance := MeshInstance3D.new()
    instance.name = "SeamOverlay"
    instance.mesh = immediate
    return instance

func _line(immediate: ImmediateMesh, first: Vector3, second: Vector3) -> void:
    immediate.surface_add_vertex(first)
    immediate.surface_add_vertex(second)

func _vec3(value: Array) -> Vector3:
    return Vector3(float(value[0]), float(value[1]), float(value[2]))
