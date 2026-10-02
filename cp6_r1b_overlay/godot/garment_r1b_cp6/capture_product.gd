extends SceneTree

const SecondaryController = preload("res://secondary_motion_controller.gd")

var _scene_root: Node3D
var _product: Node3D
var _camera: Camera3D


func _initialize() -> void:
    call_deferred("_run")


func _arguments() -> Dictionary:
    var result: Dictionary = {}
    var values: PackedStringArray = OS.get_cmdline_user_args()
    var index: int = 0
    while index < values.size():
        var key: String = values[index]
        if key.begins_with("--") and index + 1 < values.size():
            result[key.substr(2)] = values[index + 1]
            index += 2
        else:
            index += 1
    return result


func _load_json(path: String) -> Dictionary:
    var value: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
    return value if value is Dictionary else {}


func _collect_meshes(node: Node, result: Array[MeshInstance3D]) -> void:
    if node is MeshInstance3D:
        result.append(node as MeshInstance3D)
    for child: Node in node.get_children():
        _collect_meshes(child, result)


func _mesh_instances(root: Node) -> Array[MeshInstance3D]:
    var result: Array[MeshInstance3D] = []
    _collect_meshes(root, result)
    return result


func _neutralize(root: Node) -> void:
    var material := StandardMaterial3D.new()
    material.albedo_color = Color(0.64, 0.65, 0.67, 1.0)
    material.metallic = 0.0
    material.roughness = 0.78
    for mesh_instance: MeshInstance3D in _mesh_instances(root):
        for surface_index: int in range(mesh_instance.mesh.get_surface_count()):
            mesh_instance.set_surface_override_material(surface_index, material)


func _expand_bounds(mesh_instance: MeshInstance3D, state: Dictionary) -> void:
    var local: AABB = mesh_instance.mesh.get_aabb()
    for x_index: int in range(2):
        for y_index: int in range(2):
            for z_index: int in range(2):
                var local_point := local.position + Vector3(
                    local.size.x * float(x_index),
                    local.size.y * float(y_index),
                    local.size.z * float(z_index),
                )
                var point: Vector3 = mesh_instance.global_transform * local_point
                if not bool(state["initialized"]):
                    state["minimum"] = point
                    state["maximum"] = point
                    state["initialized"] = true
                else:
                    var minimum: Vector3 = state["minimum"]
                    var maximum: Vector3 = state["maximum"]
                    state["minimum"] = minimum.min(point)
                    state["maximum"] = maximum.max(point)


func _bounds(root: Node) -> AABB:
    var state: Dictionary = {
        "initialized": false,
        "minimum": Vector3.ZERO,
        "maximum": Vector3.ZERO,
    }
    for mesh_instance: MeshInstance3D in _mesh_instances(root):
        _expand_bounds(mesh_instance, state)
    var minimum: Vector3 = state["minimum"]
    var maximum: Vector3 = state["maximum"]
    return AABB(minimum, maximum - minimum)


func _environment() -> void:
    var world_environment := WorldEnvironment.new()
    var environment := Environment.new()
    environment.background_mode = Environment.BG_COLOR
    environment.background_color = Color(0.075, 0.085, 0.105, 1.0)
    environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    environment.ambient_light_color = Color(0.82, 0.86, 0.94, 1.0)
    environment.ambient_light_energy = 0.78
    environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
    world_environment.environment = environment
    _scene_root.add_child(world_environment)
    var key := DirectionalLight3D.new()
    key.light_color = Color(1.0, 0.93, 0.84, 1.0)
    key.light_energy = 2.25
    key.rotation_degrees = Vector3(-38.0, -32.0, 0.0)
    key.shadow_enabled = true
    _scene_root.add_child(key)
    var fill := DirectionalLight3D.new()
    fill.light_color = Color(0.68, 0.78, 1.0, 1.0)
    fill.light_energy = 1.2
    fill.rotation_degrees = Vector3(-18.0, 148.0, 0.0)
    _scene_root.add_child(fill)


func _floor(bounds: AABB) -> void:
    var floor := MeshInstance3D.new()
    var plane := PlaneMesh.new()
    plane.size = Vector2(50.0, 50.0)
    floor.mesh = plane
    floor.position = Vector3(bounds.get_center().x, bounds.position.y - 0.015, bounds.get_center().z)
    var material := StandardMaterial3D.new()
    material.albedo_color = Color(0.19, 0.205, 0.23, 1.0)
    material.roughness = 0.94
    floor.material_override = material
    _scene_root.add_child(floor)


func _apply_secondary(profile: Dictionary) -> Dictionary:
    var controller = SecondaryController.new()
    controller.configure(profile)
    var values: Dictionary = {}
    for frame: int in range(180):
        var drivers: Dictionary = {}
        var phase: float = float(frame) * 0.115
        var domain_index: int = 0
        for domain_value: Variant in profile.get("domains", []):
            var domain: Dictionary = domain_value
            drivers[str(domain["domain_id"])] = sin(phase + float(domain_index) * 1.8)
            domain_index += 1
        values = controller.step(1.0 / 60.0, drivers)
    controller.apply(_product, values)
    return values


func _camera_setup(bounds: AABB, distance: float) -> void:
    _camera = Camera3D.new()
    _camera.fov = 35.0
    _camera.near = 0.03
    _camera.far = 100.0
    _scene_root.add_child(_camera)
    var centre: Vector3 = bounds.get_center()
    var position := centre + Vector3(0.0, 0.03 * bounds.size.y, distance)
    var target := centre + Vector3(0.0, 0.04 * bounds.size.y, 0.0)
    _camera.look_at_from_position(position, target, Vector3.UP)
    _camera.current = true


func _overlay(label_text: String) -> void:
    var layer := CanvasLayer.new()
    var panel := ColorRect.new()
    panel.position = Vector2(20.0, 20.0)
    panel.size = Vector2(560.0, 54.0)
    panel.color = Color(0.02, 0.025, 0.035, 0.84)
    layer.add_child(panel)
    var label := Label.new()
    label.position = Vector2(34.0, 32.0)
    label.text = label_text
    label.add_theme_font_size_override("font_size", 24)
    layer.add_child(label)
    get_root().add_child(layer)


func _save_capture(output: String) -> int:
    for _frame: int in range(5):
        await process_frame
    await RenderingServer.frame_post_draw
    var image: Image = get_root().get_texture().get_image()
    if image.is_empty():
        return ERR_CANT_CREATE
    return image.save_png(output)


func _run() -> void:
    var args: Dictionary = _arguments()
    var product_path: String = str(args.get("product", ""))
    var profile_path: String = str(args.get("profile", ""))
    var output: String = str(args.get("output", "capture.png"))
    var distance: float = float(args.get("distance", "2.4"))
    var label_text: String = str(args.get("label", "GARMENT CP6"))
    var resource: Resource = ResourceLoader.load(product_path)
    if resource == null or not (resource is PackedScene):
        push_error("unable to load capture product: " + product_path)
        quit(2)
        return
    DisplayServer.window_set_size(Vector2i(1024, 1024))
    _scene_root = Node3D.new()
    get_root().add_child(_scene_root)
    _product = (resource as PackedScene).instantiate() as Node3D
    _scene_root.add_child(_product)
    _neutralize(_product)
    _apply_secondary(_load_json(profile_path))
    var product_bounds: AABB = _bounds(_product)
    _environment()
    _floor(product_bounds)
    _camera_setup(product_bounds, distance)
    _overlay(label_text)
    var error: int = await _save_capture(output)
    print("CP6_CAPTURE_PATH=" + output)
    quit(0 if error == OK else 3)
