extends SceneTree

func _collect_meshes(node: Node, rows: Array) -> void:
    if node is MeshInstance3D:
        var mesh: Mesh = node.mesh
        if mesh != null:
            var vertex_count := 0
            var triangle_count := 0
            for surface_index in range(mesh.get_surface_count()):
                var arrays: Array = mesh.surface_get_arrays(surface_index)
                var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
                var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
                vertex_count += vertices.size()
                triangle_count += indices.size() / 3
            rows.append({
                "node": str(node.name),
                "surface_count": mesh.get_surface_count(),
                "blend_shape_count": mesh.get_blend_shape_count(),
                "vertex_count": vertex_count,
                "triangle_count": triangle_count,
            })
    for child in node.get_children():
        _collect_meshes(child, rows)

func _inspect(path: String) -> Dictionary:
    var resource: Resource = load(path)
    if resource == null or not resource is PackedScene:
        return {"loaded": false, "path": path, "meshes": []}
    var scene: PackedScene = resource as PackedScene
    var root: Node = scene.instantiate()
    var rows: Array = []
    _collect_meshes(root, rows)
    var surface_count := 0
    var vertex_count := 0
    var triangle_count := 0
    var max_blend_shape_count := 0
    for row in rows:
        surface_count += int(row["surface_count"])
        vertex_count += int(row["vertex_count"])
        triangle_count += int(row["triangle_count"])
        max_blend_shape_count = max(max_blend_shape_count, int(row["blend_shape_count"]))
    root.free()
    return {
        "loaded": true,
        "path": path,
        "mesh_instance_count": rows.size(),
        "surface_count": surface_count,
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
        "max_blend_shape_count": max_blend_shape_count,
        "meshes": rows,
    }

func _matches(actual: Dictionary, expected: Dictionary) -> bool:
    return bool(actual.get("loaded", false)) \
        and int(actual.get("surface_count", 0)) == int(expected["primitive_count"]) \
        and int(actual.get("max_blend_shape_count", -1)) == int(expected["blend_shape_count"]) \
        and int(actual.get("vertex_count", 0)) > 0 \
        and int(actual.get("triangle_count", 0)) > 0

func _init() -> void:
    var expected_text := FileAccess.get_file_as_string("res://expected.json")
    var expected: Variant = JSON.parse_string(expected_text)
    if expected == null:
        push_error("failed to parse expected.json")
        quit(2)
        return
    var tunic := _inspect(str(expected["tunic"]["path"]))
    var trousers := _inspect(str(expected["trousers"]["path"]))
    var tunic_pass := _matches(tunic, expected["tunic"])
    var trousers_pass := _matches(trousers, expected["trousers"])
    var payload := {
        "contract": "GodotGarmentConsumerReceipt/1",
        "godot_version": Engine.get_version_info(),
        "read_only_consumer": true,
        "tunic": tunic,
        "trousers": trousers,
        "tunic_pass": tunic_pass,
        "trousers_pass": trousers_pass,
        "consumer_pass": tunic_pass and trousers_pass,
    }
    print("CP6_GODOT_RECEIPT=" + JSON.stringify(payload))
    quit(0 if payload["consumer_pass"] else 1)
