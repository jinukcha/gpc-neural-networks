extends SceneTree

const MARKER: String = "CP5_GODOT_SLEEVE_RECEIPT="

func _initialize() -> void:
    call_deferred("_run")

func _empty_stats() -> Dictionary:
    return {
        "mesh_instance_count": 0,
        "skeleton_count": 0,
        "surface_count": 0,
        "vertex_count": 0,
        "triangle_count": 0,
        "max_blend_shape_count": 0,
        "cp5_component_mesh_count": 0,
    }

func _accumulate(node: Node, stats: Dictionary) -> void:
    if node is Skeleton3D:
        stats["skeleton_count"] = int(stats["skeleton_count"]) + 1
    if node is MeshInstance3D and node.mesh != null:
        var mesh: Mesh = node.mesh
        stats["mesh_instance_count"] = int(stats["mesh_instance_count"]) + 1
        stats["surface_count"] = int(stats["surface_count"]) + mesh.get_surface_count()
        stats["max_blend_shape_count"] = max(int(stats["max_blend_shape_count"]), mesh.get_blend_shape_count())
        if "CP5_COMPONENT" in str(node.name):
            stats["cp5_component_mesh_count"] = int(stats["cp5_component_mesh_count"]) + 1
        for surface: int in range(mesh.get_surface_count()):
            var arrays: Array = mesh.surface_get_arrays(surface)
            var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
            var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
            stats["vertex_count"] = int(stats["vertex_count"]) + vertices.size()
            var triangles: int = indices.size() / 3 if indices.size() > 0 else vertices.size() / 3
            stats["triangle_count"] = int(stats["triangle_count"]) + triangles
    for child: Node in node.get_children():
        _accumulate(child, stats)

func _inspect_product(path: String) -> Dictionary:
    var resource: Resource = ResourceLoader.load(path)
    if resource == null or not (resource is PackedScene):
        return {"product_pass": false, "reason": "LOAD_FAILED"}
    var instance: Node = (resource as PackedScene).instantiate()
    var stats: Dictionary = _empty_stats()
    _accumulate(instance, stats)
    instance.queue_free()
    var product_pass: bool = (
        int(stats["mesh_instance_count"]) >= 2
        and int(stats["skeleton_count"]) >= 1
        and int(stats["cp5_component_mesh_count"]) >= 1
        and int(stats["max_blend_shape_count"]) >= 3
        and int(stats["vertex_count"]) > 29500
        and int(stats["triangle_count"]) > 56000
    )
    stats["product_pass"] = product_pass
    return stats

func _run() -> void:
    var sleeved_tunic: Dictionary = _inspect_product("res://products/sleeved_tunic_rigged.glb")
    var straight_robe: Dictionary = _inspect_product("res://products/straight_sleeve_robe_rigged.glb")
    var consumer_pass: bool = bool(sleeved_tunic.get("product_pass", false)) and bool(straight_robe.get("product_pass", false))
    var receipt: Dictionary = {
        "contract": "GodotSleeveFamilyRuntimeReceipt/1",
        "godot_version": Engine.get_version_info(),
        "read_only_consumer": true,
        "sleeved_tunic": sleeved_tunic,
        "straight_robe": straight_robe,
        "consumer_pass": consumer_pass,
        "secondary_motion_executed": false,
        "rig_aware_lod_executed": false,
        "mesh_scaling": "FORBIDDEN",
    }
    print(MARKER + JSON.stringify(receipt))
    quit(0 if consumer_pass else 1)
