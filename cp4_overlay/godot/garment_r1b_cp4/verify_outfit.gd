extends SceneTree

const MARKER := "CP4_GODOT_OUTFIT_RECEIPT="

func _initialize() -> void:
    call_deferred("_run")

func _read_json(path: String) -> Dictionary:
    var file := FileAccess.open(path, FileAccess.READ)
    if file == null:
        return {}
    var parsed = JSON.parse_string(file.get_as_text())
    return parsed if parsed is Dictionary else {}

func _mesh_stats(node: Node) -> Dictionary:
    var result := {
        "mesh_instance_count": 0,
        "surface_count": 0,
        "vertex_count": 0,
        "triangle_count": 0,
        "max_blend_shape_count": 0,
    }
    _accumulate_mesh_stats(node, result)
    return result

func _accumulate_mesh_stats(node: Node, result: Dictionary) -> void:
    if node is MeshInstance3D and node.mesh != null:
        var mesh := node.mesh
        result.mesh_instance_count += 1
        result.surface_count += mesh.get_surface_count()
        result.max_blend_shape_count = max(result.max_blend_shape_count, mesh.get_blend_shape_count())
        for surface in range(mesh.get_surface_count()):
            var arrays := mesh.surface_get_arrays(surface)
            var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
            var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
            result.vertex_count += vertices.size()
            result.triangle_count += indices.size() / 3 if indices.size() > 0 else vertices.size() / 3
    for child in node.get_children():
        _accumulate_mesh_stats(child, result)

func _load_instances(paths: Array[String]) -> Array[Node]:
    var instances: Array[Node] = []
    for path in paths:
        var resource = ResourceLoader.load(path)
        if resource == null or not (resource is PackedScene):
            return []
        instances.append(resource.instantiate())
    return instances

func _commit_instances(parent: Node, instances: Array[Node]) -> void:
    for instance in instances:
        parent.add_child(instance)

func _clear_children(parent: Node) -> void:
    for child in parent.get_children():
        parent.remove_child(child)
        child.queue_free()

func _run() -> void:
    var compatible := _read_json("res://data/reference_two_piece.json")
    var incompatible := _read_json("res://data/incompatible_duplicate_tunic.json")
    var mask := _read_json("res://data/body_hide_mask.json")
    var outfit_root := Node3D.new()
    outfit_root.name = "OutfitRoot"
    get_root().add_child(outfit_root)

    var paths: Array[String] = [
        "res://products/tunic_rigged.glb",
        "res://products/trousers_rigged.glb",
    ]
    var pending := _load_instances(paths)
    var compatible_admitted := compatible.get("status", "") == "ACCEPTED" and pending.size() == 2
    var before_count := outfit_root.get_child_count()
    if compatible_admitted:
        _commit_instances(outfit_root, pending)
    var committed_count := outfit_root.get_child_count()
    var committed_pass := before_count == 0 and committed_count == 2

    var tunic_stats := _mesh_stats(outfit_root.get_child(0)) if committed_pass else {}
    var trousers_stats := _mesh_stats(outfit_root.get_child(1)) if committed_pass else {}
    var state_before_reject := outfit_root.get_child_count()
    var rejected := incompatible.get("status", "") == "REJECTED_ATOMIC"
    if not rejected:
        var unexpected := _load_instances(["res://products/tunic_rigged.glb"])
        _commit_instances(outfit_root, unexpected)
    var state_after_reject := outfit_root.get_child_count()
    var atomic_rejection_pass := rejected and state_before_reject == state_after_reject

    _clear_children(outfit_root)
    await process_frame
    var unequip_pass := outfit_root.get_child_count() == 0
    var body_mask_pass := bool(mask.get("mask_pass", false)) and int(mask.get("hidden_triangle_count", 0)) > 0
    var consumer_pass := committed_pass and atomic_rejection_pass and unequip_pass and body_mask_pass
    var receipt := {
        "contract": "GodotOutfitRuntimeReceipt/1",
        "godot_version": Engine.get_version_info(),
        "read_only_consumer": true,
        "compatible_outfit_pass": committed_pass,
        "atomic_rejection_pass": atomic_rejection_pass,
        "unequip_pass": unequip_pass,
        "body_occlusion_mask_pass": body_mask_pass,
        "equipped_garment_count": committed_count,
        "tunic": tunic_stats,
        "trousers": trousers_stats,
        "consumer_pass": consumer_pass,
        "secondary_motion_executed": false,
        "rig_aware_lod_executed": false,
    }
    print(MARKER + JSON.stringify(receipt))
    quit(0 if consumer_pass else 1)
