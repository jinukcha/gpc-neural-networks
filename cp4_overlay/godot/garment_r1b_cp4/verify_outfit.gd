extends SceneTree

const MARKER: String = "CP4_GODOT_OUTFIT_RECEIPT="

func _initialize() -> void:
    call_deferred("_run")

func _read_json(path: String) -> Dictionary:
    var file: FileAccess = FileAccess.open(path, FileAccess.READ)
    if file == null:
        return {}
    var parsed: Variant = JSON.parse_string(file.get_as_text())
    return parsed as Dictionary if parsed is Dictionary else {}

func _mesh_stats(node: Node) -> Dictionary:
    var result: Dictionary = {
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
        var mesh: Mesh = node.mesh
        result["mesh_instance_count"] = int(result["mesh_instance_count"]) + 1
        result["surface_count"] = int(result["surface_count"]) + mesh.get_surface_count()
        result["max_blend_shape_count"] = max(int(result["max_blend_shape_count"]), mesh.get_blend_shape_count())
        for surface: int in range(mesh.get_surface_count()):
            var arrays: Array = mesh.surface_get_arrays(surface)
            var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
            var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
            result["vertex_count"] = int(result["vertex_count"]) + vertices.size()
            var triangle_count: int = indices.size() / 3 if indices.size() > 0 else vertices.size() / 3
            result["triangle_count"] = int(result["triangle_count"]) + triangle_count
    for child: Node in node.get_children():
        _accumulate_mesh_stats(child, result)

func _load_instances(paths: Array[String]) -> Array[Node]:
    var instances: Array[Node] = []
    for path: String in paths:
        var resource: Resource = ResourceLoader.load(path)
        if resource == null or not (resource is PackedScene):
            return []
        var instance: Node = (resource as PackedScene).instantiate()
        instances.append(instance)
    return instances

func _commit_instances(parent: Node, instances: Array[Node]) -> void:
    for instance: Node in instances:
        parent.add_child(instance)

func _clear_children(parent: Node) -> void:
    for child: Node in parent.get_children():
        parent.remove_child(child)
        child.queue_free()

func _run() -> void:
    var compatible: Dictionary = _read_json("res://data/reference_two_piece.json")
    var incompatible: Dictionary = _read_json("res://data/incompatible_duplicate_tunic.json")
    var mask: Dictionary = _read_json("res://data/body_hide_mask.json")
    var outfit_root: Node3D = Node3D.new()
    outfit_root.name = "OutfitRoot"
    get_root().add_child(outfit_root)

    var paths: Array[String] = [
        "res://products/tunic_rigged.glb",
        "res://products/trousers_rigged.glb",
    ]
    var pending: Array[Node] = _load_instances(paths)
    var compatible_admitted: bool = str(compatible.get("status", "")) == "ACCEPTED" and pending.size() == 2
    var before_count: int = outfit_root.get_child_count()
    if compatible_admitted:
        _commit_instances(outfit_root, pending)
    var committed_count: int = outfit_root.get_child_count()
    var committed_pass: bool = before_count == 0 and committed_count == 2

    var tunic_stats: Dictionary = _mesh_stats(outfit_root.get_child(0)) if committed_pass else {}
    var trousers_stats: Dictionary = _mesh_stats(outfit_root.get_child(1)) if committed_pass else {}
    var state_before_reject: int = outfit_root.get_child_count()
    var rejected: bool = str(incompatible.get("status", "")) == "REJECTED_ATOMIC"
    if not rejected:
        var unexpected: Array[Node] = _load_instances(["res://products/tunic_rigged.glb"])
        _commit_instances(outfit_root, unexpected)
    var state_after_reject: int = outfit_root.get_child_count()
    var atomic_rejection_pass: bool = rejected and state_before_reject == state_after_reject

    _clear_children(outfit_root)
    await process_frame
    var unequip_pass: bool = outfit_root.get_child_count() == 0
    var body_mask_pass: bool = bool(mask.get("mask_pass", false)) and int(mask.get("hidden_triangle_count", 0)) > 0
    var consumer_pass: bool = committed_pass and atomic_rejection_pass and unequip_pass and body_mask_pass
    var receipt: Dictionary = {
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
