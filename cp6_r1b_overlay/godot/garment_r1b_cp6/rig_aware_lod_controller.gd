class_name RigAwareLODController
extends RefCounted

var _variants: Dictionary = {}
var _lod0_max: float = 5.0
var _lod1_max: float = 14.0
var _hysteresis: float = 0.75
var _current_lod: String = ""
var _current_instance: Node3D


func configure(entry: Dictionary) -> void:
    _variants = entry.get("variants", {}).duplicate(true)
    var thresholds: Dictionary = entry.get("distance_thresholds_m", {})
    _lod0_max = float(thresholds.get("LOD0_MAX", 5.0))
    _lod1_max = float(thresholds.get("LOD1_MAX", 14.0))
    _hysteresis = float(entry.get("hysteresis_m", 0.75))


func desired_lod(distance_m: float) -> String:
    var distance: float = max(distance_m, 0.0)
    if _current_lod == "LOD0" and distance <= _lod0_max + _hysteresis:
        return "LOD0"
    if _current_lod == "LOD1":
        if distance >= _lod0_max - _hysteresis and distance <= _lod1_max + _hysteresis:
            return "LOD1"
    if _current_lod == "LOD2" and distance >= _lod1_max - _hysteresis:
        return "LOD2"
    if distance <= _lod0_max:
        return "LOD0"
    if distance <= _lod1_max:
        return "LOD1"
    return "LOD2"


func load_variant(lod_id: String) -> Node3D:
    if not _variants.has(lod_id):
        return null
    var resource: Resource = ResourceLoader.load(str(_variants[lod_id]))
    if resource == null or not (resource is PackedScene):
        return null
    var instance: Node = (resource as PackedScene).instantiate()
    return instance as Node3D


func swap_variant(
    parent: Node,
    distance_m: float,
    blend_shape_state: Dictionary,
) -> Dictionary:
    var desired: String = desired_lod(distance_m)
    if desired == _current_lod and is_instance_valid(_current_instance):
        return {"changed": false, "lod": desired, "state_applied": true}
    var candidate: Node3D = load_variant(desired)
    if candidate == null:
        return {"changed": false, "lod": desired, "state_applied": false, "reason": "LOAD_FAILED"}
    var applied: int = apply_named_blend_shapes(candidate, blend_shape_state)
    parent.add_child(candidate)
    if is_instance_valid(_current_instance):
        _current_instance.queue_free()
    _current_instance = candidate
    _current_lod = desired
    return {
        "changed": true,
        "lod": desired,
        "state_applied": applied > 0,
        "applied_blend_shape_count": applied,
    }


func apply_named_blend_shapes(root: Node, state: Dictionary) -> int:
    var applied: int = 0
    for mesh_instance: MeshInstance3D in mesh_instances(root):
        for name_value: Variant in state.keys():
            var index: int = mesh_instance.find_blend_shape_by_name(StringName(str(name_value)))
            if index >= 0:
                mesh_instance.set_blend_shape_value(index, float(state[name_value]))
                applied += 1
    return applied


func mesh_instances(root: Node) -> Array[MeshInstance3D]:
    var result: Array[MeshInstance3D] = []
    _collect_meshes(root, result)
    return result


func _collect_meshes(node: Node, result: Array[MeshInstance3D]) -> void:
    if node is MeshInstance3D:
        result.append(node as MeshInstance3D)
    for child: Node in node.get_children():
        _collect_meshes(child, result)


func skeleton_signature(root: Node) -> Array[String]:
    var signatures: Array[String] = []
    _collect_skeletons(root, signatures)
    signatures.sort()
    return signatures


func _collect_skeletons(node: Node, signatures: Array[String]) -> void:
    if node is Skeleton3D:
        var skeleton: Skeleton3D = node as Skeleton3D
        var names: PackedStringArray = PackedStringArray()
        for bone_index: int in range(skeleton.get_bone_count()):
            names.append(str(skeleton.get_bone_name(bone_index)))
        signatures.append("|".join(names))
    for child: Node in node.get_children():
        _collect_skeletons(child, signatures)
