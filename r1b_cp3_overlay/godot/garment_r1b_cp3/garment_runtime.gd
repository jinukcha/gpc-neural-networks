extends Node
class_name GarmentEquipRuntime

const PARENTS := {
    "ROOT": "",
    "PELVIS": "ROOT",
    "SPINE_01": "PELVIS",
    "SPINE_02": "SPINE_01",
    "CHEST": "SPINE_02",
    "NECK": "CHEST",
    "HEAD": "NECK",
    "L_CLAVICLE": "CHEST",
    "L_UPPER_ARM": "L_CLAVICLE",
    "L_FOREARM": "L_UPPER_ARM",
    "L_HAND": "L_FOREARM",
    "R_CLAVICLE": "CHEST",
    "R_UPPER_ARM": "R_CLAVICLE",
    "R_FOREARM": "R_UPPER_ARM",
    "R_HAND": "R_FOREARM",
    "L_THIGH": "PELVIS",
    "L_CALF": "L_THIGH",
    "L_FOOT": "L_CALF",
    "L_TOE": "L_FOOT",
    "R_THIGH": "PELVIS",
    "R_CALF": "R_THIGH",
    "R_FOOT": "R_CALF",
    "R_TOE": "R_FOOT",
}

var _equipped: Dictionary = {}

func _find_skeleton(node: Node) -> Skeleton3D:
    if node is Skeleton3D:
        return node as Skeleton3D
    for child in node.get_children():
        var found := _find_skeleton(child)
        if found != null:
            return found
    return null

func _validate_skeleton(skeleton: Skeleton3D, adapter: Dictionary) -> Dictionary:
    if skeleton == null:
        return {"pass": false, "reason": "MISSING_SKELETON"}
    for semantic in PARENTS.keys():
        if not adapter.has(semantic):
            return {"pass": false, "reason": "MISSING_ADAPTER_ENTRY", "semantic": semantic}
        var source_name: String = adapter[semantic]
        var bone_index := skeleton.find_bone(source_name)
        if bone_index < 0:
            return {"pass": false, "reason": "MISSING_REQUIRED_BONE", "semantic": semantic}
        var parent_semantic: String = PARENTS[semantic]
        if parent_semantic == "":
            continue
        var expected_parent_name: String = adapter[parent_semantic]
        var parent_index := skeleton.get_bone_parent(bone_index)
        if parent_index < 0 or skeleton.get_bone_name(parent_index) != expected_parent_name:
            return {"pass": false, "reason": "PARENT_MISMATCH", "semantic": semantic}
    return {"pass": true, "reason": "COMPATIBLE"}

func _canonical_adapter() -> Dictionary:
    var result := {}
    for semantic in PARENTS.keys():
        result[semantic] = semantic
    return result

func _mapping(internal: Skeleton3D, target: Skeleton3D, adapter: Dictionary) -> Array:
    var rows: Array = []
    for semantic in PARENTS.keys():
        rows.append({
            "semantic": semantic,
            "internal": internal.find_bone(semantic),
            "target": target.find_bone(adapter[semantic]),
        })
    return rows

func equip(slot: String, scene: PackedScene, target: Skeleton3D, adapter: Dictionary) -> Dictionary:
    var before := _equipped.size()
    if _equipped.has(slot):
        return {"pass": false, "reason": "SLOT_OCCUPIED", "state_count": before}
    var validation := _validate_skeleton(target, adapter)
    if not validation.pass:
        return {"pass": false, "reason": validation.reason, "state_count": before}
    var instance := scene.instantiate()
    var internal := _find_skeleton(instance)
    var source_validation := _validate_skeleton(internal, _canonical_adapter())
    if not source_validation.pass:
        instance.free()
        return {"pass": false, "reason": "GARMENT_SKELETON_INVALID", "state_count": before}
    add_child(instance)
    instance.name = "equipped_%s" % slot
    _equipped[slot] = {
        "instance": instance,
        "internal": internal,
        "target": target,
        "adapter": adapter.duplicate(true),
        "mapping": _mapping(internal, target, adapter),
    }
    sync_slot(slot)
    return {"pass": true, "reason": "EQUIPPED", "state_count": _equipped.size()}

func sync_slot(slot: String) -> bool:
    if not _equipped.has(slot):
        return false
    var record: Dictionary = _equipped[slot]
    var internal: Skeleton3D = record.internal
    var target: Skeleton3D = record.target
    for row in record.mapping:
        var internal_index: int = row.internal
        var target_index: int = row.target
        internal.set_bone_pose_position(internal_index, target.get_bone_pose_position(target_index))
        internal.set_bone_pose_rotation(internal_index, target.get_bone_pose_rotation(target_index))
        internal.set_bone_pose_scale(internal_index, target.get_bone_pose_scale(target_index))
    return true

func swap_character(slot: String, target: Skeleton3D, adapter: Dictionary) -> Dictionary:
    if not _equipped.has(slot):
        return {"pass": false, "reason": "SLOT_EMPTY", "state_count": _equipped.size()}
    var validation := _validate_skeleton(target, adapter)
    if not validation.pass:
        return {
            "pass": false,
            "reason": validation.reason,
            "state_count": _equipped.size(),
            "preserved_target": target_name(slot),
        }
    var record: Dictionary = _equipped[slot]
    record.target = target
    record.adapter = adapter.duplicate(true)
    record.mapping = _mapping(record.internal, target, adapter)
    _equipped[slot] = record
    sync_slot(slot)
    return {"pass": true, "reason": "CHARACTER_SWAPPED", "state_count": _equipped.size()}

func unequip(slot: String) -> Dictionary:
    if not _equipped.has(slot):
        return {"pass": false, "reason": "SLOT_EMPTY", "state_count": _equipped.size()}
    var record: Dictionary = _equipped[slot]
    _equipped.erase(slot)
    var instance: Node = record.instance
    remove_child(instance)
    instance.free()
    return {"pass": true, "reason": "UNEQUIPPED", "state_count": _equipped.size()}

func state_count() -> int:
    return _equipped.size()

func target_name(slot: String) -> String:
    if not _equipped.has(slot):
        return ""
    var target: Skeleton3D = _equipped[slot].target
    return target.name
