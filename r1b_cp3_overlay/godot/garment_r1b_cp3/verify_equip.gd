extends SceneTree

const RuntimeScript = preload("res://garment_runtime.gd")

const ORDER := [
    ["ROOT", ""], ["PELVIS", "ROOT"], ["SPINE_01", "PELVIS"],
    ["SPINE_02", "SPINE_01"], ["CHEST", "SPINE_02"], ["NECK", "CHEST"],
    ["HEAD", "NECK"], ["L_CLAVICLE", "CHEST"], ["L_UPPER_ARM", "L_CLAVICLE"],
    ["L_FOREARM", "L_UPPER_ARM"], ["L_HAND", "L_FOREARM"],
    ["R_CLAVICLE", "CHEST"], ["R_UPPER_ARM", "R_CLAVICLE"],
    ["R_FOREARM", "R_UPPER_ARM"], ["R_HAND", "R_FOREARM"],
    ["L_THIGH", "PELVIS"], ["L_CALF", "L_THIGH"], ["L_FOOT", "L_CALF"],
    ["L_TOE", "L_FOOT"], ["R_THIGH", "PELVIS"], ["R_CALF", "R_THIGH"],
    ["R_FOOT", "R_CALF"], ["R_TOE", "R_FOOT"],
]

func _canonical_adapter() -> Dictionary:
    var result := {}
    for row in ORDER:
        result[row[0]] = row[0]
    return result

func _mixamo_adapter() -> Dictionary:
    return {
        "ROOT": "mixamorig:Root", "PELVIS": "mixamorig:Hips",
        "SPINE_01": "mixamorig:Spine", "SPINE_02": "mixamorig:Spine1",
        "CHEST": "mixamorig:Spine2", "NECK": "mixamorig:Neck", "HEAD": "mixamorig:Head",
        "L_CLAVICLE": "mixamorig:LeftShoulder", "L_UPPER_ARM": "mixamorig:LeftArm",
        "L_FOREARM": "mixamorig:LeftForeArm", "L_HAND": "mixamorig:LeftHand",
        "R_CLAVICLE": "mixamorig:RightShoulder", "R_UPPER_ARM": "mixamorig:RightArm",
        "R_FOREARM": "mixamorig:RightForeArm", "R_HAND": "mixamorig:RightHand",
        "L_THIGH": "mixamorig:LeftUpLeg", "L_CALF": "mixamorig:LeftLeg",
        "L_FOOT": "mixamorig:LeftFoot", "L_TOE": "mixamorig:LeftToeBase",
        "R_THIGH": "mixamorig:RightUpLeg", "R_CALF": "mixamorig:RightLeg",
        "R_FOOT": "mixamorig:RightFoot", "R_TOE": "mixamorig:RightToeBase",
    }

func _rest_position(semantic: String) -> Vector3:
    var table := {
        "ROOT": Vector3.ZERO, "PELVIS": Vector3(0, 0.92, 0),
        "SPINE_01": Vector3(0, 0.14, 0), "SPINE_02": Vector3(0, 0.14, 0),
        "CHEST": Vector3(0, 0.14, 0), "NECK": Vector3(0, 0.13, 0),
        "HEAD": Vector3(0, 0.18, 0),
        "L_CLAVICLE": Vector3(-0.09, 0.03, 0), "L_UPPER_ARM": Vector3(-0.22, -0.02, 0),
        "L_FOREARM": Vector3(-0.27, -0.03, 0), "L_HAND": Vector3(-0.20, -0.02, 0),
        "R_CLAVICLE": Vector3(0.09, 0.03, 0), "R_UPPER_ARM": Vector3(0.22, -0.02, 0),
        "R_FOREARM": Vector3(0.27, -0.03, 0), "R_HAND": Vector3(0.20, -0.02, 0),
        "L_THIGH": Vector3(-0.09, -0.36, 0), "L_CALF": Vector3(0, -0.43, 0),
        "L_FOOT": Vector3(0, -0.38, 0.08), "L_TOE": Vector3(0, 0, 0.16),
        "R_THIGH": Vector3(0.09, -0.36, 0), "R_CALF": Vector3(0, -0.43, 0),
        "R_FOOT": Vector3(0, -0.38, 0.08), "R_TOE": Vector3(0, 0, 0.16),
    }
    return table[semantic]

func _make_skeleton(name: String, adapter: Dictionary, omit: String = "") -> Skeleton3D:
    var skeleton := Skeleton3D.new()
    skeleton.name = name
    var indices := {}
    for row in ORDER:
        var semantic: String = row[0]
        if semantic == omit:
            continue
        skeleton.add_bone(adapter[semantic])
        var index := skeleton.get_bone_count() - 1
        indices[semantic] = index
        skeleton.set_bone_rest(index, Transform3D(Basis.IDENTITY, _rest_position(semantic)))
        var parent_semantic: String = row[1]
        if parent_semantic != "" and indices.has(parent_semantic):
            skeleton.set_bone_parent(index, indices[parent_semantic])
    get_root().add_child(skeleton)
    return skeleton

func _find_skeleton(node: Node) -> Skeleton3D:
    if node is Skeleton3D:
        return node as Skeleton3D
    for child in node.get_children():
        var found := _find_skeleton(child)
        if found != null:
            return found
    return null

func _mesh_stats(scene: PackedScene) -> Dictionary:
    var instance := scene.instantiate()
    get_root().add_child(instance)
    var stack: Array[Node] = [instance]
    var surfaces := 0
    var vertices := 0
    var triangles := 0
    var skin_surfaces := 0
    var blend_shapes := 0
    while not stack.is_empty():
        var node: Node = stack.pop_back()
        for child in node.get_children():
            stack.append(child)
        if node is MeshInstance3D:
            var mesh_instance := node as MeshInstance3D
            var mesh := mesh_instance.mesh
            blend_shapes = max(blend_shapes, mesh.get_blend_shape_count())
            for surface in range(mesh.get_surface_count()):
                surfaces += 1
                vertices += mesh.surface_get_array_len(surface)
                triangles += mesh.surface_get_array_index_len(surface) / 3
                var arrays := mesh.surface_get_arrays(surface)
                var bones = arrays[Mesh.ARRAY_BONES]
                var weights = arrays[Mesh.ARRAY_WEIGHTS]
                if bones != null and weights != null and len(bones) > 0 and len(weights) > 0:
                    skin_surfaces += 1
    var skeleton := _find_skeleton(instance)
    var result := {
        "surfaces": surfaces,
        "vertices": vertices,
        "triangles": triangles,
        "skin_surfaces": skin_surfaces,
        "blend_shapes": blend_shapes,
        "bone_count": skeleton.get_bone_count() if skeleton != null else 0,
    }
    instance.free()
    return result

func _tx(rows: Array, operation: String, slot: String, target: String, result: Dictionary) -> void:
    rows.append({
        "operation": operation,
        "slot": slot,
        "target": target,
        "result": result.reason,
        "pass": result.pass,
        "state_count": result.state_count,
    })

func _initialize() -> void:
    var tunic_scene := load("res://products/feature_complete_tunic_rigged.glb") as PackedScene
    var trousers_scene := load("res://products/trousers_rigged.glb") as PackedScene
    if tunic_scene == null or trousers_scene == null:
        push_error("rigged garment GLB import failed")
        quit(2)
        return
    var canonical_map := _canonical_adapter()
    var mixamo_map := _mixamo_adapter()
    var canonical := _make_skeleton("CanonicalCharacter", canonical_map)
    var mixamo := _make_skeleton("MixamoCharacter", mixamo_map)
    var incompatible := _make_skeleton("IncompatibleCharacter", mixamo_map, "R_CALF")
    var runtime := RuntimeScript.new()
    runtime.name = "GarmentEquipRuntime"
    get_root().add_child(runtime)
    var transactions: Array = []
    var upper := runtime.equip("UPPER", tunic_scene, canonical, canonical_map)
    _tx(transactions, "equip", "UPPER", canonical.name, upper)
    var duplicate := runtime.equip("UPPER", trousers_scene, canonical, canonical_map)
    _tx(transactions, "equip_duplicate", "UPPER", canonical.name, duplicate)
    var lower := runtime.equip("LOWER", trousers_scene, canonical, canonical_map)
    _tx(transactions, "equip", "LOWER", canonical.name, lower)
    canonical.set_bone_pose_rotation(canonical.find_bone("L_UPPER_ARM"), Quaternion(Vector3.FORWARD, 0.25))
    runtime.sync_slot("UPPER")
    runtime.sync_slot("LOWER")
    var swap_ok := runtime.swap_character("UPPER", mixamo, mixamo_map)
    _tx(transactions, "swap_character", "UPPER", mixamo.name, swap_ok)
    var before_failed_target := runtime.target_name("LOWER")
    var swap_bad := runtime.swap_character("LOWER", incompatible, mixamo_map)
    _tx(transactions, "swap_incompatible", "LOWER", incompatible.name, swap_bad)
    var failed_preserved := runtime.target_name("LOWER") == before_failed_target
    var unequip := runtime.unequip("UPPER")
    _tx(transactions, "unequip", "UPPER", "", unequip)
    var re_equip := runtime.equip("UPPER", tunic_scene, canonical, canonical_map)
    _tx(transactions, "re_equip", "UPPER", canonical.name, re_equip)
    var tunic_stats := _mesh_stats(tunic_scene)
    var trousers_stats := _mesh_stats(trousers_scene)
    var all_pass: bool = (
        upper.pass and not duplicate.pass and lower.pass and swap_ok.pass and not swap_bad.pass
        and failed_preserved and unequip.pass and re_equip.pass
        and runtime.state_count() == 2
        and tunic_stats.bone_count == 23 and trousers_stats.bone_count == 23
        and tunic_stats.skin_surfaces == tunic_stats.surfaces
        and trousers_stats.skin_surfaces == trousers_stats.surfaces
    )
    var receipt := {
        "contract": "GodotGarmentEquipReceipt/1",
        "godot_version": Engine.get_version_info(),
        "read_only_consumer": true,
        "runtime_node": runtime.name,
        "transactions": transactions,
        "atomic_equip_pass": upper.pass and not duplicate.pass and lower.pass,
        "character_swap_pass": swap_ok.pass,
        "failed_swap_preserved": failed_preserved,
        "unequip_re_equip_pass": unequip.pass and re_equip.pass,
        "final_state_count": runtime.state_count(),
        "tunic": tunic_stats,
        "trousers": trousers_stats,
        "consumer_pass": all_pass,
    }
    print("R1B_CP3_GODOT_RECEIPT=" + JSON.stringify(receipt))
    quit(0 if all_pass else 3)
