extends SceneTree

const SecondaryController = preload("res://secondary_motion_controller.gd")
const LODController = preload("res://rig_aware_lod_controller.gd")
const MARKER: String = "CP6_GODOT_RUNTIME_RECEIPT="
const REQUIRED_CORRECTIVES: Array[String] = [
    "SHOULDER_RAISE",
    "UNDERARM_REACH",
    "ELBOW_BEND",
]
const PRODUCTS: Array[Dictionary] = [
    {
        "product_id": "SLEEVED_TUNIC_RIGGED_R1B",
        "owner": "sleeved_tunic",
        "profile": "res://profiles/sleeved_tunic.json",
    },
    {
        "product_id": "STRAIGHT_SLEEVE_ROBE_RIGGED_R1B",
        "owner": "straight_robe",
        "profile": "res://profiles/straight_robe.json",
    },
]


func _initialize() -> void:
    call_deferred("_run")


func _load_json(path: String) -> Dictionary:
    var text: String = FileAccess.get_file_as_string(path)
    var value: Variant = JSON.parse_string(text)
    if not (value is Dictionary):
        return {}
    return value


func _variant_path(owner: String, lod_id: String) -> String:
    return "res://products/%s/%s.glb" % [owner, lod_id.to_lower()]


func _surface_counts(mesh: Mesh, stats: Dictionary) -> void:
    stats["max_blend_shape_count"] = max(
        int(stats["max_blend_shape_count"]),
        mesh.get_blend_shape_count(),
    )
    for blend_index: int in range(mesh.get_blend_shape_count()):
        stats["blend_names"][str(mesh.get_blend_shape_name(blend_index))] = true
    for surface_index: int in range(mesh.get_surface_count()):
        var arrays: Array = mesh.surface_get_arrays(surface_index)
        var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
        var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
        stats["vertex_count"] = int(stats["vertex_count"]) + vertices.size()
        stats["triangle_count"] = int(stats["triangle_count"]) + (
            indices.size() / 3 if indices.size() > 0 else vertices.size() / 3
        )
        stats["surface_count"] = int(stats["surface_count"]) + 1


func _collect_stats(node: Node, stats: Dictionary) -> void:
    if node is Skeleton3D:
        var skeleton: Skeleton3D = node as Skeleton3D
        stats["skeleton_count"] = int(stats["skeleton_count"]) + 1
        var names: PackedStringArray = PackedStringArray()
        for bone_index: int in range(skeleton.get_bone_count()):
            names.append(str(skeleton.get_bone_name(bone_index)))
        stats["skeleton_signatures"].append("|".join(names))
    if node is MeshInstance3D and (node as MeshInstance3D).mesh != null:
        stats["mesh_instance_count"] = int(stats["mesh_instance_count"]) + 1
        _surface_counts((node as MeshInstance3D).mesh, stats)
    for child: Node in node.get_children():
        _collect_stats(child, stats)


func _inspect(instance: Node3D) -> Dictionary:
    var stats: Dictionary = {
        "mesh_instance_count": 0,
        "skeleton_count": 0,
        "surface_count": 0,
        "vertex_count": 0,
        "triangle_count": 0,
        "max_blend_shape_count": 0,
        "blend_names": {},
        "skeleton_signatures": [],
    }
    _collect_stats(instance, stats)
    var names: Array = stats["blend_names"].keys()
    names.sort()
    stats["blend_names"] = names
    stats["skeleton_signatures"].sort()
    return stats


func _load_variant(owner: String, lod_id: String) -> Node3D:
    var resource: Resource = ResourceLoader.load(_variant_path(owner, lod_id))
    if resource == null or not (resource is PackedScene):
        return null
    return (resource as PackedScene).instantiate() as Node3D


func _simulate(profile: Dictionary) -> Dictionary:
    var controller = SecondaryController.new()
    controller.configure(profile)
    var maximum_observed: Dictionary = {}
    var weights: Dictionary = {}
    for frame: int in range(240):
        var drivers: Dictionary = {}
        var phase: float = float(frame) * 0.105
        for domain_value: Variant in profile.get("domains", []):
            var domain: Dictionary = domain_value
            var identity: String = str(domain["domain_id"])
            var offset: float = float(drivers.size()) * 1.7
            drivers[identity] = sin(phase + offset)
        weights = controller.step(1.0 / 60.0, drivers)
        for name_value: Variant in weights.keys():
            var name: String = str(name_value)
            maximum_observed[name] = max(
                float(maximum_observed.get(name, 0.0)),
                abs(float(weights[name])),
            )
    var bounded: bool = true
    for state_value: Variant in controller.domain_snapshot().values():
        var state: Dictionary = state_value
        bounded = bounded and abs(float(state["position"])) <= float(state["maximum"]) + 1.0e-5
    return {
        "weights": weights,
        "domain_state": controller.domain_snapshot(),
        "maximum_observed": maximum_observed,
        "bounded": bounded,
    }


func _required_shapes(profile: Dictionary) -> Array[String]:
    var names: Array[String] = REQUIRED_CORRECTIVES.duplicate()
    for domain_value: Variant in profile.get("domains", []):
        names.append(str((domain_value as Dictionary)["blend_shape_name"]))
    return names


func _verify_product(spec: Dictionary) -> Dictionary:
    var profile: Dictionary = _load_json(str(spec["profile"]))
    var simulation: Dictionary = _simulate(profile)
    var lod_stats: Dictionary = {}
    var signatures: Array = []
    var state_applied: bool = true
    var target_names_preserved: bool = true
    for lod_id: String in ["LOD0", "LOD1", "LOD2"]:
        var instance: Node3D = _load_variant(str(spec["owner"]), lod_id)
        if instance == null:
            return {"accepted": false, "reason": "LOAD_FAILED", "lod": lod_id}
        var stats: Dictionary = _inspect(instance)
        var controller = LODController.new()
        var applied: int = controller.apply_named_blend_shapes(instance, simulation["weights"])
        state_applied = state_applied and applied >= len(profile.get("domains", []))
        for shape_name: String in _required_shapes(profile):
            target_names_preserved = target_names_preserved and shape_name in stats["blend_names"]
        lod_stats[lod_id] = stats
        signatures.append(stats["skeleton_signatures"])
        instance.free()
    var counts_descend: bool = (
        int(lod_stats["LOD0"]["vertex_count"]) > int(lod_stats["LOD1"]["vertex_count"])
        and int(lod_stats["LOD1"]["vertex_count"]) > int(lod_stats["LOD2"]["vertex_count"])
        and int(lod_stats["LOD0"]["triangle_count"]) > int(lod_stats["LOD1"]["triangle_count"])
        and int(lod_stats["LOD1"]["triangle_count"]) > int(lod_stats["LOD2"]["triangle_count"])
    )
    var skeleton_preserved: bool = signatures[0] == signatures[1] and signatures[1] == signatures[2]
    var accepted: bool = (
        bool(simulation["bounded"])
        and state_applied
        and target_names_preserved
        and counts_descend
        and skeleton_preserved
    )
    return {
        "product_id": spec["product_id"],
        "accepted": accepted,
        "secondary_motion_bounded": simulation["bounded"],
        "secondary_state_transfer_pass": state_applied,
        "corrective_and_secondary_names_preserved": target_names_preserved,
        "counts_descend": counts_descend,
        "skeleton_signature_preserved": skeleton_preserved,
        "maximum_secondary_weights": simulation["maximum_observed"],
        "lod_stats": lod_stats,
    }


func _run() -> void:
    var products: Array[Dictionary] = []
    for spec: Dictionary in PRODUCTS:
        products.append(_verify_product(spec))
    var accepted: bool = true
    for product: Dictionary in products:
        accepted = accepted and bool(product.get("accepted", false))
    var receipt: Dictionary = {
        "contract": "GodotRigAwareLODRuntimeReceipt/1",
        "godot_version": Engine.get_version_info(),
        "products": products,
        "runtime_acceptance": accepted,
        "secondary_motion_runtime": "BOUNDED_SPRING",
        "state_transfer": "NAME_BASED_ACROSS_LODS",
        "read_only_consumer": true,
    }
    print(MARKER + JSON.stringify(receipt))
    quit(0 if accepted else 1)
