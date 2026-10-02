class_name SecondaryMotionController
extends RefCounted

var _states: Dictionary = {}
var _fixed_step_hz: int = 60
var _substeps: int = 2


func configure(profile: Dictionary) -> void:
    _states.clear()
    _fixed_step_hz = int(profile.get("fixed_step_hz", 60))
    _substeps = int(profile.get("substeps", 2))
    for domain_value: Variant in profile.get("domains", []):
        var domain: Dictionary = domain_value
        var identity: String = str(domain["domain_id"])
        _states[identity] = {
            "domain": domain.duplicate(true),
            "position": 0.0,
            "velocity": 0.0,
        }


func step(delta: float, driver_values: Dictionary) -> Dictionary:
    var bounded_delta: float = min(max(delta, 0.0), 1.0 / float(_fixed_step_hz) * 2.0)
    var sub_delta: float = bounded_delta / float(max(_substeps, 1))
    for _substep: int in range(max(_substeps, 1)):
        for identity_value: Variant in _states.keys():
            _step_domain(str(identity_value), sub_delta, driver_values)
    return snapshot()


func _step_domain(identity: String, delta: float, driver_values: Dictionary) -> void:
    var state: Dictionary = _states[identity]
    var domain: Dictionary = state["domain"]
    var maximum: float = float(domain["max_weight"])
    var target: float = clamp(
        float(driver_values.get(identity, 0.0)) * float(domain["driver_gain"]),
        -maximum,
        maximum,
    )
    var omega: float = TAU * float(domain["natural_frequency_hz"])
    var damping: float = float(domain["damping_ratio"])
    var position: float = float(state["position"])
    var velocity: float = float(state["velocity"])
    var acceleration: float = omega * omega * (target - position) - 2.0 * damping * omega * velocity
    velocity += acceleration * delta
    position = clamp(position + velocity * delta, -maximum, maximum)
    if abs(position) >= maximum and sign(velocity) == sign(position):
        velocity = 0.0
    state["position"] = position
    state["velocity"] = velocity
    _states[identity] = state


func snapshot() -> Dictionary:
    var values: Dictionary = {}
    for identity_value: Variant in _states.keys():
        var identity: String = str(identity_value)
        var state: Dictionary = _states[identity]
        var domain: Dictionary = state["domain"]
        values[str(domain["blend_shape_name"])] = float(state["position"])
    return values


func domain_snapshot() -> Dictionary:
    var values: Dictionary = {}
    for identity_value: Variant in _states.keys():
        var identity: String = str(identity_value)
        var state: Dictionary = _states[identity]
        values[identity] = {
            "position": float(state["position"]),
            "velocity": float(state["velocity"]),
            "maximum": float(state["domain"]["max_weight"]),
        }
    return values


func apply(root: Node, values: Dictionary) -> int:
    var applied: int = 0
    for mesh_instance: MeshInstance3D in _mesh_instances(root):
        for name_value: Variant in values.keys():
            var index: int = mesh_instance.find_blend_shape_by_name(StringName(str(name_value)))
            if index >= 0:
                mesh_instance.set_blend_shape_value(index, float(values[name_value]))
                applied += 1
    return applied


func _mesh_instances(root: Node) -> Array[MeshInstance3D]:
    var result: Array[MeshInstance3D] = []
    _collect_mesh_instances(root, result)
    return result


func _collect_mesh_instances(node: Node, result: Array[MeshInstance3D]) -> void:
    if node is MeshInstance3D:
        result.append(node as MeshInstance3D)
    for child: Node in node.get_children():
        _collect_mesh_instances(child, result)
