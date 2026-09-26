# 계약 상세

## 1. Stable identity

```text
fortification definition   fortification/definition/<name>@<revision>
instance                   fortification/instance/<capital>/<ring>/<segment>@<revision>
span                       fortification/span/<instance>/<stable-key>
tower                      fortification/tower/<instance>/<stable-key>
gate                       fortification/gate/<instance>/<stable-key>
surface                    fortification/surface/<module>/<semantic-role>
socket                     fortification/socket/<module>/<role>
```

생성 순서, provider shape index, OCCT face ID, scene-node name은 identity authority가 아니다. 긴 ID는 readable prefix + canonical digest suffix를 사용한다.

## 2. Core objects

### `FortificationDefinition`

필수 필드:

```text
definition_id / revision / digest
supported_component_families
wall_profile_domain
foundation_profile_domain
tower_profile_domain
gate_profile_domain
battlement_profile_domain
semantic_surface_schema
socket_schema
material_slots
required_capabilities
resource_budgets
license/provenance refs
```

### `FortificationInstance`

```text
instance_id / revision
exact definition ref
exact reservation ref
parent transform
parameter selections
root-derived subseed
construction/history profile
product tier
```

### `FortificationProgram`

```text
program_id
catalog/definition/instance/reservation digests
bounded module requests
execution order derived from dependency DAG
aggregate budgets
required vs optional flags
```

## 3. Plan objects

### `WallRingPlan`

- ordered semantic ring, but each module ID is independent of array position
- inside/outside winding
- gate/tower/corner anchors
- required continuity edges

### `WallSpanPlan`

```text
span_id
source centerline interval
local frame
profile ref
foundation mode
start/end join socket
grade/curvature evidence
product tier
```

### `TowerPlan`

```text
tower_id
family
anchor and tangent refs
wall penetration/attachment mode
foundation interface
wall-walk sockets
optional BuildingProgram requirement
```

### `GateComplexPlan`

```text
gate_id
family
road/portal exact refs
clear width/height envelope
flanking towers/barbican modules
wall-walk continuity policy
door/portcullis/drawbridge sockets
service and ward sockets
```

## 4. Geometry outputs

### `FortificationGeometryPlan`

각 part는 다음을 기록한다.

```text
part_id
module_id
geometry_role
canonical local transform
bounds
indexed buffer refs
optional BREP/STEP ref
material slot IDs
semantic surface IDs
source coverage ref
```

### `SemanticSurfacePlan`

최소 roles:

```text
FOUNDATION_CONTACT
EXTERIOR_MASONRY
INTERIOR_MASONRY
WALL_WALK
PARAPET_INNER
PARAPET_OUTER
MERLON
CRENEL
GATE_PORTAL
GATE_DOOR_MOUNT
PORTCULLIS_RAIL
DRAWBRIDGE_HINGE
TOWER_INTERIOR_SOCKET
ROOF_SOCKET
STAIR_RAMP_SOCKET
UTILITY_SOCKET
DRAINAGE_SOCKET
COLLISION_HINT
NAVIGATION_HINT
```

### `FortificationSocketPlan`

Socket은 point만이 아니라 frame과 clearance를 갖는다.

```text
socket_id
role
origin + orthonormal basis
units/frame domain
clearance bounds
accepted relation/capability
required/optional
```

## 5. Provider capability query

커널은 provider 이름으로 분기하지 않고 다음 capability를 요청한다.

```text
cad.profile_extrusion@1
cad.path_sweep@1
cad.loft@1
cad.revolve@1
cad.boolean_difference@1
cad.boolean_union@1
cad.fillet_chamfer@1
cad.tessellation@1
cad.step_brep_export@1
cad.semantic_part_history@1
```

Binding 결과는 exact provider ID, adapter revision, source/runtime digest와 output contract를 고정한다.

## 6. Budgets

```text
max_ring_modules
max_span_control_points
max_modules_per_gate
max_towers
max_repeated_module_instances
max_boolean_operations
max_brep_solids
max_tessellated_vertices
max_tessellated_triangles
max_semantic_surfaces
max_sockets
max_artifact_bytes
max_repair_attempts
```

Budget 초과 시 quality를 몰래 낮추거나 required module을 삭제하지 않는다. typed failure로 닫고 partial accepted output을 게시하지 않는다.
