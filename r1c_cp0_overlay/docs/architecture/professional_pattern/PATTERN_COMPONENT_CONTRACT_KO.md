# Pattern Component Contract

## `PatternComponentDefinition/1`

Component는 재사용 가능한 pattern·construction 정본이다.

필수 필드:

```text
component_id / revision
category / family / layer_role
geometry_authority = EXACT_2D_PATTERN
boundaries
landmarks
construction_features
parameter_ports
interface_ports
mirror_policy
topology_change_policy
```

`three_dimensional_primitive_authority`는 항상 `false`다.

## Boundary

각 boundary는 다음을 소유한다.

```text
boundary_id
semantic_role
interface_family
curve_kind
orientation
disposition
notch identity
parameter binding
```

Disposition:

```text
OPEN
SEWN
FINISHED
FOLD
CUT_INTERNAL
```

`SEWN` boundary는 assembly recipe의 interface에서 정확히 한 번 소유되어야 한다. `FINISHED`, `OPEN`, `FOLD` boundary를 봉제 interface로 사용하면 admission 실패다.

## Mirror

```text
NONE
MIRROR_ALLOWED
MIRROR_REQUIRED
```

Mirror는 stable component identity와 boundary semantics를 보존한다. 좌우 component를 단순 3D scale `-1`로 만드는 경로는 제품 정본이 아니다.

## Topology change

Component 자체의 panel 수나 interface family가 바뀌는 작업은 `GUIDED_ONLY` 또는 별도 component revision으로 처리한다. SAFE_AUTO completion에서 조용히 topology를 변경하지 않는다.
