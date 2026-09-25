# 형상 생성 전략

## 1. Centerline canonicalization

1. reservation의 local frame과 meter/mm 변환을 검증한다.
2. centerline은 finite point/segment 집합으로 canonicalize한다.
3. ring winding으로 outer normal을 결정한다.
4. 연속 중복점, 0-length segment, self-intersection을 거부한다.
5. provider curve object나 native traversal order를 보존 identity로 사용하지 않는다.

수평 도시 성벽의 frame은 `+Y`를 up으로 고정하고, XZ 평면의 tangent와 ring winding 기반 outer normal로 구성한다. 거의 수직인 segment는 지원하지 않고 명시적으로 거부한다.

## 2. Deterministic segmentation

분할 경계:

```text
gate/tower/corner anchor
profile or tier change
grade break
curvature threshold
foundation-class change
maximum span length
sector/local-origin boundary
```

분할 결과는 stable source interval과 digest에서 ID를 파생한다. 입력 순서를 섞어도 같은 span set을 만든다.

## 3. Wall span

### Straight

- 2D wall profile을 local tangent 방향으로 extrude한다.
- foundation, wall body, walk, parapet을 별도 semantic solid/part로 유지한다.

### Curved

- bounded sampled path를 따라 profile sweep한다.
- sample tolerance와 maximum segment count를 profile에 고정한다.
- 급격한 twist 또는 self-intersection 가능성이 있으면 span을 재분할하거나 거부한다.

### Terrain-stepped

- terrain을 변형하지 않고 foundation step interface를 출력한다.
- wall walk의 접근 가능한 최대 grade와 step/ramp 정책은 별도 traversal requirement가 결정한다.
- terrain penetration/unsupported gap은 evidence로 측정하고 허용 envelope 밖이면 fail-closed다.

## 4. Join

가능하면 span을 거대 union하지 않고 exact join socket을 통해 조립한다.

```text
MITER     작은 방향 변화
BEVEL     큰 방향 변화, tower 불필요
TOWER     방어·실루엣·구조 결절
TRANSITION profile/tier/gate 전환
```

필요한 접합 overlap은 profile에 명시된 bounded tolerance만 허용하며, 우연한 겹침을 성공으로 간주하지 않는다.

## 5. Tower

- round tower: revolve 또는 polygonal approximation profile
- square/polygon tower: profile extrusion/loft
- wall penetration과 tangent 접속은 explicit sockets으로 구성
- tower roof는 `ROOF_SOCKET`까지만 소유하고, 경사지붕 realization은 ARCHITECTURE Producer가 담당
- playable tower interior는 `BuildingRequirement`를 발행해 WORLD-FIELD가 소유

## 6. Gate complex

Gate는 단순 벽 구멍이 아니라 compound assembly다.

```text
portal void
flanking towers or gatehouse mass
approach/forecourt interface
wall-walk bypass or bridge
leaf/portcullis/drawbridge sockets
road, collision, nav clear envelope
utility/ward/service sockets
```

단순 portal은 build123d/OCP boolean을 사용한다. bounded complex CSG가 필요하고 primary 경로가 실패한 경우에만 exact-qualified Manifold adapter를 별도 capability로 선택한다. fallback은 기록되며 동일 output이라고 가장하지 않는다.

## 7. Battlement and repetition

- merlon/crenel rhythm은 profile domain으로 계산한다.
- endpoint remainder는 대칭 또는 지정 anchor 기준으로 분배한다.
- 개별 반복 요소는 `ModuleInstancePlan`으로 보존한다.
- HERO 근접 제품은 개별 geometry, SUPPORT는 batched mesh, DISTANT는 silhouette/HLOD로 파생한다.

## 8. Semantic provenance

BREP boolean 후 native face identity가 변할 수 있으므로 OCCT face 번호를 장기 identity로 사용하지 않는다.

1. 입력 construction part에 stable semantic IDs를 부여한다.
2. 가능한 operation history를 adapter 내부 evidence로 기록한다.
3. 최종 face/surface는 geometric classifier와 source part history로 canonical semantic role에 연결한다.
4. ambiguity·unresolved face가 있으면 required acceptance를 거부한다.
5. tessellated triangle range는 semantic surface ID와 source module ID를 기록한다.

## 9. Tessellation

```text
linear tolerance
angular tolerance
minimum feature policy
normal generation policy
weld boundary policy
semantic-part merge policy
```

모든 값은 profile에 명시한다. build123d의 default가 바뀌어도 조용히 결과가 바뀌지 않게 한다. CAD tessellation 결과는 P0BB의 deterministic indexed cooker 입력이며, P0BB가 source coverage·LOD·meshlet을 계속 소유한다.

## 10. Assembly

- ring 전체는 module instances와 exact transforms의 assembly다.
- module bounds/clearance overlap을 검사한다.
- required wall-walk continuity graph를 확인한다.
- gate/road/bridge/BuildingProgram/utility socket을 exact relation으로 연결한다.
- partial required assembly는 PublishedWorld로 승격하지 않는다.
