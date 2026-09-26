# ROYAL-CAPITAL FORTIFICATION CAD — 상세 설계

문서 revision: **RCF-DESIGN-R0**  
상위 기준: **RCK-DESIGN-R0 / RC_K0**  
상태: **DESIGN COMPLETE / IMPLEMENTATION NOT STARTED**

---

## 1. 목적

`FORTIFICATION` 도메인은 왕도 커널과 NETWORK 도메인이 확정한 방어 corridor·성문 socket·교량 접점·지형 기초 구역을 소비해 다음을 결정적으로 저작한다.

```text
외성·내성 curtain wall
직선·완만한 곡선·지형 계단형 wall span
일반·코너·관문 tower
주성문·보조문·수문 gate complex
barbican과 gate forecourt 경계
wall walk·parapet·battlement
성벽 계단·ramp·서비스 통로 socket
교량·도로·utility·건물 topology와의 명시적 접속면
HERO / PLAYABLE / SUPPORT / DISTANT 제품 파생 경계
```

목표는 하나의 거대한 성곽 BREP을 만드는 것이 아니다. 작고 검증 가능한 모듈을 exact identity와 source coverage로 조립해, 도시 배치가 변해도 필요한 span만 재생성할 수 있게 한다.

---

## 2. 첫 CAD 도메인으로 선택한 이유

SITE·URBAN·NETWORK는 먼저 계획돼야 하지만 geometry writer는 아니다. 실제 hard-surface CAD 중 FORTIFICATION을 먼저 고정하면 다음 공통 경계를 가장 이르게 검증할 수 있다.

- centerline/corridor → bounded CAD span 변환
- terrain foundation interface
- 반복 모듈과 고유 HERO 모듈의 분리
- BREP/STEP authoring evidence와 게임용 indexed mesh의 분리
- semantic surface와 traversal socket 보존
- build123d provider를 PAP Definition/Instance 경계 뒤에 격리
- P0BB cooker와 Godot 소비 경계

이 경계는 이후 BUILDING·ARCHITECTURE·LANDMARK에도 재사용한다.

---

## 3. 비목표

FORTIFICATION Producer는 다음을 소유하지 않는다.

```text
도시 전체 위치와 방어 링의 선택
지형·강·수계의 원본 field 수정
도로 중심선·교량 deck alignment
건물의 방·문·계단·vertical-core topology
성벽 내부 병영의 BuildingProgram
최종 석재 재질·노후화·조명
공성전 시뮬레이션과 구조공학 인증
LOD simplification·meshlet·codec 알고리즘
Godot scene tree 이름을 통한 의미 추론
CityJSON·IFC를 제품 SSOT로 사용하는 것
```

필요한 upstream reference를 제품 framework로 복사하지 않는다. 새 scheduler, 상태 머신, registry layer 또는 다중 manifest를 추가하지 않는다.

---

## 4. 권위와 경계

| 권위 | 소유자 | FORTIFICATION의 관계 |
|---|---|---|
| 지형·수계 field | WORLD-CONTINENT / SITE | exact terrain/foundation ref를 읽음 |
| 방어 링·gate reservation | ROYAL-CAPITAL-KERNEL / NETWORK | 배치 권위; 임의 이동 금지 |
| 성벽·탑·성문 CAD | PAP Fortification Producer | 이 설계의 geometry 권위 |
| gatehouse/tower 내부 방 topology | WORLD-FIELD | 선택 playable interior에만 결합 |
| utility·ward 장치 | UTILITY Producer | 성벽 socket에 exact binding |
| material/lookdev | ART | semantic material slot을 소비 |
| collision/navigation | WORLD-FIELD / RUNTIME | CAD 결과에서 파생하되 별도 권위 |
| LOD/HLOD/cooked asset | MEGASTRUCTURE P0BB | accepted source mesh를 소비 |
| 제품 장면 | Godot | 파생 소비자; SSOT 아님 |

---

## 5. 커널 패턴

```text
DefinitionCatalogSnapshot
  → FortificationDefinition
  → FortificationInstance
  → finite FortificationProgram
  → exact FortificationReservationPlan
  → bounded span / join / tower / gate candidate plans
  → exact Producer binding
  → project-owned build123d adapter
  → module geometry + semantic surfaces + sockets
  → FortificationAssemblyPlan
  → ArtifactIndex / CapitalAssemblyPlan
  → P0BB cook / Godot derived product
```

### 5.1 Definition

`FortificationDefinition`은 재사용 가능한 형식과 허용 domain을 소유한다.

```text
wall profile family
foundation family
span/join family
allowed tower families
gate-complex families
battlement pattern
wall-walk and access policy
material slot schema
semantic surface schema
CAD/tessellation budgets
required Producer capabilities
```

### 5.2 Instance

`FortificationInstance`는 exact Definition revision/digest와 배치를 소유한다.

```text
reservation ref
ring/segment identity
parent transform
chosen finite parameters
root-derived subseed
construction epoch / repair layer
product tier
required access and utility sockets
```

### 5.3 Program

`FortificationProgram`은 하나의 generation request에서 실행할 유한 모듈과 전체 예산을 소유한다. array order는 identity가 아니며, 모든 모듈은 namespaced stable ID를 갖는다.

---

## 6. 입력 계약

필수 입력은 `FortificationReservationPlan`이다.

```text
reservation_id / revision / digest
capital instance ref
ring role and defended scope
canonical local frame
centerline segments or sampled curve with declared tolerance
exclusive corridor and vertical envelope
inside/outside winding
foundation-zone and terrain-sample refs
gate / tower / corner / bridge anchors
required road and portal clearance
wall-walk continuity requirements
threat arcs and visibility hints
product tier and camera-distance envelope
budget profile
```

### 6.1 입력 불변조건

- 모든 좌표는 finite하고 canonical frame으로 정규화한다.
- meter geometry와 integer-millimeter placement 사이 변환이 명시적이어야 한다.
- centerline self-intersection을 허용하지 않는다.
- gate anchor는 정확히 하나의 road/portal socket에 연결된다.
- required tower/gate anchor는 corridor 밖으로 clamp하지 않는다.
- terrain sample revision과 reservation digest가 stale이면 실행하지 않는다.
- ring winding으로 inside/outside를 결정하며 provider face normal 순서를 신뢰하지 않는다.

---

## 7. 출력 계약

### 7.1 정본 출력

```text
FortificationAssemblyPlan
FortificationGeometryPlan
FortificationSemanticSurfacePlan
FortificationSocketPlan
FoundationInterfacePlan
TraversalInterfacePlan
ModuleInstancePlan
SourceCoverageMap
CADProviderReceipt
FortificationQualificationEvidence
```

### 7.2 선택적 저작 증거

```text
STEP or BREP stored copy
provider diagnostic geometry
technical section drawing
```

STEP/BREP은 검토·재저작용 evidence일 수 있지만 제품 identity나 Godot SSOT가 아니다. 정본 제품 경계는 project-owned canonical metadata와 indexed geometry buffer다.

### 7.3 출력 불변조건

- 모든 part와 semantic surface는 source module/span으로 역추적된다.
- required socket은 geometry의 실제 위치와 일치한다.
- 같은 exact input/adapter/runtime/tolerance는 같은 canonical 결과를 생성한다.
- required module이 실패하면 전체 assembly는 부분 성공으로 게시되지 않는다.
- optional ornament만 fail-open 가능하다.

---

## 8. 컴포넌트 문법

### 8.1 Curtain wall

```text
STRAIGHT_SPAN
CURVED_SPAN
TERRAIN_STEPPED_SPAN
RETAINING_SPAN
RIVERBANK_SPAN
```

각 span은 foundation, wall body, wall walk, inner parapet, outer parapet을 별도 semantic part로 유지한다.

### 8.2 Join

```text
MITER_JOIN
BEVEL_JOIN
TOWER_JOIN
GATE_TRANSITION_JOIN
GRADE_BREAK_JOIN
```

긴 ring 전체를 union하지 않고 join interface를 가진 bounded modules로 조립한다.

### 8.3 Tower

```text
ROUND_TOWER
SQUARE_TOWER
POLYGONAL_TOWER
CORNER_TOWER
GATE_TOWER
WATCH_TOWER
```

Tower는 wall span endpoint/tangent와 foundation interface를 소비한다. 내부 방은 별도 WORLD-FIELD `BuildingProgram`으로 결합할 수 있다.

### 8.4 Gate complex

```text
MINOR_GATE
MAIN_GATEHOUSE
BARBICAN_GATE
WATER_GATE
POSTERN
```

Gate complex는 road clear envelope, portal plane, door/portcullis/drawbridge socket, wall-walk bypass 또는 연결을 명시한다. 실제 door state/physics는 RUNTIME 도메인 소유다.

### 8.5 Repeated modules

```text
MERLON
CRENEL
DRAIN_SPOUT
WALL_LAMP_SOCKET
BANNER_SOCKET
WARD_PYLON_SOCKET
GUARDRAIL_MODULE
```

반복 요소는 가능한 `ModuleInstancePlan`으로 유지한다. 수천 개의 merlon을 하나의 BREP boolean 결과로 합치지 않는다.

---

## 9. 제품 티어

| 티어 | 대상 | 형상 경계 |
|---|---|---|
| `HERO` | 주성문, 주요 tower, palace citadel gate | 고유 silhouette, portal·socket·선택 interior 완전 보존 |
| `PLAYABLE` | wall walk, 보조문, 접근 가능한 tower | traversal surface, collision/nav 입력, 중간 상세 |
| `SUPPORT` | 일반 outer-wall span과 반복 tower | modular exterior, 단순 collision, 제한 socket |
| `DISTANT` | 원경 ring과 후면 wall | silhouette/HLOD source, 실제 merlon 개별 형상 생략 가능 |

Definition/Instance identity는 티어와 무관하게 보존한다. 낮은 티어는 같은 의미 객체의 파생 geometry다.

---

## 10. 역사층과 손상 경계

```text
construction_epoch
wall_ring_generation
expansion_phase
repair_epoch
stone_replacement_ratio
parapet_rebuild_pattern
blocked_postern
utility_retrofit
battle_damage_state
```

역사층은 geometry recipe parameter이지만 구조적 required clearance와 traversal을 훼손할 수 없다. 손상은 별도 `DamageEnvelope`와 product profile을 통해 적용하며 기본 Definition을 덮어쓰지 않는다.

---

## 11. 초기 fixture

`fortification/caelmere-outer-ring@1`은 설계 검증용 fixture다.

```text
outer ring 역할
하나의 straight span
하나의 완만한 curved span
하나의 terrain-stepped span
일반 tower 2개
corner tower 1개
minor gate 1개
main gatehouse 1개
wall-walk continuity required
main gate road clearance required
HERO + PLAYABLE + SUPPORT tier 혼합
```

정확한 왕도 직경·벽 높이·tower 간격은 SITE/URBAN/NETWORK child design에서 고정하기 전까지 example domain이며 컨셉 이미지의 축척을 사실로 취급하지 않는다.

---

## 12. 종료 기준

이 상세 설계는 다음을 고정하면 종료된다.

- Definition/Instance/Program과 exact reservation 입력
- component grammar와 typed output
- build123d 격리 adapter 경계
- optional Manifold CSG 경계
- semantic surface·socket·source coverage
- 유한 R0A~R0F child roadmap
- selected OSS source와 라이선스/provenance
- typed failures와 focused validation

실제 geometry·runtime·Godot 검증은 후속 단계다.
