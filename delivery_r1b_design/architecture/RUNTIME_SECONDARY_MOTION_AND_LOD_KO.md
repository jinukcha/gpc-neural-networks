# Rig-aware LOD / Secondary Motion 설계

## 1. Primary와 secondary deformation 분리

```text
skeletal skinning
+ sparse local corrective
+ optional region-local secondary motion
```

모든 정점을 runtime cloth 대상으로 만들지 않는다.

## 2. Secondary-motion domain

허용 예:

```text
loose hem
skirt panel
robe tail
loose sleeve lower section
pleat free edge
closure loop
hanging accessory
```

기본 제외:

```text
bonded interfacing
button / buckle
reinforced facing
waistband core
tight upper torso
```

`SecondaryMotionProfile/1`:

```text
domain ID and owner vertices
anchor vertices / semantic bones
calibrated mass and damping
stretch and bending response
collision subset
outfit interaction priority
update frequency tier
deterministic fallback
LOD disposition
```

Backend는 교체 가능하지만 계약은 공통이다. Godot runtime에서 제품 topology를 수정하지 않는다.

## 3. LOD ownership

`RiggedGarmentLODSet/1`은 LOD별 독립 mesh와 다음 transfer를 소유한다.

```text
LOD0 → LODn surface map
bone-weight transfer map
corrective transfer map
component and feature retention map
secondary-motion domain reduction
material-zone map
silhouette-critical edge set
```

### LOD0

full 14-component 또는 family-complete 제품, 모든 corrective와 feature silhouette 유지.

### LOD1

facing/interfacing의 내부 상세는 병합 가능하나 dart, pleat, gather, gusset과 closure silhouette는
보존한다. Bone semantic identity는 LOD0와 동일하다.

### LOD2

내부 lining topology와 비가시 hardware 상세를 축약할 수 있다. 주요 hem, sleeve, collar와 closure
silhouette를 보존하며 secondary motion을 축약한다.

## 4. 전환 gate

```text
bone map identity                  exact
weight normalization              PASS
corrective semantic owner         preserved
visible silhouette jump           bounded
body penetration after switch      0
material zone loss                 0 for visible zones
runtime topology mutation          0
```

## 5. Runtime budget

절대 수치를 전 의상에 강제하지 않고 `RuntimeBudgetProfile/1`로 관리한다.

```text
target platform
maximum active garments
maximum skinned vertices by LOD
maximum corrective channels
maximum secondary-motion particles
update frequency
memory and upload budget
```

성능을 맞추기 위해 garment feature authority를 삭제하지 않고 더 낮은 LOD product를 선택한다.
