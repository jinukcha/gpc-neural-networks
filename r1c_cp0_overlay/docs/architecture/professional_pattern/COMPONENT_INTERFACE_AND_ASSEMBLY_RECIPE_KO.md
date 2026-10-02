# Component Interface and Assembly Recipe

## `ComponentInterfaceSpec/1`

두 component boundary를 결합하는 유일한 authority다.

```text
interface_id
endpoint_a / endpoint_b
semantic_role
orientation_relation
length_policy
ratio_min / ratio_max
notch_policy
seam_allowance_policy
turn_of_cloth_policy
construction_predecessors
```

길이 정책:

```text
EQUAL
BOUNDED_EASE
GATHERED
```

Interface admission은 다음을 검사한다.

1. component instance와 boundary 존재
2. 두 boundary가 `SEWN`
3. interface family 일치
4. 방향 관계 일치
5. notch correspondence
6. 동일 boundary의 중복 소유 없음
7. 모든 sewn boundary가 정확히 한 interface에 귀속

## `GarmentAssemblyRecipe/1`

Recipe는 garment family의 component instance와 interface 집합을 소유한다.

```text
recipe_id
garment_family
component_instances
interface_ids
required_layers
allowed_completion_modes
topology_change_policy
technical_profile_id
visual_profile_id
```

CP0 reference recipe는 front/back bodice와 mirrored set-in sleeves를 조립한다. 오른쪽 underarm interface가 누락된 fixture는 geometry 실행 전에 거부된다.

## 금지

- 좌표가 가까운 edge의 자동 weld
- mesh primitive를 component로 등록
- 누락 seam을 임의 triangle로 연결
- completion 과정의 암묵적 panel 추가
