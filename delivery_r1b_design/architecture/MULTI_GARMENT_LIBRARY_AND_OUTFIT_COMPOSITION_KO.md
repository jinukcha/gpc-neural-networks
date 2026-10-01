# Multi-Garment Library / Outfit Composition 설계

## 1. 목적

의상 종류가 늘어도 각 의상이 독립적으로 build되고, 호환되는 제품은 전체 재시뮬레이션 없이
outfit으로 조합되도록 한다.

## 2. Registry

`GarmentLibraryRegistry/1`은 다음 index만 소유한다.

```text
garment family ID
available variant IDs
current product version
semantic slots and coverage
required skeleton contract
supported body-profile range
layer class
build product references
qualification status
license/source provenance
```

Registry는 의상 구현 코드를 소유하지 않는다. 각 family 커널의 product receipt를 참조한다.

## 3. Slots와 coverage

단일 slot 독점 모델을 피한다.

```text
slots
torso_inner / torso_mid / torso_outer
lower_inner / lower_outer
arm_inner / arm_outer
hand / foot / head_neck
full_body_inner / full_body_outer
armor_underlayer / accessory

coverage regions
chest_front / chest_back
waist / pelvis
left/right upper arm / forearm
left/right thigh / calf
neck / head / hand / foot
```

긴 robe는 여러 coverage region과 slot을 동시에 소유할 수 있다.

## 4. Layering

`LayerStackProfile/1`:

```text
layer class
nominal thickness
compressible thickness range
minimum clearance
friction pair class
collision priority
hide-body policy
secondary-motion priority
```

Outfit compile은 다음을 판정한다.

```text
skeleton compatibility
slot and coverage compatibility
total local thickness budget
shell intersection risk
closure obstruction
hem and sleeve motion-domain conflict
corrective driver conflict
runtime budget
```

## 5. Body occlusion

`BodyOcclusionMask/1`은 body triangle semantic regions로 발행한다. 의상 내부의 보이지 않는 body
surface만 숨기며 관절 주변 safety band를 유지한다. Texture alpha에 종속된 수기 mask를 정본으로
삼지 않는다.

## 6. Atomic outfit transaction

```text
resolve products
→ validate rig adapters
→ validate layer stack
→ compile body hide mask
→ compile collision and secondary-motion plan
→ preview
→ atomic equip commit
```

중간 실패 시 기존 outfit과 body mask를 유지한다. Equip과 unequip은 idempotent해야 한다.

## 7. Variant 확장

같은 family의 다음 차이는 variant로 관리할 수 있다.

```text
color/material assignment
trim and closure option
bounded length option
accepted size/body profile
compatible lining option
```

panel 수, seam graph, sleeve topology나 bind-region ownership이 달라지는 경우는 topology variant
또는 새 family다.

## 8. Library scale acceptance

```text
duplicate family ID                 0
missing product receipt             0
unresolved skeleton contract        0
incompatible outfit admitted        0
body mask outside coverage          0
layer order cycle                   0
registry → product hash mismatch    0
```
