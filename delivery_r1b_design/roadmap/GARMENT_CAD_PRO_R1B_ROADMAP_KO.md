# GARMENT-CAD-PRO-R1B 구현 로드맵

## 프로그램 목표

R1A CP6-R1 완료본을 immutable predecessor로 유지하고, 제조·simulation authority를 실제 게임
캐릭터 Skeleton에 장착되는 다수 의상군 제품으로 전환한다.

## CP0 — Canonical skeleton / body skin source / compatibility

```text
CanonicalSkeletonPackage/1
SkeletonAdapterMap/1
BodySkinSourcePackage/1
SkeletonCompatibilityReceipt/1
```

기준 humanoid rig와 body skin source를 발행하고 현재 캐릭터/더미 rig를 adapter fixture로
입장시킨다. 의상 weight transfer는 아직 수행하지 않는다.

## CP1 — Automatic bind pilot / tunic + trousers

수락된 R1A sleeveless tunic과 trousers를 대상으로 body-surface barycentric transfer, semantic region,
seam/layer regularization과 hardware bind를 구현한다.

```text
SkinWeightField/1
RigBindPlan/1
tunic automatic bind
trousers automatic bind
```

## CP2 — Skeletal motion / local corrective

10-pose suite를 skeletal deformation으로 재실행하고 shoulder, underarm, elbow, waist, pelvis, crotch,
knee와 stride의 sparse local corrective를 구현한다. Full-body pose morph는 진단 비교만 유지한다.

## CP3 — Godot equip product / character swap

```text
RiggedGarmentProduct/1
GodotGarmentEquipPackage/1
atomic equip / unequip
Skeleton adapter validation
body occlusion
character swap
```

정확한 Godot 4.7.2 Linux import/reopen과 실제 캐릭터 장착을 닫는다.

## CP4 — Multi-garment registry / outfit layering

GarmentLibraryRegistry, OutfitAssemblyPlan, semantic coverage, layer thickness, body hide mask와 collision
admission을 구현한다. Tunic + trousers 조합과 incompatible fixture를 검증한다.

## CP5 — Sleeved tunic / straight-sleeve robe generalization

직접 팔 치수와 sleeve-cap construction을 소비해 sleeved tunic과 straight-sleeve robe를 실제
rig product로 추가한다. Shoulder, elbow와 underarm gusset corrective를 일반화한다.

## CP6 — Rig-aware LOD / secondary motion

LOD0/1/2의 bone-weight, corrective와 feature transfer를 구현한다. Hem, loose sleeve, pleat와 robe tail의
bounded secondary-motion domain을 추가하고 runtime budget profile을 검증한다.

## CP7 — Library-scale closeout

최소 다음 여섯 family를 하나의 registry와 outfit compiler에서 닫는다.

```text
sleeveless tunic
trousers
sleeved tunic
straight-sleeve robe
skirt or divided skirt
jacket or coat
```

완료 gate:

```text
all family products have rig receipts
fixed motion qualification PASS
at least three multi-garment outfits PASS
incompatible combinations reject atomically
LOD and secondary-motion PASS
Godot import/equip/unequip/character swap PASS
R1A predecessor unchanged
```

## 첫 실제 작업

**`GARMENT-CAD-PRO-R1B / CP0 — CANONICAL HUMANOID SKELETON / BODY SKIN-SOURCE / RIG COMPATIBILITY CONTRACT`**

R1A의 body, anthropometry와 accepted game products를 read-only 입력으로 사용한다. 기준 semantic
skeleton, 외부 rig adapter, canonical body weights, surface barycentric source와 compatibility receipt를
구현한다. Garment skin transfer, corrective, secondary motion과 Godot equip은 아직 수행하지 않는다.
