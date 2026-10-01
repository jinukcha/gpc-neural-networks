# GARMENT-CAD-PRO-R1B — 리그 인식 게임 의상 플랫폼 설계 정본

## 0. 문서 상태

```text
program                  GARMENT-CAD-PRO-R1B
predecessor              GARMENT-CAD-PRO-R1A / CP6-R1
predecessor status       COMPLETE / immutable
document role            architecture authority
implementation status    NOT STARTED
target engine            Godot 4.7.2 Linux
primary runtime          game humanoid Skeleton3D
```

R1A는 신체 계측, 파라메트릭 패턴, 제조 construction, 물성 calibration,
motion-fit, 제조 2D와 neutral-gray 3D 제품까지 닫았다. R1B는 이 정본을
수정하지 않고 실제 게임 캐릭터 Skeleton에 장착 가능한 의상 제품 계층을 추가한다.

R1B의 목표는 한 벌을 리깅하는 것이 아니다. 앞으로 추가될 다수의 상의, 하의,
로브, 외투, 치마, 장갑, 신발, 머리 장식과 방어구 하부 의복이 같은 계약을 소비하고,
의상별 특수 구조는 해당 family 커널이 소유하게 만드는 것이다.

---

## 1. 제품 목표

R1B가 완료되면 다음 흐름이 성립해야 한다.

```text
R1A Pattern / Construction / Material authority
    ↓
Feature-complete garment topology
    ↓
Canonical humanoid skeleton + body skin source
    ↓
Garment-specific bind plan
    ↓
Skin weights + sparse local corrective deformation
    ↓
Optional secondary-motion domains
    ↓
Rig-aware LOD products
    ↓
Outfit layer composition
    ↓
Godot import / character-rig adapter / atomic equip
```

완료 제품은 다음을 지원한다.

- 캐릭터 Skeleton 호환성 사전 판정
- 체형별 garment rest topology와 skin binding
- Skeleton animation 기반 primary deformation
- 어깨, 겨드랑이, 팔꿈치, 골반, 샅, 무릎의 국소 corrective
- hem, loose sleeve, skirt, pleat free edge의 선택적 secondary motion
- shell, lining, facing, interfacing, closure hardware의 서로 다른 bind 정책
- 여러 의상의 동시 장착과 layer/coverage/occlusion 해결
- LOD 전환 시 bone identity, silhouette와 corrective 의미론 보존
- Godot에서 topology 수정 없는 read-only 소비

---

## 2. 비목표

R1B는 다음을 하지 않는다.

- 별도 범용 게임 엔진 또는 별도 의류 런타임 프레임워크를 만들지 않는다.
- Blender나 Godot에서 수기 weight painting을 정본으로 삼지 않는다.
- 캐릭터마다 복제된 의상 소스를 만들지 않는다.
- 애니메이션 pose마다 전체 의상을 교체하는 full-body morph 방식을 기본으로 사용하지 않는다.
- Rig 오류를 GLB import 후 vertex snap, weld, shrinkwrap 또는 mesh scaling으로 수정하지 않는다.
- 모든 의상을 하나의 거대 `garment.py`, `rig_utils.py`, `common` 폴더에 넣지 않는다.
- R1A 제조 정본이나 이미 수락된 CP6-R1 제품을 덮어쓰지 않는다.

---

## 3. 세 제품 계층

전문 시스템에서는 제조 패턴, simulation 제품과 게임 제품을 구분한다.

### 3.1 Manufacturing authority

```text
PatternDocument
ConstructionPackage
MaterialMeasurementSet
ManufacturingPatternPackage
```

치수, 패턴 곡선, seam, 시접, notch, lining, interfacing, 여밈과 공정 순서의 정본이다.

### 3.2 Simulation garment product

```text
FeatureCompleteTopologyPackage
WarpGarmentModelPackage
MotionFitQualificationReceipt
```

rest topology, component/layer ownership, 물성, seam과 contact의 정본이다.

### 3.3 Rigged game garment product

```text
RiggedGarmentProduct
SkinWeightField
CorrectiveDeformationSet
SecondaryMotionProfile
RiggedGarmentLODSet
GodotGarmentEquipPackage
```

Skeleton animation과 runtime 장착에 필요한 파생 제품이다. 게임 제품은 제조 정본을
역으로 수정하지 않는다.

---

## 4. 권위 방향과 변경 전파

```text
Body measurements changed
→ size/block/alteration stale
→ pattern stale
→ topology stale
→ bind plan stale
→ skin/corrective/LOD/runtime product stale

Skeleton rest pose changed
→ skeleton adapter stale
→ bind plan and all rig products stale
→ pattern and manufacturing authority remain valid

Material changed
→ simulation and secondary-motion profile stale
→ skeleton weights remain valid unless thickness/layer topology changes

Camera or evidence style changed
→ evidence only stale
```

최소 cache identity:

```text
canonical skeleton hash
external rig adapter hash
body skin-source hash
body profile hash
garment design and construction hash
feature topology hash
material and layer-stack hash
bind profile hash
skin-weight solver profile hash
corrective driver profile hash
LOD profile hash
outfit composition hash
Godot runtime lock hash
```

---

## 5. 공통 계약

R1B의 공유 계약은 실제로 여러 garment family가 소비하는 것만 포함한다.

```text
CanonicalSkeletonPackage/1
SkeletonAdapterMap/1
SkeletonCompatibilityReceipt/1

BodySkinSourcePackage/1
BodySurfaceSemanticMap/1

GarmentFamilyDefinition/1
GarmentVariantDefinition/1
GarmentBindProfile/1
RigBindPlan/1
SkinWeightField/1

CorrectiveDriverSet/1
CorrectiveDeformationSet/1
SecondaryMotionProfile/1

RiggedGarmentLODSet/1
RiggedGarmentProduct/1

OutfitAssemblyPlan/1
BodyOcclusionMask/1
OutfitCompatibilityReceipt/1

GodotGarmentEquipPackage/1
RuntimeGarmentReceipt/1
```

각 계약은 version, canonical hash, upstream source hash와 loss report를 갖는다.

---

## 6. 다수 의상군을 위한 소유 구조

```text
source/wuxia_garment_oss/
├── rig/
│   ├── skeleton_contract/
│   ├── skeleton_adapter/
│   ├── body_skin_source/
│   └── qualification/
├── outfit/
│   ├── registry/
│   ├── layering/
│   ├── occlusion/
│   ├── compatibility/
│   └── assembly/
├── runtime/
│   └── godot/
│       ├── equip_contract/
│       ├── skeleton_binding/
│       ├── corrective_driver/
│       ├── secondary_motion/
│       └── lod_runtime/
└── garments/
    ├── sleeveless_tunic/
    │   ├── rig_binding/
    │   ├── corrective_deformation/
    │   ├── secondary_motion/
    │   ├── rig_lod/
    │   └── runtime_product/
    ├── trousers/
    │   └── ...
    ├── sleeved_tunic/
    │   └── ...
    ├── straight_sleeve_robe/
    │   └── ...
    ├── skirt/
    │   └── ...
    └── coat/
        └── ...
```

`rig/`는 semantic bone과 body skin-source 같은 공통 계약을 소유한다.
실제 weight field, gusset 영향, sleeve cap 보정, crotch와 knee corrective는 해당 의상
family가 소유한다. 공통 폴더가 의상별 예외를 흡수하는 구조를 금지한다.

수기 소스는 원칙적으로 파일당 500 LOC, 함수당 80 LOC 이하를 유지한다. 단순 facade나
이름만 다른 helper로 제한을 회피하지 않고 데이터 소유권과 알고리즘 단계에 따라 분리한다.

---

## 7. Garment family 확장 모델

`GarmentFamilyDefinition/1`은 코드 plugin framework가 아니라 데이터 계약과 소유 커널의
등록 단위다.

필수 항목:

```text
family_id
semantic category
manufacturing authority resolver
required body measurements
required canonical bones
body coverage regions
outfit slots
layer class
thickness/compression profile
bind regions
corrective regions
secondary-motion domains
supported LOD profile
compatible closures and accessories
incompatibility rules
product builders
qualification suite
```

권장 category:

```text
torso_inner
torso_mid
torso_outer
full_body_inner
full_body_outer
lower_body_inner
lower_body_outer
arm_accessory
hand
foot
head_neck
armor_underlayer
accessory
```

같은 family에서 style, material, size와 trim만 다른 경우는
`GarmentVariantDefinition/1`로 표현한다. topology와 bind 정책이 질적으로 바뀌면 새 family
또는 명시적 topology variant가 필요하다.

---

## 8. Outfit composition

여러 의상을 장착할 때 단순 slot 독점만 사용하지 않는다. 긴 로브처럼 torso와 lower body를
동시에 덮는 의상이 있기 때문이다.

`OutfitAssemblyPlan/1`은 다음을 소유한다.

```text
ordered garment product IDs
semantic body coverage
layer order
compressed thickness envelope
shell-to-shell clearance
body occlusion mask
collision pair policy
secondary-motion priority
compatible and incompatible feature pairs
runtime budget profile
```

기본 layer order:

```text
BODY
BASE
INNER
MID
OUTER
ARMOR
ACCESSORY
```

같은 order에서도 semantic region별로 국소 우선순위를 가질 수 있다. Outfit compile은 기존
garment topology를 수정하지 않고 hide mask, collision admission과 runtime plan만 발행한다.

---

## 9. 리깅 제품의 기본 원칙

1. Skeleton semantic ID가 bone name보다 우선한다.
2. 외부 캐릭터 rig는 adapter를 통해 canonical skeleton에 매핑한다.
3. body surface의 skin weight와 semantic region은 직접 정본으로 고정한다.
4. garment weight는 surface barycentric transfer, region admission과 seam/layer 규칙으로 계산한다.
5. full-pose morph는 진단 fixture로만 보존하고 runtime 기본 경로는 skeletal skinning이다.
6. skinning 잔차는 작은 국소 corrective로 처리한다.
7. loose 영역만 secondary motion을 허용한다.
8. bonded interfacing, button과 rigid hardware는 별도 bind policy를 사용한다.
9. Godot는 import와 장착 소비자이며 topology authoring 도구가 아니다.
10. 실패 시 equip transaction 전체를 원자적으로 취소한다.

---

## 10. 완료 정의

R1B COMPLETE는 다음이 모두 성립해야 한다.

```text
canonical humanoid skeleton and adapter contract
body skin-source package
automatic bind for accepted R1A tunic and trousers
local corrective system
Godot equip/unequip and character swap
outfit layer and occlusion compiler
sleeved tunic and straight-sleeve robe generalization
rig-aware LOD and secondary motion
at least six garment families admitted by one registry
fixed skeletal motion suite PASS
fresh-process GLB/Godot reopen PASS
immutable R1A predecessor preserved
```

R1B가 끝나기 전까지 새 의상군을 단순 복제 방식으로 대량 추가하지 않는다.
