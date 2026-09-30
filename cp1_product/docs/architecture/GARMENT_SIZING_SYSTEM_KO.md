# GARMENT-SIZING-R0A — 전 의류 공통 사이징 시스템 정본

## 1. 목적

이 문서는 프로젝트의 모든 의류 family에 적용되는 sizing authority를 정의한다.
목표는 S/M/L 같은 표준 사이즈, 캐릭터 체형 자동 맞춤, 직접 상세 치수 입력을 하나의
의류 CAD 파이프라인으로 통합하는 것이다.

```text
표준 사이즈 선택
자동 체형 맞춤
직접 상세 치수 입력
        ↓
SizedPatternInstance
        ↓
복식별 panel/seam/topology compile
        ↓
Warp garment model
        ↓
검증된 wearable asset
```

이 시스템은 완성된 3D 메시를 확대하는 기능이 아니다. 모든 사이즈 변경은 2D 패턴과
봉제 관계를 다시 해석하고, body envelope 위에 다시 배치한 뒤 물리 모델을 다시 컴파일한다.

## 2. 적용 범위

공통 sizing authority는 다음 의류군에 필수 적용한다.

```text
상의: shirt, tunic, vest, bodice, jacket
전신: robe, dress, coat
하의: skirt, trousers, leggings
부위: sleeve, hood, collar, cuff, glove
외피: cape, cloak, mantle
내의와 layer: undergarment, padded layer, liner
```

갑옷, 신발, 벨트와 rigid wearable도 같은 `BodyMeasurementProfile`, fit/layer 계약과
cache identity를 재사용할 수 있다. 다만 cloth panel resolver를 강제하지 않고 각 제품군의
전용 CAD kernel이 패턴·shell·solid geometry를 소유한다.

## 3. 제품 원칙

### 3.1 사이즈 개수는 가변이다

내부 계약은 S/M/L 세 개에 고정하지 않는다. 첫 구현에서 S/M/L을 regression fixture로
사용할 수 있지만 size table은 XS–XXL, 숫자 호수, 지역별 체계 또는 임의 ID를 허용한다.

### 3.2 표준 사이즈와 개인 맞춤을 분리한다

```text
Grading
  base pattern을 표준 size 간 규칙으로 증감

Custom alteration
  같은 size 안에서 어깨·가슴·복부·길이·자세 차이를 보정
```

둘을 동일한 전역 배율로 처리하지 않는다.

### 3.3 모든 복식이 같은 공식을 쓰지 않는다

공통 영역은 치수의 의미, fit/ease, material, layer, size 선택과 결과 계약이다.
실제 제도 공식과 topology 전환은 각 복식 family가 소유한다.

### 3.4 직접 정점 수정은 정본이 아니다

패턴 parameter가 정본이다. 2D/3D editor에서 임의 정점 수정이 필요한 경우 별도 override와
provenance를 발행하며, 자동 sizing 연결이 끊겼는지 명시한다.

## 4. 세 가지 사용자 모드

### 4.1 `STANDARD_SIZE`

사용자가 size ID를 직접 선택한다.

```text
input: design + size_table + size_id + fit/material/layer
output: 해당 size의 graded SizedPatternInstance
```

상점 표시, NPC 군중, 규격품 제작과 regression에 적합하다.

### 4.2 `AUTO_BODY_FIT`

캐릭터 `BodyMeasurementProfile`을 사용한다.

```text
1. 가장 가까운 base size 선택
2. body block/height block 선택
3. 상세 치수 차이를 alteration plan으로 변환
4. topology admission 판단
5. sized pattern 발행
```

### 4.3 `CUSTOM_MEASUREMENTS`

사용자가 직접 상세 치수를 입력하거나 body profile 일부를 override한다. 단위, provenance,
confidence와 validation warning을 함께 기록한다.

세 모드는 최종적으로 동일한 `SizedPatternInstance`를 발행한다.

## 5. 공통 계약

## 5.1 `BodyMeasurementProfile/1`

신체 치수와 측정 provenance의 정본이다.

### 필수 영역

```text
identity
  body_id, units, mesh/skeleton hash, measurement method

vertical
  stature, torso height, shoulder-to-waist front/back,
  waist-to-hip, armscye depth, limb lengths

sections
  neck, chest/bust, underbust, waist, high hip, hip,
  upper arm, elbow, wrist, thigh, knee, calf, ankle

front/back distribution
  front arc, back arc, left/right asymmetry

shoulder/posture
  shoulder width/length/slope, forward head,
  spinal curvature, pelvic tilt

landmarks and sections
  landmark IDs, section loop IDs, confidence, source hash
```

전체 둘레만 저장하지 않는다. 같은 가슴둘레라도 넓은 등과 큰 앞가슴은 서로 다른 앞판·뒤판을
요구하므로 front/back arc를 별도로 저장한다.

### 품질 상태

```text
COMPLETE
PARTIAL_WITH_DERIVED_VALUES
MANUAL_OVERRIDE
UNSUPPORTED_MEASUREMENT_SET
```

유도된 값은 원 측정값과 구분한다.

## 5.2 `GarmentSizeTable/1`

임의 개수의 size와 base size를 소유한다.

```text
size_table_id
base_size_id
ordered size IDs
target body measurements
finished garment POMs 또는 grade increments
compatible body blocks
height blocks
```

size ID의 문자열 의미를 코드에 하드코딩하지 않는다.

## 5.3 `GradeRuleSet/1`

size 간 변화는 named pattern landmark 또는 POM 단위로 정의한다.

```text
chest_side_point: dx/dy
waist_side_point: dx/dy
shoulder_endpoint: dx/dy
armscye_underarm: dx/dy
hem_line: vertical offset
notch: curve arc-length rule
```

mesh vertex index를 grade rule authority로 사용하지 않는다.

## 5.4 `GarmentFitProfile/1`

신체 치수에서 완성 의복 치수로 변환하는 정책이다.

```text
fit_class
wearing ease
movement ease
style ease
front/back ease allocation
length policy
mobility clearance
negative-ease admission
```

권장 fit class:

```text
compression
close_fitted
fitted
regular
loose
oversized
outerwear
```

## 5.5 `MaterialSizingProfile/1`

```text
warp stretch
weft stretch
bias/shear response
thickness
compressibility
shrinkage
preferred grain direction
negative-ease limit
```

재질은 패턴 치수와 Warp material compile 양쪽에 영향을 준다.

## 5.6 `LayerStackProfile/1`

```text
layer order
inner garment thickness
local clearance map
compression allowance
contact priority
```

외투가 속옷과 같은 body surface에 직접 붙지 않도록 한다.

## 5.7 `GarmentSizingRequest/1`

```text
mode
garment_design_id
body_profile_id 또는 custom measurements
size_table_id / requested size ID
fit_profile_id
material_profile_id
layer_stack_id
overrides
```

## 5.8 `SizedPatternInstance/1`

모든 sizing mode의 단일 출력이다.

```text
selected size and blocks
resolved measurements
applied alteration plan
topology class and variant
panels and named boundaries
pattern landmarks and curves
grainline
notches
seamline and cutline
seam correspondence
POM report
warnings and unsupported conditions
input hashes
```

## 6. Base size와 block 선택

## 6.1 가장 가까운 size 선택

단일 가슴둘레가 아니라 정규화된 다차원 score를 사용한다.

```text
score(size) = Σ weight_i × robust_error(body_i, target_i)
```

주요 차원:

```text
chest/bust
waist
hip
shoulder width
front/back torso length
armscye depth
stature or height block
```

치수별 허용 오차와 중요도는 garment family가 소유한다. 바지는 hip/rise/thigh를 더 중요하게,
튜닉은 chest/shoulder/torso를 더 중요하게 평가한다.

## 6.2 body block

하나의 regular block으로 모든 체형을 해결하지 않는다.

```text
REGULAR
BROAD_SHOULDER
FULL_CHEST
FULL_ABDOMEN
FULL_HIP
STRAIGHT_TORSO
TALL
SHORT
```

block 선택 후에도 소규모 alteration은 허용한다.

## 6.3 topology admission

resolver는 항상 숫자를 반환하지 않는다.

```text
NORMAL_GRADE
CUSTOM_ALTERATION
ALTERNATE_BLOCK_REQUIRED
TOPOLOGY_CHANGE_REQUIRED
OUT_OF_SUPPORTED_RANGE
```

예:

```text
큰 bust prominence → dart/princess topology
큰 복부 돌출 → front length와 waist shaping
큰 상완 → sleeve/gusset variant
큰 waist-to-hip 차이 → dart/gather/panel variant
```

## 7. 완성 의복 치수 계산

기본식:

```text
finished measurement
= body measurement
+ wearing ease
+ movement ease
+ style ease
+ inner-layer allowance
- material stretch compensation
+ shrinkage allowance
```

front/back 분배는 별도 계산한다.

```text
finished_front_chest_arc
= body.front_chest_arc
+ front wearing/style ease
+ bust/abdomen alteration

finished_back_chest_arc
= body.back_chest_arc
+ back ease
+ shoulder-blade movement allowance
```

## 8. 복식별 resolver

권장 ownership 구조:

```text
source/wuxia_garment_oss/sizing/
├── body_profile/
├── size_table/
├── fit_profile/
├── material_profile/
├── layer_stack/
├── selection/
└── instance/

source/wuxia_garment_oss/garments/
├── sleeveless_tunic/
│   ├── design.py
│   ├── sizing.py
│   ├── pattern.py
│   ├── topology.py
│   └── admission.py
├── straight_sleeve_robe/
├── trousers/
├── cloak/
└── ...
```

`utils`, `misc`, `common` bucket을 만들지 않는다. 둘 이상의 family가 실제로 공유하는 안정된
계약과 알고리즘만 `sizing/`에 둔다.

## 9. 제품군별 주요 치수

| 제품군 | 주요 body/POM 입력 | topology 특이점 |
|---|---|---|
| 튜닉·셔츠 | chest, waist, shoulder, armscye, torso length | dart, yoke, gusset |
| 로브·드레스 | torso + hip + total length + sweep | waist join, flare, panel count |
| 바지 | waist, hip, rise, crotch depth, thigh, inseam | crotch curve, dart, gusset |
| 소매 | armscye, bicep, elbow, wrist, arm length | sleeve cap correspondence |
| 망토·클로크 | neck, shoulder, length, sweep | open front, hood/collar join |
| 장갑 | hand length/width, finger lengths | finger topology |
| 후드 | head/neck/face opening | center seam, gusset |

## 10. Dynamic boundary와 seam correspondence

패턴 크기가 바뀌면 boundary sample 수를 물리 길이에 따라 다시 정한다.

```text
sample_count = ceil(boundary_length / target_edge_length)
```

봉제되는 A/B 경계는 공통 normalized arc-length domain으로 재샘플링한다.

```text
t_i = i / (pair_count - 1)
A(t_i) ↔ B(t_i)
```

보장해야 하는 것:

```text
seam coverage = 1.0
unknown endpoint = 0
notch order preserved
orientation explicit
critical open boundary not sewn
seam-length mismatch within family threshold
```

## 11. Meshing과 Warp compile

body/size가 바뀌면 다음을 다시 발행한다.

```text
2D panel geometry
triangulation
rest UV basis
dual-area mass
grain axes
interior edge/rest dihedral
seam pairs and IDs
attachment targets
initial body arrangement
contact clearance
WarpGarmentModelPackage
```

재사용되는 것은 다음이다.

```text
XPBD execution kernels
checkpoint format
qualification framework
GLB exporter
Godot read-only consumer
```

## 12. Body envelope arrangement

고정 타원이나 하나의 avatar radius table을 전역 사용하지 않는다.

`BodyEnvelopePackage/1` 권장 구성:

```text
section heights
front/back section curves
left/right section curves
neck ring
shoulder surface
armscye landmarks
waist/high-hip/hip rings
actual collision mesh
```

pattern vertex는 해당 vertical section과 normalized lateral coordinate로 배치하고,
material thickness와 layer clearance를 더한다.

## 13. 게임과 제작 런타임

정식 Warp solve는 매 게임 프레임 수행하지 않는다.

실행 시점:

```text
character creation commit
body slider commit
garment equip/change
material or layer change
```

두 단계 preview:

```text
slider 이동 중
→ low-resolution pattern/envelope preview

slider 확정
→ final pattern compile + Warp solve + cache
```

캐시 key:

```text
body_profile_hash
garment_design_hash
size_table/size/block identity
fit_profile_hash
material_profile_hash
layer_stack_hash
topology_class
meshing_profile_hash
solver_profile_hash
```

## 14. 검증

### 14.1 패턴 전 검증

```text
measurement completeness and consistency
section loop closure
supported body range
ease and clearance minimum
pattern self-intersection = 0
positive panel area
landmark/curve continuity
seam ownership = 100%
notch and grainline presence
```

### 14.2 simulation 후 검증

```text
body penetration
exact non-sewn self-intersection
seam gap and coverage
edge strain
neck/armhole/crotch clearance
hem/stride clearance
contact pressure concentration
final tail convergence
```

### 14.3 최소 체형 matrix

```text
REFERENCE
SHORT_BROAD
TALL_NARROW
MUSCULAR_UPPER
FULL_CHEST_OR_ABDOMEN
```

각 fixture는 최소 front/back/left/right와 주요 fit detail PNG를 발행한다.

## 15. 현재 CP2B migration

현재 CP2B 고정 값은 `REFERENCE` body와 `M/REGULAR` 튜닉 fixture로 역직렬화한다.

```text
현재 pattern.py 상수
→ BodyMeasurementProfile reference fixture
→ TunicDesignSpec
→ GarmentFitProfile regular
→ TunicSizeResolver
→ 동일 SizedPatternInstance 재생성
```

첫 migration의 성공 조건은 reference body에서 기존 2D pattern/POM과 허용 범위 내 parity를
보이는 것이다. 기존 180-frame 결과는 비교 oracle이며 자동 폐기하지 않는다.

## 16. 완료 정의

전 의류 sizing foundation은 다음이 모두 성립해야 완료다.

```text
size count가 3개에 고정되지 않음
STANDARD/AUTO/CUSTOM 세 모드가 같은 output contract 사용
공통 계약과 복식별 resolver 책임 분리
단순 3D scaling 경로 없음
body front/back arc와 posture 반영
material/layer coupling 반영
dynamic seam correspondence 지원
topology switch 또는 explicit HOLD 지원
두 번째 garment family에서 공용성 검증
```
