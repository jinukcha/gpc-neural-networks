# GARMENT-SIZING-R0A 로드맵

## 목표

모든 의류 family가 공유하는 sizing authority를 구현하고, 표준 사이즈·자동 체형 맞춤·직접
상세 치수 입력이 동일한 `SizedPatternInstance`로 수렴하도록 한다. 첫 파일럿은 CP2B
민소매 롱 튜닉이며, 두 번째 복식에서 공용성을 검증하기 전에는 완료로 판정하지 않는다.

## 선행 원칙

- 현재 CP2B reference pattern과 CP3 180-frame 결과를 oracle로 보존한다.
- 기존 3D cloth mesh를 scale하거나 shrinkwrap하지 않는다.
- 모든 변경은 2D pattern parameter와 seam authority에서 시작한다.
- S/M/L은 첫 fixture일 뿐이며 계약은 arbitrary-N size table을 지원한다.
- 공통 sizing 계약과 복식별 resolver를 분리한다.
- 각 checkpoint는 현재 변경 범위만 검증하고 180-frame 전체 실행을 반복하지 않는다.

## CP0 — GLOBAL CONTRACT FOUNDATION / REFERENCE BODY ADAPTER

### 구현

```text
BodyMeasurementProfile/1
GarmentSizeTable/1
GradeRuleSet/1
GarmentFitProfile/1
MaterialSizingProfile/1
LayerStackProfile/1
GarmentSizingRequest/1
SizedPatternInstance/1
```

현재 CP2B 상수를 reference body, regular fit, medium woven material과 M size fixture로
역직렬화한다. 아직 패턴을 다시 triangulate하거나 Warp simulation을 실행하지 않는다.

### 수락

```text
reference measurement provenance 명시
front/back arc 분리
units와 hash identity 고정
S/M/L fixture가 계약으로 표현됨
size count가 코드 상수로 고정되지 않음
기존 reference 패턴의 입력 치수 재현
```

## CP1 — BASE SIZE / BLOCK SELECTION / ALTERATION PLAN

### 구현

- 다차원 normalized size score
- garment-family별 measurement weight
- REGULAR/BROAD_SHOULDER/FULL_CHEST/FULL_ABDOMEN/TALL/SHORT block
- grade와 custom alteration 분리
- unsupported range와 topology-switch decision

### 수락

```text
동일 body request에 deterministic selection
한 치수만으로 size를 고르지 않음
base size와 block 선택 이유 receipt 제공
과도한 alteration은 alternate block 또는 HOLD
```

## CP2 — CP2B TUNIC SIZE RESOLVER / THREE-SIZE PILOT

### 구현

- `TunicDesignSpec`
- `TunicSizeResolver`
- S/M/L grade rule fixture
- detailed measurement alteration
- shoulder, neckline, armscye, chest/waist front/back, torso length와 hem 계산

### 수락

```text
REFERENCE M parity
S/M/L ordered growth
AUTO_BODY_FIT result
CUSTOM_MEASUREMENTS result
단순 전체 배율 없음
pattern landmark ownership 명확
```

## CP3 — DYNAMIC PATTERN / SEAM / MESH COMPILATION

### 구현

- sized panel curves
- physical-length boundary sampling
- normalized arc-length seam correspondence
- notch/grainline propagation
- dynamic vertex/triangle/seam count
- topology class/variant admission

### 수락

```text
pattern self-intersection 0
unknown seam endpoint 0
seam coverage 1.0
critical open boundary 보존
sample count가 size 변화에 따라 합리적으로 변함
reference pattern parity 유지
```

## CP4 — BODY ENVELOPE / LAYER-AWARE ARRANGEMENT

### 구현

- `BodyEnvelopePackage/1`
- section curve 기반 front/back arrangement
- actual body collision binding
- material thickness와 layer clearance
- dynamic attachment target
- sized `WarpGarmentModelPackage`

### 수락

```text
고정 타원 authority 제거
front/back ownership 유지
initial penetration bounded
inner-layer clearance 충족
mass/grain/bending/seam 재컴파일 PASS
```

## CP5 — FIVE-BODY FIT QUALIFICATION

### fixture

```text
REFERENCE
SHORT_BROAD
TALL_NARROW
MUSCULAR_UPPER
FULL_CHEST_OR_ABDOMEN
```

각 body에서 standard/auto/custom 중 필요한 mode를 실행한다.

### 수락

```text
pattern gate PASS
body penetration PASS
exact non-sewn self-intersection PASS
seam closure PASS
neck/armhole/hem fit clearance PASS
final convergence PASS
front/back/left/right + fit detail PNG 검수
```

## CP6 — SECOND GARMENT FAMILY GENERALIZATION

straight-sleeve robe 또는 trousers 중 하나를 선택한다.

### 목적

- 공통 계약이 튜닉 전용으로 오염되지 않았는지 확인
- garment-family resolver와 topology의 독립성 확인
- size table, fit, material, layer, cache identity 재사용 확인

### 완료 판정

```text
공통 sizing contract 수정 없이 두 번째 family 지원
family별 resolver가 별도 kernel owner를 가짐
전역 utils/common bucket 없음
STANDARD/AUTO/CUSTOM 공통 output 유지
```

## 이후 확장

```text
CP7: automatic mesh landmark/section measurement extractor
CP8: low-resolution live slider preview
CP9: cache registry와 invalidation
CP10: regional size table import/export
```

CP7 이후는 R0A 완료에 필수는 아니다. CP0–CP6이 전 의류 sizing foundation의 종료 기준이다.
