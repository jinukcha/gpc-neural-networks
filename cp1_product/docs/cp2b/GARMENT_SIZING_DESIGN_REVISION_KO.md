# GARMENT-SIZING-R0A 설계 보완 기록

## 판정

전 의류 공통 sizing foundation을 제품 설계에 추가했다. 이번 revision은 설계·로드맵 변경이며
패턴, Warp solver, CP3 frame 결과와 제품 acceptance를 변경하지 않는다.

## 추가된 정본

```text
docs/architecture/GARMENT_SIZING_SYSTEM_KO.md
docs/roadmap/GARMENT_SIZING_R0A_ROADMAP_KO.md
```

기존 Warp backend 설계에는 sizing integration boundary, dynamic seam/meshing, body envelope,
material/layer coupling과 두 개의 병렬 roadmap을 반영했다.

## 전 의류 적용 정책

모든 cloth garment family는 다음 공통 계약을 사용한다.

```text
BodyMeasurementProfile
GarmentSizeTable
GradeRuleSet
GarmentFitProfile
MaterialSizingProfile
LayerStackProfile
GarmentSizingRequest
SizedPatternInstance
```

실제 제도 공식과 topology는 각 복식 family의 resolver가 소유한다. 하나의 범용 resize 함수나
완성 3D mesh scaling으로 대체하지 않는다.

## Sizing mode

```text
STANDARD_SIZE
AUTO_BODY_FIT
CUSTOM_MEASUREMENTS
```

S/M/L은 첫 회귀 fixture이며 size table 계약은 임의 개수의 size를 허용한다.

## 현재 제품 상태 보존

CP2B reference pattern과 CP3 180-frame result/PNG는 reference body oracle로 유지한다.
현재 `technical_pass=false`, `product_acceptance=false`, CP2C blocked 판정은 변경하지 않는다.

## 다음 구현

`GARMENT-SIZING-R0A / CP0 — GLOBAL CONTRACT FOUNDATION / REFERENCE BODY ADAPTER / CP2B TUNIC RESOLVER PILOT`
