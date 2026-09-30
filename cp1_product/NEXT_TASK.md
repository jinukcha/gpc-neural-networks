# 다음 실제 작업

**`GARMENT-SIZING-R0A / CP0 — GLOBAL CONTRACT FOUNDATION / REFERENCE BODY ADAPTER / CP2B TUNIC RESOLVER PILOT`**

재개 입력은 최신 통합본 하나다. 현재 CP2B reference pattern과 CP3 180-frame 결과를
변경하거나 다시 실행하지 않는다.

이번 단계의 범위:

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

현재 CP2B의 고정 치수를 reference body, M size, regular fit, medium woven material fixture로
역직렬화하고, S/M/L 세 regression fixture를 정의한다. 계약은 세 사이즈에 고정하지 않고
arbitrary-N size table을 허용해야 한다.

`STANDARD_SIZE`, `AUTO_BODY_FIT`, `CUSTOM_MEASUREMENTS` 세 mode가 동일한
`SizedPatternInstance`로 수렴하도록 구현한다. CP0에서는 2D pattern triangulation과
Warp simulation을 다시 실행하지 않는다.

수락 조건:

```text
reference body provenance와 units 고정
front/back chest·waist arc 분리
size table이 가변 길이
base size/block 선택 계약 존재
fit/material/layer hash identity 존재
기존 CP2B reference 치수 재현
완성 3D mesh scaling 경로 없음
source file <= 500 LOC / function <= 80 LOC
```
