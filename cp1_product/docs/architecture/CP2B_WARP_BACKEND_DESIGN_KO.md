# CP2B Warp backend 정본

## 1. Authority

CP2B는 Blender Cloth가 아니라 프로젝트 소유 XPBD garment backend를 사용한다.
Upstream `warp-lang==1.17.0`은 array/kernel/device runtime으로만 사용하고,
deprecated `warp.sim` 또는 비상업 GarmentCode Warp fork는 사용하지 않는다.

```text
BodyMeasurementProfile
+ GarmentSizingRequest
+ garment-family PatternResolver
→ SizedPatternInstance
→ provider seamline
→ body-envelope-aware static input
→ WarpGarmentModelPackage
→ WarpSimulationState
→ final_simulation_mesh.npz
→ GLB
→ Godot read-only consumer
```

기존 CP2B의 고정 패턴은 삭제하지 않는다. 현재 기준 체형과 4패널 민소매 롱 튜닉은
`REFERENCE_BODY / REGULAR_TUNIC` 회귀 oracle로 보존한다. 이후 모든 새 의상 instance는
전역 sizing authority를 거쳐 `SizedPatternInstance`를 만든 뒤 Warp backend로 들어간다.

전역 sizing 정본은 다음 문서다.

```text
docs/architecture/GARMENT_SIZING_SYSTEM_KO.md
docs/roadmap/GARMENT_SIZING_R0A_ROADMAP_KO.md
```

## 2. 금지되는 크기 조정

다음 방식은 제품 sizing으로 인정하지 않는다.

```text
완성 3D cloth mesh의 XYZ 균일 배율
기준 M mesh를 가슴·허리별로 비균일 변형
봉제 후 정점 snap 또는 body shrinkwrap
세 사이즈에 고정된 if/else 분기
front/back arc를 전체 둘레의 50:50으로 가정
체형 차이를 무조건 ease 증가로 흡수
```

사이즈 변경은 2D 패턴 parameter, pattern landmark, 곡선, seam correspondence,
meshing density와 body-aware arrangement를 다시 계산하는 작업이다. Warp solver는 같은
실행 커널을 재사용하지만, static model package는 sized pattern마다 다시 컴파일한다.

## 3. CP1 static ownership

- four separate panel topologies; topology weld 금지
- dual triangle area를 3개 정점에 보존적으로 분배
- UV `v`를 warp, UV `u`를 weft 방향으로 사용
- triangle UV inverse와 rest 3D derivative를 보존
- manifold interior edge마다 양면·반대 정점·rest dihedral을 소유
- 8 named seams / 273 pair를 distance constraint input으로 소유
- 52 shoulder vertices를 explicit target에 연결
- body/self-contact, friction, substep, frame state는 CP2 이후 소유

이 수치는 CP2B reference fixture의 결과다. 전역 sizing이 활성화되면 패널 role과 named
seam identity는 유지할 수 있지만, vertex·triangle·seam sample·attachment 수는
`SizedPatternInstance`와 meshing profile에 따라 달라질 수 있다. 제품 계약은 고정 정수
상수 대신 다음 identity를 소유해야 한다.

```text
garment_design_id
topology_class
body_profile_hash
fit_profile_hash
material_profile_hash
layer_stack_hash
sized_pattern_hash
meshing_profile_hash
solver_profile_hash
```

## 4. Sizing integration boundary

Warp backend는 신체 치수를 직접 해석하지 않는다. 책임 방향은 다음과 같다.

```text
sizing/body_profile
  신체 측정, section, landmark, 자세와 비대칭 authority

sizing/fit_profile
  착용·동작·스타일 ease, 재질 신축, layer clearance

sizing/selection
  standard size, body block, height block 선택

garments/<family>/sizing
  해당 복식의 패턴 치수 해석

garments/<family>/pattern
  panel landmark와 곡선 생성

garments/<family>/topology
  seam role, dart/gusset/panel variant 선택

drape/warp_backend
  이미 해석된 sized pattern을 물리 모델로 컴파일하고 simulate
```

Warp 쪽에서 `chest`, `waist`, `shoulder`를 다시 계산하거나 범용
`resize_garment()`를 제공하지 않는다.

## 5. Standard size와 custom fit

내부 CAD 계약은 사이즈 수를 3개로 제한하지 않는다. S/M/L은 첫 regression fixture일 뿐이다.

```text
STANDARD_SIZE
  SizeTable의 정확한 pattern grade 결과

AUTO_BODY_FIT
  가장 가까운 base size/block 선택 후 상세 치수 alteration

CUSTOM_MEASUREMENTS
  사용자가 제공한 상세 치수와 override를 사용
```

세 모드는 모두 하나의 `SizedPatternInstance`를 출력하며 이후 provider/Warp 경로는 동일하다.

## 6. Dynamic seam and meshing

기존 CP2B의 fixed sample count는 reference fixture에만 유효하다.

```text
shoulder_left = 13
bodice_side_left = 22
waist_front = 35
```

새 sized pattern에서는 boundary length와 target edge length로 sampling 수를 정한다.
봉제되는 두 경계는 normalized arc length `t ∈ [0,1]`로 재샘플링해 동일 correspondence를
발행한다. 두 boundary의 native point 수가 달라도 seam coverage는 1.0이어야 한다.

```text
pair_count = max(
  ceil(length_a / target_spacing),
  ceil(length_b / target_spacing)
)

A(t_i) ↔ B(t_i)
```

neckline, armhole, hem 같은 open boundary는 seam compiler가 자동 봉제하지 않는다.
notch identity와 방향은 pattern landmark 기준으로 보존한다.

## 7. Body-aware arrangement and collision

고정 `garment_radii()` 또는 `_body_radii()` 타원은 reference fixture에만 허용한다.
전역 sizing 이후 static input은 `BodyEnvelopePackage`와 실제 collision mesh를 사용한다.

```text
section heights
front/back/left/right section curves
neck ring
shoulder surface
armscye landmarks
waist/hip rings
actual collision mesh
```

2D pattern vertex는 vertical section과 normalized lateral coordinate를 사용해 body envelope에
배치한다. layer clearance와 material thickness는 arrangement와 contact clearance에 함께 반영한다.

## 8. Material and layer coupling

같은 신체와 디자인이라도 material과 layer stack이 다르면 pattern과 Warp model이 달라진다.

```text
finished measurement
= body measurement
+ wearing ease
+ movement ease
+ style ease
+ inner-layer allowance
- stretch compensation
+ shrinkage allowance
```

material sizing profile은 warp/weft stretch, shear, thickness, compressibility, shrinkage와
negative-ease admission을 소유한다. Warp material profile은 sized pattern의 결과를 받아
mass, strain compliance, bending과 contact thickness를 컴파일한다.

## 9. CP3 recovery status

현재 CP3 recovery 180-frame 결과와 PNG는 reference fixture의 실행 증거로 보존한다.
숫자 gate는 통과했으나 exact CP2 source authority와 exact triangle self-intersection gate가
닫히지 않았으므로 제품 수락은 계속 false다.

Sizing 설계 추가는 기존 frame 결과를 폐기하거나 다시 실행하는 이유가 아니다. 이후
reference body에 대해 새 sizing resolver가 동일한 2D 치수를 재생성할 때 현재 패턴 및
frame 결과와 비교하는 oracle로 사용한다.

## 10. Roadmap

### Warp closeout track

1. CP3-R1: exact CP2 source rebind와 exact non-sewn triangle intersection gate.
2. CP4: canonical mesh GLB parity, Godot reopen, 원본 6뷰, CP2B closeout.

### Global garment sizing track

1. SIZING CP0: 공통 measurement/fit/material/layer/size-table 계약과 reference body adapter.
2. SIZING CP1: arbitrary-N size table, base size/block 선택과 alteration plan.
3. SIZING CP2: CP2B `TunicSizeResolver`, S/M/L fixture와 상세치수 mode.
4. SIZING CP3: dynamic boundary sampling, seam correspondence, topology admission.
5. SIZING CP4: body envelope arrangement, actual collision, layer-aware model compile.
6. SIZING CP5: 5개 체형 fixture와 fit qualification.
7. SIZING CP6: straight-sleeve robe 또는 trousers로 공용 계약 검증.

CP2C 또는 새 복식 family를 대량 추가하기 전에 SIZING CP0–CP3을 우선 완료한다.
