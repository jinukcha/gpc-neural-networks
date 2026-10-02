# GARMENT-CAD-PRO-R1C — Blender-free Product Pipeline REV1

## 1. 결정

R1C의 자동 제품화 경로에서 Blender를 **필수 런타임·exporter·visual acceptance 도구에서 제거**한다.

```text
기존 필수 경로
Pattern → Warp → Blender → GLB → PNG

REV1 필수 경로
Pattern → Warp → Direct GLB → glTF validation → Godot 4.7.2 → PNG
```

Blender는 사용자가 명시적으로 수기 sculpt, topology 편집, UV 편집 또는 artist-authored corrective를 요청한 경우에만 선택적 외부 도구로 사용할 수 있다. 선택적 Blender 작업은 자동 제품 정본을 조용히 덮어쓸 수 없으며, 별도의 `GUIDED_ARTIST_OVERRIDE` transaction과 provenance를 가져야 한다.

## 2. Immutable predecessor

```text
predecessor
GARMENT-CAD-PRO-R1C / CP4-R1

predecessor decision
CP4_R1_ACCEPTED_PATTERN_DRIVEN_TUNIC

policy
CP4-R1 BLEND / GLB / receipts / 12-view PNG를 read-only로 보존
```

CP4-R1의 `.blend`는 과거 accepted predecessor의 증거로만 유지한다. REV1 이후 새 제품은 `.blend`를 생성하지 않는다.

## 3. 목표

1. 실행 단계와 대형 runtime 의존성을 줄인다.
2. CAD·simulation 제품의 권위를 Blender scene이 아니라 source pattern과 compiled geometry가 소유하게 한다.
3. 최종 시각 수락 화면을 실제 게임 엔진인 Godot 4.7.2와 일치시킨다.
4. GLB export, reopen, validation과 runtime import를 deterministic contract로 만든다.
5. component·boundary별 metric fidelity와 직접 art review를 제품 acceptance의 필수 조건으로 만든다.

## 4. Canonical pipeline

```text
CP3 repaired pattern snapshot
        ↓
component-aware triangulation
        ↓
avatar-aware arrangement
        ↓
component / boundary metric decomposition
        ↓
isometric arrangement repair
        ↓
Warp material settling
        ↓
technical qualification
        ↓
deterministic direct GLB writer
        ↓
GLB fresh-process reopen
        ↓
Khronos glTF Validator
        ↓
Godot 4.7.2 fresh import / reopen
        ↓
neutral-gray 12-view capture
        ↓
direct visual art review
        ↓
product acceptance
```

## 5. Runtime ownership

### 5.1 Python pattern/materialization kernels

소유 범위:

```text
pattern curves
triangulation
component ownership
boundary sampling
seam correspondence
arrangement
metric decomposition
normals / UV generation
direct GLB serialization
product fingerprint
```

### 5.2 Warp 1.17.0

소유 범위:

```text
material settling
body collision
strain / convergence
bounded secondary motion
```

Warp는 GLB export나 visual acceptance를 소유하지 않는다.

### 5.3 Direct GLB writer

필수 attribute:

```text
POSITION
NORMAL
TEXCOORD_0
indices
```

Rigged 단계에서 추가:

```text
JOINTS_0
WEIGHTS_0
inverse bind matrices
corrective morph targets
secondary-motion morph targets
```

필수 metadata:

```text
component stable ID
component family
boundary ownership
seam interface ID
material zone
source snapshot SHA-256
arrangement receipt SHA-256
product fingerprint
```

GLB writer는 stable component order, stable primitive order, stable accessor order와 canonical float serialization을 사용한다. 동일 입력은 동일 product fingerprint를 발행해야 한다.

### 5.4 Khronos glTF Validator

구현 시 exact release 또는 commit을 lock한다. Terminal gate:

```text
errors                         0
unresolved resources           0
invalid accessors              0
non-finite attributes          0
invalid index references       0
invalid skin / animation data  0
```

Warning은 무시하지 않는다. 허용 warning은 ID·사유·owner와 disposition을 receipt에 기록한다.

### 5.5 Godot 4.7.2

Godot은 다음을 소유한다.

```text
fresh import
fresh-process reopen
mesh fingerprint readback
neutral-gray material override
body-visible runtime scene
12-view PNG capture
diagnostic overlay
final game-runtime visual acceptance
```

Godot에서 금지:

```text
pattern fitting
vertex repair
weld / snap
shrinkwrap
mesh scaling으로 size 생성
skin weight 수기 수정
topology 변경
```

## 6. Metric fidelity authority

CP4-R1 baseline:

```text
pattern_to_arrangement_ratio p50  0.9166309015
pattern_to_arrangement_ratio p95  1.8909642372
pattern_to_arrangement_ratio p99  5.4355218328
```

단순한 `L_arrangement / L_pattern`만 사용하지 않는다. 압축과 신장을 대칭적으로 다루기 위해 다음을 게시한다.

```text
raw_ratio       r = L_arrangement / L_pattern
symmetric_ratio s = max(r, 1/r)
log_strain      e = abs(log(r))
```

모든 structural edge를 정확히 한 owner class에 배치한다.

```text
COMPONENT_BOUNDARY
SEAM_NEIGHBORHOOD
COMPONENT_INTERIOR
FOLD_OR_CONSTRUCTION_LINE
COLLAR_ATTACH
COLLAR_OUTER
CUFF_ATTACH
CUFF_END
CUFF_FOLD
SLEEVE_CAP_FRONT
SLEEVE_CAP_BACK
SLEEVE_BODY
```

필수 decomposition:

```text
edge count coverage          1.0
unknown owner edge           0
component-level p50/p95/p99
boundary-level p50/p95/p99
area ratio
triangle angle distortion
worst-edge ledger
```

### 6.1 CP4-R2 admission gate

Seam-critical boundary:

```text
arc-length error            ≤ 3%
symmetric_ratio p95         ≤ 1.10
symmetric_ratio p99         ≤ 1.20
```

Sleeve-cap / collar / cuff interior:

```text
symmetric_ratio p95         ≤ 1.20
symmetric_ratio p99         ≤ 1.50
```

General component interior:

```text
symmetric_ratio p95         ≤ 1.25
symmetric_ratio p99         ≤ 1.60
```

Hard HOLD:

```text
unowned edge                > 0
non-design edge ratio       > 2.50
boundary fold-over          > 0
triangle inversion          > 0
```

Preferred quality target는 admission gate보다 엄격하게 운영한다.

```text
critical boundary p95/p99   ≤ 1.05 / 1.10
component interior p95/p99  ≤ 1.15 / 1.30
```

## 7. Isometric arrangement repair

수정 허용 owner:

```text
sleeve-cap front/back arrangement
collar attach / outer arrangement
cuff attach / end / fold arrangement
```

수정 금지:

```text
CP3 source pattern
CP3 parameter set
CP3 interface semantics
accepted CP4-R1 predecessor
final settled vertex 수기 이동
```

Repair 순서:

```text
worst-edge localization
→ owner component / boundary 결정
→ local chart parameterization 수정
→ boundary constraints 고정
→ interior isometric relaxation
→ component-local rebuild
→ Warp resettle
→ before/after metric comparison
```

## 8. Direct visual art review

자동 화면 유효성 검사만으로 `visual_review=PASS`를 발행하지 않는다.

시각 receipt의 상태:

```text
AUTOMATED_EVIDENCE_READY
DIRECT_REVIEW_PASS
DIRECT_REVIEW_FAIL
```

제품 acceptance에는 `DIRECT_REVIEW_PASS`가 필요하다.

필수 Godot 원본 PNG:

```text
front
left
right
back
front_three_quarter
back_three_quarter
garment_only
shoulder_cap_detail
underarm_detail
neck_collar_detail
wireframe_front
seam_notch_overlay
```

Before/after 조건:

```text
same camera transform
same projection
same viewport size
same material override
same light / exposure
same body visibility
```

직접 확인할 결함:

```text
floating component
sleeve-cap collapse
underarm pinch / twist
collar float / overlap
cuff inversion
unowned loose triangle
non-design open boundary
body penetration
normal inversion
left/right unintended asymmetry
non-garment silhouette
```

## 9. Product contracts

```text
MetricDecompositionReceipt/1
IsometricArrangementRepairReceipt/1
DirectGLBProductReceipt/1
GLTFValidationReceipt/1
GodotVisualEvidenceReceipt/2
DirectArtReviewReceipt/1
BlenderFreeProductAcceptanceReceipt/1
```

최종 acceptance:

```text
technical_pass
AND metric_fidelity_pass
AND gltf_validation_pass
AND godot_consumer_pass
AND direct_art_review == PASS
```

## 10. Optional Blender lane

Blender는 다음 요청에서만 허용한다.

```text
manual sculpt
manual topology edit
artist-authored UV
artist-authored corrective
```

이 lane의 결과는 다음 계약을 요구한다.

```text
GUIDED_ARTIST_OVERRIDE
before / after product fingerprint
edited component ownership
source-of-truth disposition
manual change receipt
Godot requalification
```

Blender가 설치되지 않았다는 이유로 자동 제품 pipeline이 차단되어서는 안 된다.

## 11. Packaging / diet

REV1 신규 제품의 기본 전달물:

```text
source
updated design / roadmap
pattern and arrangement receipts
Warp qualification
GLB
GLTF validation report
Godot project scripts
12-view PNG
contact sheet
licenses / upstream locks
manifest
```

제외:

```text
new .blend files
Blender runtime / archive
Blender cache
intermediate renderer cache
nested archive
duplicate GLB
failed generations
```

CP4-R1의 기존 `.blend`는 immutable predecessor evidence이므로 현재 통합본에서는 보존한다. CP4-R2 이후 새 `.blend`를 추가하지 않는다.

## 12. Migration rule

```text
CP4-R1
→ accepted historical predecessor

CP4-R2-R1-REV1
→ first Blender-free metric-fidelity product

CP5
→ straight robe uses Blender-free pipeline only

CP6
→ R1B rig / corrective / secondary-motion rebind to direct GLB

CP7
→ feature-aware LOD and Godot terminal closeout
```
