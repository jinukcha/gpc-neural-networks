# GARMENT-CAD-PRO-R1C — Blender-free REV1 Roadmap

## Authority

```text
immutable predecessor
GARMENT-CAD-PRO-R1C / CP4-R1

accepted product
R1C_CP4_R1_SET_IN_SLEEVE_TUNIC

revision
BLENDER_FREE_REV1
```

이 문서는 R1C의 남은 CP4-R2·CP5·CP6·CP7을 Blender-free 제품 경로로 전환하는 authoritative roadmap이다.

## CP4-R2-R1-REV1

### 이름

```text
COMPONENT–BOUNDARY METRIC DECOMPOSITION
/ ISOMETRIC ARRANGEMENT REPAIR
/ DIRECT GLB PRODUCT
/ GODOT DIRECT ART AUDIT
```

### 입력

```text
CP4-R1 accepted product and receipts
CP3 repaired pattern snapshot
CP3 assembled pattern package
WOOL_TWILL_MEDIUM_REFERENCE
```

### 구현 범위

1. 4,014 structural edge를 component·boundary·interior owner로 전부 분류한다.
2. raw ratio, symmetric ratio, log strain, area ratio와 angle distortion을 발행한다.
3. sleeve-cap front/back, collar attach/outer, cuff attach/end/fold의 worst edge만 국소 수리한다.
4. CP3 source pattern과 interface semantics는 수정하지 않는다.
5. Warp resettle을 동일 wool-twill material에서 실행한다.
6. 기존 Python direct GLB writer를 R1C materialized product에 적용한다.
7. exact pinned Khronos glTF Validator를 실행한다.
8. Godot 4.7.2에서 fresh import/reopen과 동일 카메라 12뷰를 생성한다.
9. 자동 evidence와 분리된 direct art audit receipt를 발행한다.

### 금지

```text
Blender runtime
.blend 생성
Blender export
Blender visual gate
final vertex manual repair
pattern rewrite
mesh scaling
```

### Terminal gate

```text
edge owner coverage              1.0
unknown edges                    0
critical boundary fidelity       PASS
component interior fidelity      PASS
technical qualification          PASS
glTF validation errors           0
Godot fresh import / reopen      PASS
direct art review                PASS
post-settle vertex repair        0
```

### 결과

```text
CP4_R2_ACCEPTED_BLENDER_FREE_TUNIC
```

통과하지 못하면 CP5로 진행하지 않고 owner-local metric defect만 재수리한다.

---

## CP5 — Straight Robe Blender-free Rebase

### 목표

Full-length front/back panel, side gore, center-front opening과 straight sleeve를 exact pattern authority에서 생성한다.

### 필수 component

```text
front_left_full_length
front_right_full_length
back_left_full_length
back_right_full_length
side_gore_left
side_gore_right
sleeve_left
sleeve_right
front_facing_left/right
optional collar
```

### 제품 경로

```text
exact pattern
→ component-aware triangulation
→ metric-aware arrangement
→ Warp settling
→ direct GLB
→ glTF Validator
→ Godot visual audit
```

Blender는 포함하지 않는다.

### Terminal gate

```text
continuous full-length panel flow
center-front overlap / closure continuity
side-gore continuity
hem continuity
walk / turn / sit clearance
metric fidelity
technical qualification
Godot direct art review
```

---

## CP6 — R1B Runtime Rebind

CP4-R2와 CP5의 direct GLB 제품을 R1B runtime 계약에 연결한다.

```text
body-to-garment skin transfer
shoulder / underarm / elbow corrective
robe hem secondary motion
outfit registry
body occlusion
atomic equip / character swap
Godot 4.7.2 runtime qualification
```

Blender scene 또는 Blender-exported skin을 authority로 사용하지 않는다.

---

## CP7 — Rig-aware LOD / Terminal Closeout

### LOD compiler

```text
component and feature ownership
seam-critical boundary locks
corrective-preserving attributes
secondary-motion identity
meshoptimizer optional pinned tool
```

### Evidence

```text
actual distance near / mid / far
same screen coverage LOD0 / LOD1 / LOD2
motion board
wireframe / seam overlay
metric fidelity board
```

### R1C terminal decision

```text
GARMENT_CAD_PRO_R1C_COMPLETE
```

다음이 모두 성립해야 한다.

```text
pattern-driven sleeved tunic accepted
pattern-driven straight robe accepted
Blender-free direct GLB pipeline accepted
R1B runtime rebind accepted
rig-aware LOD accepted
Godot terminal visual audit accepted
```
