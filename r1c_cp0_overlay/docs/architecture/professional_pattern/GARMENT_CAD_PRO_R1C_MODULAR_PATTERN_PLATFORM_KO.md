# GARMENT-CAD-PRO-R1C — Modular Pattern Platform

## 목표

R1C는 여러 의상 부위를 독립적으로 제도하고, 절대 치수와 비율 입력을 함께 사용하며, 미완성 상태를 source-pattern 단계에서 진단·완성·보정하는 전문 의류 플랫폼이다.

```text
Body / Block / Material authority
→ GarmentAssemblyRecipe
→ PatternComponentInstance
→ ratio parameter resolution
→ ComponentInterface admission
→ completion / bounded repair
→ exact assembled pattern
→ material-aware 3D rest topology
→ visual acceptance
→ R1B rig/runtime rebind
```

## 비목표

CP0에서는 다음을 수행하지 않는다.

- pattern geometry 생성 또는 변경
- triangulation
- cloth simulation
- skin transfer·corrective·secondary motion·LOD
- Godot import
- 기존 R1B 제품 수락 상태 변경

## Authority 경계

### Pattern component

Component는 독립 3D mesh가 아니라 다음을 소유한다.

```text
exact 2D curve boundary
landmark
construction feature
parameter port
interface port
layer role
topology-change policy
```

### Interface

Component 간 결합은 좌표 근접성이 아니라 `ComponentInterfaceSpec/1`으로만 허용한다.

```text
semantic family
orientation relation
length/ease policy
notch correspondence
seam allowance
turn-of-cloth
construction predecessor
```

### Visual acceptance

기술 수치가 통과하더라도 필수 화면과 visual gate가 실패하면 제품을 수락하지 않는다.

```text
product_acceptance = technical_pass AND visual_review == PASS
```

## 자동 처리 경계

```text
SAFE_AUTO
→ mirror, notch, metadata, bounded identity-preserving correction

GUIDED
→ preview와 영향 범위를 제시하고 명시적 commit

HOLD
→ topology 변경, component family 변경, 구조적 불완전
```

모든 수정은 pattern authority로 되돌아가야 하며 최종 3D vertex 후처리로 결함을 숨기지 않는다.
