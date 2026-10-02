# Completion and Repair Authority

## 분류

```text
AUTO_COMPLETION
BOUNDED_REPAIR
TOPOLOGY_REDESIGN_REQUIRED
```

### Auto completion

- mirror-authoritative component 생성
- 누락 notch·grainline·bounded facing 파생
- identity-preserving metadata completion

### Bounded repair

- curve tangent continuity
- normalized-arc notch 재투영
- bounded sleeve-cap ease 재분배
- mirror drift와 boundary orientation 수정

### Topology redesign

- panel 추가·삭제
- set-in ↔ raglan sleeve
- one-piece ↔ two-piece sleeve
- side gore 추가
- collar family 또는 closure topology 변경

Topology redesign은 SAFE_AUTO로 commit하지 않으며 `ALTERNATE_COMPONENT_REQUIRED`, `TOPOLOGY_CHANGE_REQUIRED`, `GUIDED_DECISION_REQUIRED`, `HOLD` 중 하나를 발행한다.

## 원자성

```text
diagnosis
→ RepairPlan preview
→ clone
→ source-pattern mutation
→ dependency resolve
→ hard constraint validation
→ commit 또는 전체 폐기
```

최종 3D mesh의 shrinkwrap·snap·weld·manual vertex pull은 completion/repair로 인정하지 않는다.
