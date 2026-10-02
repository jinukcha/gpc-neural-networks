# R1C CP1 Ratio Parameter Engine

## 소유 범위

CP1은 패턴 geometry를 생성하지 않고 설계 parameter의 해석과 provenance만 소유한다.

```text
ABSOLUTE
RELATIVE
AUTO_DERIVED
```

지원 reference scope:

```text
BODY_RELATIVE
BLOCK_RELATIVE
COMPONENT_RELATIVE
BOUNDARY_RELATIVE
MATERIAL_RELATIVE
```

## 평가 순서

```text
definition validation
→ reference admission
→ dependency DAG
→ cycle rejection
→ deterministic topological evaluation
→ physical quantity validation
→ bounds disposition
→ immutable publication
```

`RELATIVE`는 stable reference path와 dimensionless ratio를 요구한다. `AUTO_DERIVED`는 제한된 AST만 사용하며 Python evaluation, 파일 접근과 임의 함수 호출을 허용하지 않는다.

## Bound disposition

```text
REJECT
SAFE_CLAMP
ALTERNATE_COMPONENT_REQUIRED
```

`SAFE_CLAMP`는 requested/published value, delta와 owner를 receipt에 남긴다. Topology 범위를 벗어난 parameter는 자동 왜곡하지 않고 `ALTERNATE_COMPONENT_REQUIRED`로 거부한다.

## 원자성

Cycle, missing reference, unit·quantity mismatch, non-finite ratio와 hard bound failure에서 `ResolvedParameterSet`은 발행되지 않는다.

```text
partial_publication_count = 0
```

## CP1 실행 경계

```text
pattern geometry    NOT EXECUTED
triangulation       NOT EXECUTED
simulation          NOT EXECUTED
Godot               NOT EXECUTED
```
