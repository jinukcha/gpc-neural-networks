# 검증·실패 계약

## 1. 설계 단계 검증

```text
required docs/data/schema/example present
JSON/CSV parse
schema examples validate
selected upstream bytes equal exact archives
license/provenance coverage
no cache/symlink/unsafe path
parent RC_K0 + patch payload = final tree
```

## 2. 구현 단계 focused validation

### Contract

- ID/revision/digest uniqueness
- Definition/Instance/Program 분리
- exact reservation/catalog/provider fence
- shuffled input stability
- budget and required/optional semantics

### Geometry

- non-empty finite vertices/indices
- closed/watertight where solid is required
- positive bounded volume
- bounds and foundation contact envelope
- no unexpected component fragmentation
- no invalid self-intersection evidence
- required portal clearance
- required wall-walk continuity
- tower/gate/span socket alignment
- semantic surface and source coverage 100%

### Determinism

- same clean input/runtime two or more runs byte-identical canonical outputs
- unrelated optional module 추가가 existing stable IDs를 변경하지 않음
- source list/input ordering 변화가 결과를 변경하지 않음

### Product integration

- P0BB source coverage, cook and LOD recipe
- static/dynamic collision derivation
- navigation input and wall/gate traversal
- Godot stored-copy import/reopen

## 3. Typed failures

| 상태 | 의미 |
|---|---|
| `SOURCE_REFERENCE_MISSING` | exact source/license/provenance 없음 |
| `CAD_RUNTIME_UNAVAILABLE` | build123d/OCP runtime 미반입 |
| `CAD_RUNTIME_VERSION_MISMATCH` | lock과 실제 runtime 불일치 |
| `STALE_INPUT_REFERENCE` | catalog/reservation/terrain digest 불일치 |
| `INVALID_RING_WINDING` | inside/outside 결정 불가 |
| `CENTERLINE_SELF_INTERSECTION` | canonical centerline 자기 교차 |
| `ZERO_LENGTH_SPAN` | 유효 길이 없음 |
| `GRADE_OUT_OF_DOMAIN` | 허용 지형 경사 초과 |
| `CURVATURE_OUT_OF_DOMAIN` | sweep 허용 곡률 초과 |
| `GATE_CLEARANCE_UNSATISFIED` | road/portal clear envelope 불충족 |
| `TOWER_OVERLAP` | tower/module envelope 충돌 |
| `BOOLEAN_FAILED` | required CSG 실패 |
| `SEMANTIC_COVERAGE_INCOMPLETE` | surface/triangle source mapping 결손 |
| `SOCKET_GEOMETRY_MISMATCH` | socket frame과 실제 형상 불일치 |
| `WALL_WALK_DISCONNECTED` | required traversal graph 단절 |
| `GEOMETRY_BUDGET_EXCEEDED` | vertex/triangle/solid/operation budget 초과 |
| `NONDETERMINISTIC_RESULT` | clean repeat canonical bytes 불일치 |
| `OPTIONAL_ORNAMENT_DROPPED` | optional만 fail-open; required assembly 사용 가능 |

Required span/tower/gate 실패 시 accepted partial ring을 만들지 않는다.

## 4. 판정 분리

```text
source preservation
runtime admission
functional geometry qualification
Godot product qualification
human art review
stage completion
```

하나의 PASS로 합치지 않는다.
