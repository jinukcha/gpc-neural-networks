# RC_K0 — ROYAL-CAPITAL-KERNEL 전체 설계·유한 로드맵

## 읽기 순서

1. `docs/00_STATUS.md`
2. `docs/01_OVERALL_DESIGN.md`
3. `docs/02_KERNEL_CONTRACT.md`
4. `docs/03_DOMAIN_BOUNDARIES.md`
5. `docs/04_OSS_PLAN.md`
6. `docs/05_WORKFLOW.md`
7. `docs/06_ROADMAP.md`
8. `docs/07_CHILD_DESIGN_PACKETS.md`
9. `docs/08_ACCEPTANCE.md`

기계 판독 자료는 `data/`, 예제 계약은 `examples/`, JSON Schema는 `schemas/`에 있다.

## 고정 미술 기준

`references/concept/royal_capital_concept.png`

- SHA-256: `ecde17820fa243674deb453aba605693750e9710f3a47e6492c6057e2a15f0ad`
- 크기: `4163381` bytes
- 해상도: `1672 × 941`
- 역할: 도시의 공간 위계·실루엣·지구 관계를 정하는 미술 기준 fixture
- 비권위 항목: 이미지 안의 영문 라벨, 가상 축척자, 정확한 치수, 인구, 구조 안전성

## 핵심 판정

이 프로젝트는 새 도시 프레임워크를 만들지 않는다. 기존 WorldFeatureDefinition, PAP Definition/Instance, DefinitionCatalogSnapshot, ArtifactIndex/ArtifactStore, WORLD-FIELD 건물·배치·충돌 계약과 MEGASTRUCTURE cooked-asset 경계를 재사용한다.

후속 구현 체크포인트는 `RCF_D0_full.zip`을 동일 최초 구현 기준본으로 유지해 누적 패치를 만든다. 첫 CAD child의 exact build123d/OCP runtime admission은 로컬 independent cold A/B까지 통과했다.

## 현재 구현 체크포인트

```text
RC-FORT-R0-DESIGN  COMPLETE
RC-FORT-R0A-CP0    COMPLETE / CLOSED
RC-FORT-R0A-CP1    COMPLETE / CLOSED
provider adapter   NEUTRAL CONTRACT PASS
cold A/B           BYTE-IDENTICAL PASS
next               ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP2 — STRAIGHT WALL-SPAN PILOT
```
