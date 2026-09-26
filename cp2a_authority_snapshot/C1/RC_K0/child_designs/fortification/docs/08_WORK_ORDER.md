# 첫 구현 작업 지시 — RC-FORT-R0A-CP0

## 작업명

```text
ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP0
— EXACT BUILD123D/OCP RUNTIME ADMISSION,
  SELECTED SOURCE FENCE
  & PROVIDER-CAPABILITY PREFLIGHT
```

## 고정 입력

- parent design packet: `RCF_D0_full.zip`
- build123d source commit: `e22d34dae17111e5b9fdb361317e055d0daae466`
- build123d source ZIP SHA-256: `77fe1b25fc56e7d59a89b95ae3c81af2bd0acad1814669296d786291b7f325ca`
- CadQuery reference commit: `c11b3f93278bb25053991f7071451f71664c90fc`
- Building Tools reference commit: `0216534645b15b3f8aac0bd1d7e6674e2972b9b3`

## 구현 범위

1. actual target Python ABI를 고정한다.
2. build123d/OCP와 모든 Linux-applicable dependency wheel을 exact hash로 lock한다.
3. 제품 tree 밖 fresh destination에 offline materialize한다.
4. `pip check`, import, extrude, sweep, loft, boolean, tessellate, STEP/BREP export probes를 실행한다.
5. independent cold A/B fixture 결과와 provider/runtime receipt를 보존한다.
6. source/runtime/adoption 상태를 분리한다.

## 비범위

```text
Fortification product adapter implementation
wall geometry authoring
Godot import
Manifold fallback execution
full project regression
```

## 보존

긴 runtime 검증 전에 동일 최초 기준본 대비 다음을 만든다.

```text
RCF_A0_src.zip
RCF_A0_patch.zip
RCF_A0_full.zip
```

실패·timeout 시 작업 directory와 마지막 checkpoint를 삭제·초기화·rollback하지 않는다.
