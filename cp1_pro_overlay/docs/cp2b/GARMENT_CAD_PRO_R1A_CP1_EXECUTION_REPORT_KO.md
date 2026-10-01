# GARMENT-CAD-PRO-R1A / CP1 실행 보고서

## 범위

CP2 튜닉 M의 POM·landmark authority를 첫 `PatternDocument/1`로 migration하고, exact curve, expression DAG, hard/soft constraint, atomic edit, undo/redo, save/reopen을 구현한다.

## Terminal decision

실행 산출물 `build/pattern_cad_cp1/cp1_receipt.json`과 `PROFESSIONAL_CHECKS.json`을 정본으로 사용한다.

```text
triangulation        NOT EXECUTED
Warp simulation      NOT EXECUTED
mesh scaling         FORBIDDEN
```

## Recovery disclosure

현재 대화의 로컬 실행면은 CP0 ZIP을 열기 전에 `ClientError`를 반환했고 GitHub에도 CP0 byte copy가 없었다. 두 경로 실패 후 CP3 sizing authority에서 CP1에 필요한 Anthropometry V2 입력을 제한 재구현했다. 따라서 CP0 byte-exact continuity를 주장하지 않으며, sizing·drape·기존 tunic build owner의 전후 tree hash를 별도로 고정한다.
