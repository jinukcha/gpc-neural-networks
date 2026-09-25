# ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP1 완료 보고서

```text
source preservation          PASS
functional qualification     PASS
stage completion             R0A_CP1_COMPLETE / CLOSED
next                          R0A-CP2
```

## 구현

- `PROFILE_EXTRUSION` neutral request와 result dataclass/JSON 계약
- 프로젝트 `METER`, `+Y up`, `-Z forward` → provider `MILLIMETER`, `+Z up` 변환
- determinant +1 orientation과 1000:1 length scale
- model/tessellation tolerance domain
- typed reject/fail normalization과 required failure 시 geometry 미게시
- provider tessellation을 canonical neutral indexed mesh로 변환
- STEP/BREP memory export, byte hash, reopen volume 검증
- source/runtime/contract/output digest가 포함된 `CADProviderReceipt`
- public annotation과 초기 package import에서 build123d/OCP type 비노출

## 검증

```text
fresh CP0 runtime materialization  PASS
qualification fixture              PASS
negative fixtures                   5 / 5 PASS
unittest                            4 / 4 PASS
CLI smoke                           PASS
public API audit                    PASS
cold A/B files                      5 / 5 BYTE-IDENTICAL
neutral mesh                        8 vertices / 12 triangles
volume                              48.0 m³
STEP/BREP reopen                    48.0 m³ / 48.0 m³
```

## 발견·수정 기록

1. 최초 extrusion call이 `amount` 없이 `dir`만 전달되어 provider가 거부했다. 실패 결과를 보존하고 explicit amount+unit direction으로 수정했다.
2. project meter 값을 provider numeric meter로 직접 사용하고 STEP `Unit.M`으로 내보내면 build123d reopen이 millimeter 숫자로 변환됐다. adapter 경계를 meter→millimeter 1000:1로 명시해 수정했다.
3. 초기 public-type 검사가 provider 이름과 neutral field name의 `build123d` 문자열까지 type leakage로 오판했다. 실제 resolved annotation module을 검사하도록 validator를 교정했다.

## 비범위

straight wall 의미 부품, socket, source surface coverage, LOD, Godot는 CP2 이후다.
