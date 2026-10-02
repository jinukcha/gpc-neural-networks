# GARMENT-CAD-PRO-R1C

## 현재 기준

- Immutable predecessor: `GARMENT-CAD-PRO-R1B / CP6`
- Active checkpoint: `R1C / CP0`
- CP0 terminal decision: `CP0_COMPLETE_VISUAL_AND_MODULAR_FOUNDATION`
- Geometry, triangulation, simulation, Godot execution: CP0에서 수행하지 않음

## 정본 경로

```text
build/r1c_cp0/
contracts/r1c_cp0/
source/wuxia_garment_oss/pattern_components/
source/wuxia_garment_oss/pattern_assembly/
source/wuxia_garment_oss/visual_acceptance/
docs/architecture/professional_pattern/
docs/roadmap/GARMENT_CAD_PRO_R1C_ROADMAP_KO.md
```

## 제품 생성 순서

```text
PatternComponentDefinition
→ GarmentAssemblyRecipe
→ ComponentInterfaceSpec admission
→ ratio parameter resolution (CP1)
→ completion / bounded repair (CP3)
→ pattern-driven 3D materialization
→ direct visual acceptance
→ R1B rig/runtime rebind
```

3D primitive를 pattern component 정본으로 사용하거나, 완성 mesh를 snap·weld·shrinkwrap하여 source-pattern 결함을 은폐하는 경로는 금지한다.

## 다음 구현

`GARMENT-CAD-PRO-R1C / CP1 — ABSOLUTE / RELATIVE / AUTO-DERIVED RATIO PARAMETER ENGINE`
