# GARMENT-CAD-PRO-R1C / CP2 — Pattern Component Library

## 목적

CP2는 CP0의 modular contract와 CP1의 `ResolvedParameterSet/1`을 변경하지 않고 실제 exact 2D component geometry와 물리 길이 기반 interface solver를 발행한다. Triangulation, 3D arrangement, cloth simulation은 수행하지 않는다.

## 소유 커널

```text
pattern_geometry/
├── model.py
├── curve.py
├── inputs.py
├── bodice.py
├── sleeve.py
├── accessories.py
├── library.py
├── interface_solver.py
├── assembler.py
├── fixtures.py
├── contracts.py
└── evidence.py
```

## Component authority

첫 라이브러리는 다음 component를 소유한다.

```text
BODICE_FRONT_BASIC / revision 2
BODICE_BACK_BASIC / revision 2
SET_IN_SLEEVE_BASIC / revision 2
COLLAR_STAND_BASIC / revision 1
CUFF_STRAIGHT_BASIC / revision 1
SIDE_GORE_BASIC / revision 1
```

각 geometry는 line 또는 quadratic/cubic Bézier control point, semantic boundary, notch의 physical arc length와 normalized arc position, landmark, internal construction line과 CP1 parameter-set hash를 보존한다.

## Set-in sleeve

Sleeve cap은 front와 back을 독립 곡선으로 제도한다. Front/back armscye의 실제 arc length와 CP1 `cap_ease`를 입력으로 사용해 각 cap half-width를 deterministic root solve한다. Shoulder, front pitch, back pitch notch는 semantic role과 normalized arc correspondence로 연결한다.

## Interface solver

Interface solver는 contract admission 이후 다음을 검사한다.

```text
physical boundary length
bounded ease / gather ratio
orientation relation
semantic notch correspondence
seam allowance policy
turn-of-cloth disposition
```

Notch mismatch fixture는 `REJECTED_ATOMIC`이며 assembled package를 발행하지 않는다.

## Assembly product

`AssembledPatternPackage/1`은 7개 component instance와 16개 seam interface를 소유한다. 모든 product는 exact 2D authority만 참조하며 3D primitive fallback을 금지한다.

## 다음 단계 경계

CP3는 이 geometry library와 assembly package를 immutable input으로 사용해 completion diagnosis, bounded source-pattern repair와 atomic transaction을 구현한다.
