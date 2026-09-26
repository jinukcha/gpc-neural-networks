# ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP0 closeout

## Decision

```text
source preservation          PASS
selected source fence        PASS
exact 58-wheel intake        PASS
fresh local materialization  PASS ×2
local pip check              PASS ×2
native capability            9 / 9 PASS ×2
cold A/B canonical           PASS
global install               false
global environment unchanged true
stage completion             R0A_CP0_COMPLETE / CLOSED
R0A-CP1 start                ALLOWED
```

## Exact runtime identity

```text
Python       CPython 3.13.5 / cp313 / Linux x86-64
build123d    0.13.1.dev12+ge22d34dae
source       e22d34dae17111e5b9fdb361317e055d0daae466
OCP          cadquery-ocp-novtk 8.0.1.0.0
OCP SHA-256  1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f
wheel count  58
```

## Local independent A/B

```text
runtime A    ADMITTED / pip check PASS / 9 of 9 PASS
runtime B    ADMITTED / pip check PASS / 9 of 9 PASS
canonical    e28815485ffc3ca0ad51b52daf43badc5e9a0f92e47951490968e8190cbdaaab — byte-identical
volume       240.0
vertices     24
triangles    12
BREP SHA     e42b6afce98ebb9c854b3670c5d6d44237c0411d7b49a944e6168d6b0985cce8
STEP norm    5bf3cefd4f8555d61c55cfe7a8e520e1ed979367eb74d4d700bc535a713453b8
```

The first B materialization was interrupted by the tool timeout after creating a partial venv. It was moved to `failed_runtime_B_timeout`, recorded, and never reused. Retry B used a fresh destination.

## STEP determinism boundary

The two raw STEP files are not byte-identical because the exporter writes the output file name and current timestamp into `FILE_NAME`. Both raw hashes are preserved. The canonical digest normalizes only those fields. BREP raw bytes are identical, both STEP/BREP files reopen to the same volume, and all geometry checks match.

## Scope boundary

This closeout admits the exact CAD runtime and primitive provider capabilities only. It does not implement the project-owned adapter, wall geometry, semantic source coverage, Godot import, Manifold fallback, or product release.

## Official next

```text
ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP1 — PROJECT-OWNED PROVIDER ADAPTER, NEUTRAL REQUEST/RESULT TYPES, UNIT/FRAME/TOLERANCE CONTRACT & CAD PROVIDER RECEIPT
```
## Focused closeout validation

```text
CP0 validator                 64 / 64 PASS
source archive fence          3 / 3 PASS
selected source bytes       105 / 105 PASS
source API surface            7 / 7 PASS
wheel delivery intake        15 / 15 PASS
JSON parse                    56 / 56 PASS
CSV parse                     21 / 21 PASS
closed-world file registry   367 / 367 PASS
cache / pyc / .godot          0
```

Two validation-tool defects were preserved and repaired without rolling back the admitted runtime or source tree: the delivery evidence was first placed under the self-contained runtime delivery root, and a path-substring hygiene false positive was replaced by directory-component inspection.

