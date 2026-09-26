# Provider와 project-owned adapter 경계

## 1. 단일 primary writer

FORTIFICATION hard CAD의 primary provider는 `build123d` 하나다. CadQuery는 비교·회귀·API 참고용이며 동일 asset을 두 writer가 병렬로 authoritative 생성하지 않는다.

## 2. Adapter 원칙

제품 코드는 build123d object를 public contract로 노출하지 않는다.

```text
project FortificationProgram
  → project-owned adapter request
  → build123d/OCP execution
  → neutral CAD receipt + geometry buffers + optional STEP/BREP
  → project validation
```

Adapter가 소유하는 것:

```text
unit/frame conversion
explicit tolerance injection
stable operation labels
exception/status normalization
semantic part history
neutral indexed buffer extraction
optional STEP/BREP stored copy
runtime/source/provenance receipt
```

Adapter가 소유하지 않는 것:

```text
city placement
Definition meaning
final stable object identity
LOD/HLOD
Godot scene structure
```

## 3. Runtime admission

build123d source snapshot만으로 실행 가능하다고 주장하지 않는다. R0A-CP0에서 다음 complete offline closure가 필요하다.

```text
Python ABI and platform
build123d exact source/wheel
cadquery-ocp-novtk/OCP exact wheel
all declared runtime dependencies and transitive wheels
license/provenance for all admitted artifacts
fresh isolated materialization
pip check
import probe
extrude/sweep/loft/boolean/tessellate/export capability probes
cold A/B deterministic fixture
```

Global pip install, network-on-run, silent dependency substitution은 금지한다.

## 4. Optional Manifold

Manifold3D 3.5.3은 기존 WORLD-FIELD qualified boundary를 재사용한다. 역할은 bounded complex CSG/mesh closure이며 build123d를 대체하는 second authoring authority가 아니다.

```text
primary simple/normal path    build123d/OCP
optional bounded mesh CSG     Manifold3D exact adapter
collision decomposition       CoACD — 별도 RUNTIME 소유
```

## 5. Reference-only sources

### CadQuery

- workplane/sketch/assembly API와 OCP 결과 비교
- shell/fillet/chamfer/sweep/loft/boolean 회귀 아이디어
- 제품 import 금지, second writer 금지

### Building Tools

- arch/door/stairs/railing/roof grammar와 사용자 parameter 분해 참고
- Blender/BMesh 결과를 canonical 제품 경로로 사용하지 않음
- MIT source를 그대로 product module로 복사하지 않고 알고리즘·계약만 검토

### 제외

TopologicPy, IfcOpenShell, Sverchok, ProcFunc, Infinigen은 이번 FORTIFICATION CAD 설계의 필수 source가 아니다. 필요한 시점에 별도 child design에서 사용한다.
