# R0A-CP1 — project-owned build123d provider adapter

이 디렉터리는 build123d/OCP를 제품 계약 뒤에 격리하는 얇은 project-owned adapter를 보존한다.

```text
public input       neutral JSON/dataclass
provider interior  build123d/OCP
public result      neutral indexed mesh + stored-copy refs + typed status
receipt            source/runtime/unit/frame/tolerance/output digests
```

현재 구현 범위는 `PROFILE_EXTRUSION` 한 종류다. 이는 CP2 straight wall pilot이 사용할 최소 형상 경계이며, 도시 배치·성벽 의미·LOD·Godot scene은 소유하지 않는다.
