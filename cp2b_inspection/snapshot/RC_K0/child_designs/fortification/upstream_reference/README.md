# Selected upstream reference source

이 디렉터리는 FORTIFICATION CAD 상세 설계와 후속 adapter 구현을 검토하기 위한 selected source copy다.

```text
build123d       primary API/topology/tessellation/export reference
cadquery        OCP comparison oracle
building_tools  architectural grammar reference
```

금지:

- product source에서 이 디렉터리를 직접 import
- upstream object/face index를 stable identity로 사용
- CadQuery를 parallel authoritative writer로 승격
- Building Tools의 Blender/BMesh runtime을 제품 필수 경로로 승격

원본 archive identity, license와 각 선택 파일의 SHA-256은 `../provenance/` 및 `../data/OSS_SOURCE_USAGE.csv`에 있다.
