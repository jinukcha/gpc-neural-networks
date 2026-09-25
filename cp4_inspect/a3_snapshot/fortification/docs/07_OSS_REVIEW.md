# OSS 검토와 작업 키트 편입

## 결론

새로운 대형 OSS 다운로드는 필요하지 않다. Library에 이미 고정된 source snapshot 중 이번 FORTIFICATION CAD에 필요한 코드만 선별해 작업 키트에 포함했다.

| OSS | 역할 | 패킷 처리 |
|---|---|---|
| build123d | primary hard-CAD API·topology·tessellation·export 참고 | 관련 core/docs/examples/tests 선별 포함 |
| CadQuery | OCP workplane/sketch/assembly/boolean 비교 oracle | 관련 source/docs/examples/tests 선별 포함 |
| Building Tools | arch/door/stairs/railing/roof grammar 참고 | 관련 MIT source/tests 선별 포함 |
| Manifold3D 3.5.3 | optional bounded complex CSG | 기존 qualified project boundary 재사용; 중복 source 미포함 |
| Shapely / OR-Tools | upstream reservation/placement | NETWORK 소유; 이번 CAD source 미포함 |
| TopologicPy / IfcOpenShell | topology/IFC oracle | 이번 child 필수 아님 |
| Sverchok | GPL geometry reference | 불필요·직접 복사 금지; 미포함 |

## build123d 선택 이유

이번 selected copy는 다음 능력의 구현과 문서를 보존한다.

```text
BuildLine / BuildSketch / BuildPart
extrude / sweep / loft / revolve / thicken
fillet / chamfer / offset / split / mirror
solid/shell/face topology operations
joints and assembly concepts
tessellate / tessellate_with_uvs
STEP/BREP/STL/OBJ/glTF exporter boundary
```

하지만 source snapshot은 실행 runtime이 아니다. `cadquery-ocp-novtk`와 모든 dependency closure가 실제로 반입되기 전까지 `CAD_RUNTIME_UNAVAILABLE`이다.

## CadQuery 선택 이유

- build123d와 같은 OCP 계열에서 workplane·sketch·assembly·boolean 결과를 비교
- API 설계와 failure behavior 참고
- 동일 asset의 authoritative second writer로 사용하지 않음

## Building Tools 선택 이유

- arch, door, roof, stairs, railing의 parameter 분해와 geometry grammar 참고
- Blender/BMesh에 의존하므로 제품 runtime·SSOT로 사용하지 않음
- MIT license를 유지한 selected source reference만 포함

## 복사 정책

```text
selected upstream files are reference-only
product code must not import upstream_reference/
exact original relative path retained under each project root
license/NOTICE and archive receipt retained
per-file SHA-256 recorded
full upstream archives are not nested in the work kit
```

전체 archive는 Library의 exact source가 authority이며, 이 패킷은 필요한 선택본만 재현한다.
