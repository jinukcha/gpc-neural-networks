# CP2B Warp backend 정본

## Authority

CP2B는 Blender Cloth가 아니라 프로젝트 소유 XPBD garment backend를 사용한다.
Upstream `warp-lang==1.17.0`은 array/kernel/device runtime으로만 사용하고,
deprecated `warp.sim` 또는 비상업 GarmentCode Warp fork는 사용하지 않는다.

```text
human-pattern 2D block
→ provider seamline
→ avatar-aware static input
→ WarpGarmentModelPackage
→ WarpSimulationState
→ final_simulation_mesh.npz
→ GLB
→ Godot read-only consumer
```

## CP1 static ownership

- four separate panel topologies; topology weld 금지
- dual triangle area를 3개 정점에 보존적으로 분배
- UV `v`를 warp, UV `u`를 weft 방향으로 사용
- triangle UV inverse와 rest 3D derivative를 보존
- manifold interior edge마다 양면·반대 정점·rest dihedral을 소유
- 8 named seams / 273 pair를 distance constraint input으로 소유
- 52 shoulder vertices를 explicit target에 연결
- body/self-contact, friction, substep, frame state는 CP2 이후 소유

## Recovery note

CP0 대화 첨부는 현재 실행면에서 읽을 수 없고 연결 저장소에도 동일 바이트가 없어,
CP1은 수락된 R2/R3 정수·패턴 계약을 만족하도록 필요한 입력만 재구현한다. 이전
procedural R4 recovery mesh는 정점·삼각형·seam·attachment 수가 달라 사용하지 않는다.
이 CP1 결과는 원본 CP0 `native_input_mesh.npz`와 byte-exact하다고 주장하지 않는다.

## Roadmap

1. CP1: static model authority.
2. CP2: body/self-contact, friction, adaptive substep, atomic checkpoint/resume.
3. CP3: frame 1–180 및 마지막 20-frame qualification.
4. CP4: canonical mesh GLB parity, Godot reopen, 원본 6뷰, closeout.
