# Pattern-driven Materialization Pipeline

실제 3D rest garment는 다음 경로에서만 생성한다.

```text
exact component curves
→ adaptive boundary sampling
→ interface-aware seam correspondence
→ constrained component triangulation
→ avatar-aware arrangement
→ seam/fold/layer constraints
→ calibrated material settling
→ technical qualification
→ visual acceptance
```

금지:

```text
joint-chain tube sleeve
기존 garment에 skirt primitive 부착
post-solver shrinkwrap
manual vertex pull
snap/weld로 seam 결함 은폐
완성 mesh scaling
arbitrary hole fill
```

수락된 rest topology가 생성된 이후에만 R1B의 skin transfer, local corrective, secondary motion, outfit compiler와 LOD를 다시 적용한다.
