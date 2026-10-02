# Garment Family Extension Guide

새 garment family는 다음 ownership을 가져야 한다.

```text
garments/<family>/
├── recipe/
├── parameter_binding/
├── completion_policy/
├── materialization/
├── visual_acceptance/
└── rig_rebind/
```

공용 `pattern_components`에는 여러 family가 실제로 공유하는 exact pattern·construction definition만 둔다. Family별 silhouette, interface 선택, completion rule과 topology-change policy를 공용 helper로 끌어올리지 않는다.

새 family admission 체크:

1. required body/block measurements
2. component instance set
3. interface closure
4. absolute/relative/auto-derived parameters
5. completion and repair policy
6. materialization recipe
7. visual profile
8. R1B runtime rebind policy

Variant는 panel/interface topology가 동일하고 bounded parameter만 달라지는 경우에만 사용한다. Panel 수, sleeve family, closure topology, layer ownership이 바뀌면 새 topology variant 또는 새 family다.
