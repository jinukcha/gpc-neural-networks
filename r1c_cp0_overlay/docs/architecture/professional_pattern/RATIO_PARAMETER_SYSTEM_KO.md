# Ratio Parameter System

CP1에서 구현할 parameter mode:

```text
ABSOLUTE
RELATIVE
AUTO_DERIVED
```

Reference kind:

```text
BODY_RELATIVE
BLOCK_RELATIVE
COMPONENT_RELATIVE
BOUNDARY_RELATIVE
MATERIAL_RELATIVE
```

모든 relative parameter는 reference identity, ratio, physical bounds, source revision을 가진다.

예:

```text
sleeve_length
= clamp(body.arm_length × ratio, minimum_m, maximum_m)
```

`0.8`과 같은 무소속 비율은 유효하지 않다. CP1은 dependency DAG, cycle rejection, bounded evaluation과 immutable resolution receipt를 구현하며 geometry는 생성하지 않는다.
