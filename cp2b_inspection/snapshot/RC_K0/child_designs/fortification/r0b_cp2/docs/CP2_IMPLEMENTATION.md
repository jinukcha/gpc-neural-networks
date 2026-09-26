# R0B-CP2 implementation

## Geometry routes

```text
MITER
  incoming overlap section
  expanded bisector section at anchor
  outgoing overlap section

BEVEL
  incoming overlap section
  incoming setback cut
  outgoing setback cut
  outgoing overlap section

PROFILE_TRANSITION
  incoming profile section
  linearly interpolated profile at anchor
  outgoing profile section
```

All five wall parts remain separate solids. The join owns only the bounded overlap module and does not rewrite either source span.

## Socket contract

The join mirrors incoming/outgoing span, wall-walk and foundation endpoint sockets. The qualification evidence records position and basis-angle errors for six pairs. No socket identity is inferred from scene names or CAD face order.

## Deferred

```text
terrain grade and foundation steps
miter/bevel assembly into a full ring
source surface coverage for join geometry
towers and gates
collision/navigation/Godot product
```
