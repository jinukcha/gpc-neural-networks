# R0B-CP3 implementation

## Terrain-stepped route

A horizontal plan axis is divided into explicit terrace intervals. Every interval owns five bounded profile-extrusion units:

```text
foundation
wall_body
wall_walk
inner_parapet
outer_parapet
```

Each unit is placed at the terrain terrace elevation. Step boundaries publish lower/upper foundation and wall-walk sockets. Stair or ramp realization is not fabricated in this checkpoint.

## Retaining route

The retaining fixture uses an asymmetric L-shaped foundation profile. The lower outside terrain, upper inside terrain and bearing stratum are explicit. The upper wall profile is referenced to the inside terrain elevation.

## Foundation interface

The Producer records:

```text
terrain source ID / revision / digest
station interval
contact elevation
foundation base and top elevation
embedment depth
contact area
unsupported gap
step or retaining transition
```

Terrain is never silently filled, cut or flattened.

## Evidence

Contact is sampled at a fixed station spacing. Every required sample must be within the declared gap envelope. Grade evidence records overall longitudinal grade, terrace step heights, retained height and equivalent cross-slope ratio.

## Deferred

```text
terrain editing
wall-walk stair/ramp geometry
terrain collision
navigation
source-surface coverage
mixed-span assembly
Godot product
```
