# R0B-CP4 implementation

## Mixed fixture

The closeout fixture contains a single required chain:

```text
straight span
→ miter join
→ curved span
→ bevel join
→ terrain-stepped span
→ retaining span
→ profile-transition join
```

Every module points to an accepted R0B result, neutral mesh, provider receipt, stored-copy index and socket contract by exact path and SHA-256. The assembly uses rigid transforms only; no global boolean is performed.

## Segmentation

Span intervals partition a canonical station domain. Join modules occupy exact boundary stations. Module and connection arrays are intentionally shuffled in the fixture; chain order, span order and stable keys are derived from IDs, source intervals, exact source digests and the dependency chain. Array order is never identity.

## Source coverage

Every source triangle is mapped to:

```text
module instance
source result and neutral-mesh digest
semantic part or terrain construction unit
source triangle index
assembly triangle index
project-owned geometric surface role
```

Surface roles are derived from the dominant triangle normal in the nearest project local frame. Provider face IDs and provider traversal order are not used.

## Deferred

```text
towers and battlement modules
collision and navigation
P0BB product cook
Godot import
terrain mutation
```

## Accepted result

```text
qualification PASS
focused validation 58/58 PASS
R0B COMPLETE / CLOSED
```
