# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — TOWER JOINS

## 1. Authority

```text
accepted parent                 C1.zip
accepted parent SHA-256         91cf2025f311772b0a137d3509efa0b1d65b96832a78f1e4f46ba9cf3b06e878
fixed cumulative-patch baseline RCF_D0_full.zip
fixed baseline SHA-256          6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2
roadmap authority               RC_K0/child_designs/fortification/data/ROADMAP.csv
roadmap source blob             e6e4ddb762c93540166f4733bd81d5d3d38b2d79
```

The accepted `ROADMAP.csv` row is frozen without paraphrase:

```text
stage               R0C
checkpoint          CP2
title               tower joins
outputs             TowerJoinPlan
focused_validation  tangent/wall-walk
non_goals           gates
gate                R0C_CP2_PASS
next                 R0C-CP3
```

The accepted finite roadmap expands the CP2 geometry scope as:

```text
corner/tangent/wall-penetrating tower joins
```

This file freezes CP2-A specification scope only. It does not claim join geometry, functional qualification, `R0C_CP2_PASS`, or R0C-CP2 completion.

## 2. Owned scope

CP2 owns non-destructive joining between accepted R0C-CP1 tower geometry and accepted R0B wall-span geometry.

Required join kinds:

```text
CORNER
TANGENT
WALL_PENETRATING
```

Required accepted tower families:

```text
ROUND
SQUARE
POLYGONAL
```

Required source interfaces:

```text
wall span: SPAN_START / SPAN_END
 tower:    TOWER_WALL_LEFT / TOWER_WALL_RIGHT
 traversal when playable: WALL_WALK_IN / WALL_WALK_OUT
```

Required continuity and evidence domains:

- exact source tower and span references, revisions, digests, and completion receipts;
- socket origin, orthonormal basis, units/frame domain, and clearance bounds;
- attachment position and tangent/up/inside-frame alignment;
- wall-body continuity;
- wall-walk continuity for playable fixtures;
- foundation-interface continuity;
- bounded overlap extent and ratio;
- bounded unsupported gap;
- inside/outside projection and clearance evidence;
- generated join bounds;
- accepted source geometry before/after digest equality;
- fixed tessellation receipt;
- STEP/BREP stored-copy reopen receipt;
- clean deterministic replay;
- typed failures and `partial_output_published=false`.

## 3. Construction boundary

CP2 may realize a join only as one of the following bounded, project-owned forms:

```text
bounded join module
bounded transition piece
non-destructive assembly reference
```

Accepted R0C-CP1 tower STEP/BREP and accepted R0B span STEP/BREP are immutable source references.

Forbidden:

- merging the complete tower and wall system into one unbounded Boolean solid;
- overwriting accepted tower or span STEP/BREP;
- deforming accepted source geometry to force socket agreement;
- using provider-native face indices, OCCT face IDs, scene-node names, or input-array order as durable identity;
- widening inherited tolerances to hide a mismatch;
- publishing a required partial result after any required source, socket, continuity, budget, or runtime failure.

The accepted ROUND family remains the project-owned 32-sided approximation unless a later authoritative roadmap revision explicitly changes it.

## 4. Fixture freeze

CP2-A freezes a bounded representative matrix rather than a tower-family × span-family cross product.

Required representative fixtures:

```text
ROUND     × TANGENT          × accepted STRAIGHT_SPAN
SQUARE    × CORNER           × accepted STRAIGHT_SPAN
POLYGONAL × WALL_PENETRATING × accepted STRAIGHT_SPAN
```

The exact accepted tower result IDs, span result IDs, socket IDs, and source digests are bound during CP2-B materialization and must remain exact references thereafter. Until that binding and geometry execution occur, every required row remains `FROZEN_PENDING_IMPLEMENTATION`, never `PASS`.

The following span combinations are outside the required R0C-CP2 fixture set:

```text
CURVED_SPAN          NOT_REQUIRED_AT_R0C_CP2
TERRAIN_STEPPED_SPAN NOT_REQUIRED_AT_R0C_CP2
RETAINING_SPAN       NOT_REQUIRED_AT_R0C_CP2
```

This is a CP2-A bounded fixture decision. The authoritative roadmap requires the three join kinds but does not require a full span-family cross product.

## 5. Explicit non-goals

```text
gates and gate complexes                 R0D
battlement rhythm / merlon / crenel       R0C-CP3
second profile fixture and R0C closeout   R0C-CP4
roof realization                          ARCHITECTURE Producer
interior room programs                    WORLD-FIELD / architecture
collision and navigation products         later product stages
Godot stored-copy import/reopen            R0E-CP4
```

No excluded item is implemented or reported as passed by CP2-A.

## 6. CP2-A publication gate

CP2-A closes only when all of the following are true:

```text
exact C1_transfer fresh reopen                         PASS
C1 / C1P / C1S / C1R SHA-256 and ZIP CRC             PASS
C1 FILES.sha256 closed-world replay                   PASS
this scope freeze                                     JSON/MD parse PASS
TOWER_JOIN_CONTRACT.json                              JSON parse PASS
CP2_FIXTURE_MATRIX.csv                                CSV parse PASS
accepted C1 source bytes unchanged except registry   PASS
C1 + CP2-A source delta = CP2-A full                 PASS
RCF_D0_full + cumulative patch = CP2-A full          PASS
full / source / patch / receipts ZIP reopen          PASS
```

CP2-A completion status is therefore:

```text
source preservation          PASS
specification freeze         PASS
functional qualification     NOT_STARTED_PRODUCT
stage completion             CP2_A_COMPLETE / CLOSED
R0C-CP2                      OPEN
Godot product                NOT_STARTED
```

## 7. Next exact checkpoint

```text
ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 CP2-B
— FIRST REPRESENTATIVE FAMILY JOIN,
  EXACT SOURCE/SOCKET BINDING,
  FRAME ALIGNMENT
  & BOUNDED OVERLAP EVIDENCE
```

CP2-B does not start automatically.
