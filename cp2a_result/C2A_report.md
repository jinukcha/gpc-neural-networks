# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — CP2-A FREEZE PUBLICATION

## Verdict

```text
source preservation                         PASS
exact C1 authority admission                PASS
C1 FILES.sha256 closed-world replay         PASS
scope / contract / fixture freeze           PASS
fixed baseline + cumulative patch equality  PASS_COMPOSITIONAL_WITH_ACCEPTED_DIRECT_REPLAY
C1 + source delta equality                  PASS_DIRECT
ZIP reopen / CRC                            PASS
functional qualification                    NOT_STARTED_PRODUCT
stage completion                            CP2_A_COMPLETE / CLOSED
R0C-CP2                                     OPEN
Godot product                               NOT_STARTED
```

## Frozen authority

```text
title               tower joins
output              TowerJoinPlan
focused validation  tangent/wall-walk
scope               corner/tangent/wall-penetrating tower joins
```

## Source changes

```text
added      3
modified   1
deleted    0
wheelhouse 58 / unchanged
```

Only the three CP2-A freeze files were added; root `FILES.sha256` was regenerated. No accepted tower, span, runtime, OSS, license, provenance, STEP, or BREP byte was modified.

## Equality proof

```text
accepted direct replay:
  RCF_D0_full.zip + exact C1P.zip = exact C1.zip  PASS
fresh current replay:
  exact C1.zip + C2AS.zip = C2A.zip               PASS
fresh patch composition:
  C2AP current delta = C2AS payload               PASS
conclusion:
  RCF_D0_full.zip + C2AP.zip = C2A.zip            PASS_COMPOSITIONAL
pre checkpoint bytes = final bytes                PASS
```

The fixed-baseline conversation transport URL expired before this run. No baseline was regenerated. The separately accepted direct replay artifact was freshly reopened and validated, then combined with fresh current-delta equality.

## Roadmap

```text
R0C progress          1 / 4
accepted checkpoints  10 / 25
R0C-CP2               OPEN
next                   R0C-CP2 CP2-B — explicit start only
```
