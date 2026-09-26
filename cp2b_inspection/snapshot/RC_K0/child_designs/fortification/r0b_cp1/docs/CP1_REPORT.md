# R0B-CP1 completion report

```text
task                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0B-CP1
parent                       A4.zip
parent SHA-256               8f0d0a6667c8521c6f2720f2580304c4e06b480bd9e470384bc2828516edf751
fixed baseline               RCF_D0_full.zip
baseline SHA-256             6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2

source preservation          PASS
runtime reuse                PASS
functional qualification     PASS
stage completion             R0B_CP1_COMPLETE / CLOSED
```

## Accepted geometry

### Straight span

```text
centerline kind              LINE
length                       24.0 m
segments                     1
samples / local frames       2 / 2
semantic solids              5
sockets                      10
vertices / triangles         40 / 60
volume                       2273.28 m3
clean replay                 35 / 35 byte-identical
```

### Curved span

```text
centerline kind              CIRCULAR_ARC
radius                       48.0 m
sweep                        30 deg
arc length                   25.132741229 m
segments                     6
samples / local frames       7 / 7
maximum chord error          0.045685364 m
semantic solids              5
sockets                      10
vertices / triangles         140 / 260
volume                       2377.552882447 m3
clean replay                 35 / 35 byte-identical
```

## Canonical frame

```text
project frame                RH_X_RIGHT_Y_UP_NEG_Z_FORWARD
up                           +Y
inside                       left of travel
outside                      right of travel
orientation determinant      +1
provider-native curve order  not authoritative
```

## Negative gates

```text
zero length                  REJECTED
vertical / graded span       REJECTED — deferred to R0B-CP3
radius below profile offset  REJECTED
turn above 35 degrees        REJECTED
section budget exceeded      REJECTED
runtime mismatch             REJECTED

total                        6 / 6 PASS_EXPECTED_REJECTION
```

The first qualification attempt is preserved under `reports/history/qualification_attempt_01_failed`.
It failed because the negative harness invoked only static fixture parsing for the derived section-budget case.
The harness was corrected to execute the complete canonical-centerline contract; existing source and generated
positive outputs were preserved before retry.

## Deferred

```text
miter / bevel / transition joins
terrain-stepped foundation interfaces
curved semantic-surface triangle coverage closeout
towers and gates
collision and navigation
Godot import
```

## Next

```text
ROYAL-CAPITAL-FORTIFICATION-CAD-R0B-CP2
— MITER / BEVEL / PROFILE-TRANSITION JOINS,
  SOCKET ALIGNMENT
  & BOUNDED OVERLAP
```

Explicit approval is required.
