# R0C-CP3 scope freeze

## Official roadmap scope

```text
battlement rhythm, merlon/crenel/module instancing
```

Reporting title:

```text
ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP3
— BATTLEMENT RHYTHM,
  MERLON / CRENEL
  & MODULE INSTANCING
```

## Accepted authority

```text
parent             C2.zip
parent SHA-256     686b9857cc37d0dff2c0c26bc3630d8c94a53c878a92001d4f0e3aa4200e69b2
parent files       1,618
parent tree        sha256:389f7c0865ece121f370296936f02b04efa06a4c5d9bf25fe32c0a01325edcb7
R0C progress       2 / 4
accepted CP        11 / 25
```

## Included in R0C-CP3

- shared `MERLON` and `CRENEL` module definitions;
- linear-open and closed-perimeter battlement rhythm plans;
- pitch, phase, start/end and corner policies;
- closed-loop phase closure and bounded endpoint residual;
- stable project-owned module instance IDs;
- finite instance count and bounded artifact budget;
- Definition/Instance separation and `ModuleInstancePlan` output;
- deterministic replay, accepted host immutability and no-partial-output;
- straight-span, round-tower, square-tower and polygonal-tower representative fixtures.

## Explicitly excluded

- tower and wall-walk socket redesign — R0C-CP4;
- second fortification profile fixture — R0C-CP4;
- final semantic source coverage and R0C closeout — R0C-CP4;
- roof realization — architecture Producer;
- interiors — WORLD-FIELD / architecture;
- collision and navigation — later product stages;
- Godot product import — `NOT_STARTED`.

## Non-destructive realization boundary

Allowed:

- shared module definitions plus project-owned instance transforms;
- bounded endpoint/corner modules where a fixture explicitly requires them;
- crenel represented as a declared void definition when no solid is needed.

Forbidden:

- fusing thousands of merlons into one BREP Boolean solid;
- duplicating definition geometry per instance;
- overwriting accepted wall or tower STEP/BREP;
- changing accepted host geometry to force pitch closure;
- using input array order as durable identity;
- unbounded instance expansion;
- widening tolerance to hide pitch, phase or endpoint defects;
- publishing partial accepted output after any required module failure.

## CP3-A closeout condition

CP3-A closes only after the scope, contract and finite fixture matrix are present in the C2-derived tree, `FILES.sha256` is closed-world, and both direct replay paths pass:

```text
RCF_D0_full.zip + C3AP.zip = C3A.zip
C2.zip          + C3AS.zip = C3A.zip
```

CP3-A does not claim battlement geometry functional qualification. That begins with CP3-B.
