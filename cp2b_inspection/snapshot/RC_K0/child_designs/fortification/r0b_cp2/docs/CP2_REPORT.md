# R0B-CP2 completion report

```text
source preservation          PASS
runtime reuse                PASS
miter join                   PASS
bevel join                   PASS
profile transition           PASS
socket alignment             PASS — 6 / 6 per family
bounded overlap              PASS
clean replay                 PASS — 36 / 36 files per family
negative gates               PASS — 9 / 9 expected rejection
source spans unchanged       PASS
focused validation           PASS — 47 / 47
stage completion             R0B_CP2_COMPLETE / CLOSED
next                         R0B-CP3 — EXPLICIT APPROVAL REQUIRED
```

## Accepted functional fixtures

```text
MITER                  10 degrees, 3 sections, 8 m overlap each side
BEVEL                  30 degrees, 4 sections, 10 m overlap each side, 5 m setback
PROFILE_TRANSITION      0 degrees, 3 sections, standard → reinforced profile, 6 m overlap each side
```

## Preserved failures and bounded corrections

1. The first runtime/qualification attempt stopped after a tool timeout left a partial venv that passed `pip check` but did not contain build123d. The partial runtime and failure logs were preserved. A second fresh exact runtime was materialized and qualified.
2. The first focused validator used a turn-angle comparison tighter than the canonical `acos` calculation noise. It rejected `10.000000019°` against a `10°` fixture. Geometry and qualification outputs were not changed; only the validator tolerance was corrected to `1e-6°`, and the failed report was preserved.
3. The first package verifier compared raw ZIP external mode bits, treating `0100644` and `0644` as different. The archives were retained; the verifier was corrected to compare `stat.S_IMODE`, after which full/tree, baseline+patch/full and parent+source/full all passed.

Join source-surface coverage remains deliberately deferred to R0B-CP4. Terrain adaptation, collision, navigation and Godot product work are outside this checkpoint.
