# ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2 — accepted closeout

```text
official subtitle             TOWER JOINS
checkpoint                    CP2-D-R1
source preservation           PASS
exact runtime                 PASS
functional qualification      PASS
stage completion              R0C_CP2_COMPLETE / CLOSED
Godot product                 NOT_STARTED
R0C progress                  2 / 4
accepted implementation CP    11 / 25
```

## Required family matrix

- `R0C_CP2_FX01`: ROUND × TANGENT × STRAIGHT_SPAN — PASS
- `R0C_CP2_FX02`: SQUARE × CORNER × STRAIGHT_SPAN — PASS
- `R0C_CP2_FX03`: POLYGONAL × WALL_PENETRATING × STRAIGHT_SPAN — PASS

All three fixtures passed wall-body, wall-walk, and foundation-interface continuity. Accepted tower/span source digests were unchanged.

## CP2-D closeout

- negative gates: `18 / 18 PASS`
- atomic no-partial-output: `PASS`
- `partial_output_published=false`
- FX01/FX02/FX03 final clean A/B: byte-identical
- STEP/BREP independent reopen: `PASS`
- closed-world registry: `PASS — 1,617 rows`

Historical accepted references embed old GitHub runner absolute paths in three JSON receipts. Raw comparison against those historical files is therefore false, but A/B replay is byte-identical and project-relative semantic comparison is PASS. The historical accepted reference was not overwritten.

## Final artifacts

Persistent Library destination:

```text
/Deliveries/ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP2/
```

| File | Bytes | SHA-256 |
|---|---:|---|
| `C2.zip` | 259487144 | `686b9857cc37d0dff2c0c26bc3630d8c94a53c878a92001d4f0e3aa4200e69b2` |
| `C2P.zip` | 254651940 | `51bf174668ac5b9c4373f5147fbba3db5189026a96e3432055af761831873c0a` |
| `C2S.zip` | 377063 | `3627c59740e1d5bc15ba5a254994aa8a849a1542bdc9f45c0e42a9e91c166527` |
| `C2R.zip` | 103764 | `79362bbc8a0d1f19970602318fb48cddc59b90e5761c0c14dda43a7519772722` |

## Direct reconstruction

```text
RCF_D0_full.zip + C2P.zip = C2.zip    PASS
C1.zip          + C2S.zip = C2.zip    PASS
C2P current payload = C2S payload     PASS
```

Each path was extracted into a fresh directory and compared by path, bytes, executable/mode, deletion set, file count, and closed-world registry.

## Next roadmap task

```text
ROYAL-CAPITAL-FORTIFICATION-CAD-R0C-CP3
— BATTLEMENT RHYTHM,
  MERLON / CRENEL
  & MODULE INSTANCING
```

Automatic continuation is forbidden; explicit approval is required.
