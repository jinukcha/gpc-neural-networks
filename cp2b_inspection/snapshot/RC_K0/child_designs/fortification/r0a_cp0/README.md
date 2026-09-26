# RC-FORT R0A-CP0 — exact build123d/OCP runtime admission

```text
source/reference fence       PASS
exact wheel delivery         58 / 58 PASS
remote fresh runtime A/B     PASS / PASS
local fresh runtime A/B      PASS / PASS
pip check                    PASS ×2
native capability            9 / 9 PASS ×2
cold A/B canonical           PASS
artifact transport           PASS
stage completion             R0A_CP0_COMPLETE / CLOSED
R0A-CP1 start                ALLOWED
```

The exact CPython 3.13.5 Linux wheelhouse, complete hash lock, delivery logs, probe artifacts and provenance are retained under `runtime/`. Extracted venvs remain outside the product tree. The first interrupted local B materialization was preserved and never reused; B was admitted in a new fresh destination.

Raw STEP bytes differ because the exporter embeds output path and timestamp in `FILE_NAME`. Only those fields are normalized for canonical comparison. Raw STEP hashes remain evidence; BREP bytes, imported geometry, volume and tessellation counts are identical.

Next: `ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP1 — PROJECT-OWNED PROVIDER ADAPTER, NEUTRAL REQUEST/RESULT TYPES, UNIT/FRAME/TOLERANCE CONTRACT & CAD PROVIDER RECEIPT`.
