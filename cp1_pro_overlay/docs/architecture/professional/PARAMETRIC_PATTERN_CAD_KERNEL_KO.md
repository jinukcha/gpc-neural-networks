# GARMENT-CAD-PRO-R1A — Parametric Pattern CAD Kernel

## Authority

`PatternDocument/1` is the editable 2D authority. A resolved 3D or triangulated mesh is a derivative and may not write back into this document.

```text
BodyMeasurementProfile/2
→ scalar POM expression DAG
→ named pattern points
→ exact line / quadratic Bézier / cubic Bézier curves
→ hard and soft geometric constraints
→ atomic edit transaction
→ canonical save / reopen
```

## Implemented in CP1

- Safe scalar expression DAG without Python `eval`.
- Cycle, unknown symbol, division-by-zero and non-finite rejection.
- Exact curve authority for line, quadratic Bézier and cubic Bézier.
- Panel ownership and `OPEN / SEWN / INTERNAL` boundary disposition.
- Hard/soft `COINCIDENT`, `HORIZONTAL`, `VERTICAL`, `FIXED_DISTANCE`, `MIN_DISTANCE`, `SYMMETRY_X` constraints.
- Candidate validation before commit.
- Failed transaction preserves revision and document hash.
- Undo, redo and canonical JSON save/reopen.
- CP2 tunic M landmark parity gate at `1e-12 m`.

## Deferred

Dart, pleat, gather, gusset, grading, triangulation and Warp simulation are not CP1 responsibilities.

## Recovery boundary

The local CP0 integrated ZIP could not be opened after two independent acquisition attempts. CP1 reconstructs the minimum `BodyMeasurementProfile/2` authority from the preserved CP3 sizing inputs and records `byte_exact_cp0_predecessor=false`. Existing sizing, drape and tunic build owners remain immutable.
