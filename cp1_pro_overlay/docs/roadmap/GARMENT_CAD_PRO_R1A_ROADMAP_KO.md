# GARMENT-CAD-PRO-R1A Roadmap

## CP0 — Anthropometry V2

Recovered minimum authority for CP1 from preserved CP3 sizing inputs. The inaccessible local CP0 ZIP is not claimed byte-exact.

## CP1 — Parametric Pattern Document / Constraint Graph / Edit Transaction

Implemented:

- `PatternDocument/1`
- POM expression DAG
- exact 2D curve authority
- hard/soft geometric constraints
- atomic edit commit and failure preservation
- undo / redo
- canonical save / fresh reopen
- tunic M reference migration and parity

Not executed:

- feature topology operations
- triangulation
- Warp simulation

## Next — CP2

`GARMENT-CAD-PRO-R1A / CP2 — INDUSTRIAL GRADING / DART–PLEAT–GATHER–GUSSET FEATURE GRAPH`

CP2 consumes CP1 documents without mutating their committed revision. It adds named grade points, rule propagation and topology-aware construction features. Triangulation and product simulation remain deferred.
