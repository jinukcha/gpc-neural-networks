# R1C CP3 Completion / Repair Implementation

## Scope

CP3 consumes the immutable CP2 exact 2D geometry library and assembled pattern package. It does not triangulate panels or run cloth simulation.

## Authority split

- `CompletionDiagnosisReport/1` identifies missing components, interfaces, parameters, notches, curve drift, and seam-length drift.
- `RepairPlan/1` converts only identity-preserving issues into bounded source-pattern operations.
- `RepairPreview/1` projects before/after state without mutating source authority.
- `CompletionTransactionReceipt/1` records commit, guided wait, topology HOLD, or atomic rollback.

## Dispositions

- `SAFE_AUTO`: mirrored component restoration, missing interface/parameter/notch restoration, or source-curve drift within the safe budget.
- `GUIDED`: topology-preserving curve drift outside safe-auto limits but within the guided budget. Preview is published; state is not committed without approval.
- `HOLD`: missing non-mirrored required component or any topology-changing requirement.

All operations target source pattern ownership. Final-mesh shrinkwrap, snap, weld, or vertex pulling are forbidden.
