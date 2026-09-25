# R0B-CP1 implementation

## Scope

- horizontal `LINE` and bounded `CIRCULAR_ARC` centerlines
- deterministic sampling from chord-error and turn-angle limits
- right-handed local frame: tangent, fixed +Y up, left-of-travel inside
- five non-unioned semantic wall parts
- bounded sampled-section ruled lofts
- per-part STEP/BREP, neutral mesh and provider receipt
- stable start/end/walk/foundation/tower/utility sockets

## Deferred

- miter/bevel/profile-transition joins
- terrain grade and stepped foundations
- curved semantic-surface triangle coverage closeout
- towers, gates, collision, navigation and Godot import
