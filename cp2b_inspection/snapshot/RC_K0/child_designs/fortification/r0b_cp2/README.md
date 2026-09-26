# R0B-CP2 — Join geometry, socket alignment and bounded overlap

This checkpoint adds three bounded join families behind the existing project-owned CAD boundary:

```text
MITER
BEVEL
PROFILE_TRANSITION
```

Each join consumes one incoming END socket set and one outgoing START socket set, preserves the source spans unchanged, emits six exact interface sockets, records alignment error, and limits source-span overlap by an explicit meter and ratio contract.

```text
functional qualification   PASS
focused validation         47 / 47 PASS
stage completion           R0B_CP2_COMPLETE / CLOSED
```

Provider-native face or curve identity is not authoritative. Surface coverage remains deferred to R0B-CP4. R0B-CP3 requires explicit approval.
