# Visual Product Acceptance Matrix

## Product decision

```text
product_acceptance = technical_pass AND visual_review == PASS
```

## 필수 evidence category

```text
PRIMARY
DETAIL
DIAGNOSTIC
MOTION
LOD_ACTUAL_DISTANCE
LOD_EQUAL_COVERAGE
```

Primary view는 front/left/right/back/front-three-quarter/back-three-quarter를 포함한다. Diagnostic은 body-visible fit, body-occlusion runtime, wireframe, seam/notch, component ownership, layer cutaway와 normal view를 포함한다.

## Zero-tolerance gate

```text
floating component                 0
unowned loose triangle             0
non-design open boundary           0
neck/facing separation             0
sleeve-cap/armhole gap              0
underarm seam gap                  0
normal inversion                   0
layer-order reversal               0
closure separation                 0
non-finite geometry                0
```

## Bounded gate

```text
body penetration p99              <= 2 mm
undesigned silhouette asymmetry   <= 1%
LOD1 silhouette IoU               >= 0.985
LOD2 silhouette IoU               >= 0.960
LOD feature boundary displacement <= 1.5 px
required motion views             PASS
```

CP0는 profile과 clean/rejected fixture를 발행할 뿐 실제 garment capture를 수락하지 않는다. 실제 제품은 CP4 이후 동일 authority를 소비해야 한다.
