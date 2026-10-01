# Rigged Garment Acceptance Matrix

## 1. Skeleton / adapter

| Gate | 판정 |
|---|---|
| Required semantic bones | 100% resolved |
| Parent hierarchy | canonical-compatible |
| Rest transforms | finite |
| Left/right mapping | no inversion |
| Unsupported required bone | 0 |
| Adapter hash mismatch | reject |

## 2. Skin weights

| Gate | 판정 |
|---|---|
| Non-finite / negative weight | 0 |
| Zero-influence vertex | 0 |
| Weight sum | 1.0 within tolerance |
| Maximum influences | profile bound |
| Forbidden region influence | 0 |
| Seam discontinuity | family-specific gate |
| Rigid hardware deformation | policy match |

## 3. Deformation

고정 suite:

```text
neutral A
arms forward
arms overhead
cross-body reach
deep elbow bend
torso twist
forward bend
seated
squat
walk stride
```

의상군별 추가 suite:

```text
sleeved garments      elbow and shoulder cycles
trousers              sit / squat / stride / step-up
skirt / robe          stride / turn / sit
coat                  arm raise / cross-body / twist
gloves                fist / finger spread
footwear              ankle flex / toe-off
```

검사:

```text
body penetration
seam inversion
triangle inversion
gusset collapse
lining order reversal
closure separation
corrective overshoot
normal and UV distortion
secondary-motion instability
```

## 4. Outfit composition

```text
incompatible slot admission           0
coverage/body-mask mismatch            0
unresolved layer collision             0
layer order cycle                      0
outer garment inside inner garment     0
secondary-domain ownership conflict     0
atomic equip rollback failure           0
```

## 5. LOD

```text
bone semantic identity             exact across LOD
weight normalization               PASS
corrective owner preservation      PASS
body penetration at transition     0
visible silhouette discontinuity   bounded by product profile
material zone parity               PASS
```

## 6. Export / Godot

```text
canonical save/reopen
GLB skin and inverse bind parity
Godot 4.7.2 clean import
fresh-process reopen
equip / unequip / re-equip
character rig swap
read-only product consumption
topology mutation count 0
```

## 7. 시각 증거

각 product closeout은 실제 엔진 또는 실제 product geometry에서 다음 PNG를 발행한다.

```text
front / left / right / back neutral
shoulder / underarm detail
waist / crotch or knee detail
fixed motion contact sheet
skin-weight diagnostic
corrective before/after
outfit layer cutaway
LOD comparison
```

이미지 생성 도구나 예상 컨셉 아트를 acceptance 증거로 사용하지 않는다.
