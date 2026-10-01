# Skin Weight Transfer / Corrective Deformation 설계

## 1. Automatic bind pipeline

```text
garment component vertex
→ bind-region admission
→ body surface candidate query
→ semantic-region filtering
→ barycentric body-weight interpolation
→ component-specific constraint
→ seam/layer regularization
→ influence pruning and normalization
→ deformation qualification
```

단순 nearest vertex 복사는 사용하지 않는다.

## 2. Component별 정책

### Shell

body surface weight를 barycentric interpolation하고 seam을 가로지르는 급격한 discontinuity를
제한한다. silhouette와 construction feature 경계를 보존한다.

### Lining

shell driver map을 기본으로 소비하되 turn-of-cloth offset과 sliding allowance를 별도 소유한다.
shell vertex index를 무조건 복제하지 않는다.

### Facing / Interfacing

facing은 owner shell region의 weight를 따른다. bonded interfacing은 해당 surface driver와
동일한 rigidized weight field를 사용한다.

### Dart / Pleat / Gather

fold 양쪽이 서로 다른 bone field로 갈라지지 않도록 feature-local pair constraint를 둔다.
pleat free edge와 gather fullness는 corrective 또는 secondary-motion domain을 가질 수 있다.

### Underarm gusset

torso, clavicle과 upper-arm 영향이 혼합되는 별도 커널이다. 좌우 대칭은 stable semantic map으로
발행하며 전역 smoothing으로 처리하지 않는다.

### Trousers crotch / knee

pelvis, left/right thigh와 calf 사이의 영향 분할을 명시한다. crotch seam과 waistband는 별도
admission을 갖는다.

### Hardware

button, buckle, rigid closure는 단일 bone rigid bind 또는 surface attachment로 제한한다.

## 3. Weight invariants

```text
finite weights                         required
weight sum per vertex                  exactly 1 within tolerance
zero-influence vertex                  0
negative weight                        0
maximum influences                     profile controlled, normally 4 or 8
required semantic region violation     0
left/right semantic leakage            0 outside transition region
seam discontinuity                     family-specific bounded gate
```

## 4. `RigBindPlan/1`

```text
garment product and topology hashes
canonical skeleton hash
body skin-source hash
component bind policy
semantic region masks
body source triangles and barycentrics
seam regularization groups
rigid attachment groups
influence limit
solver profile
admission status
```

## 5. Corrective deformation

Skeleton skinning 잔차는 full pose 교체가 아니라 sparse local corrective로 처리한다.

`CorrectiveDriverSet/1`:

```text
driver ID
semantic joint IDs
angle / quaternion feature extraction
activation curve
supported rig-adapter range
```

`CorrectiveDeformationSet/1`:

```text
owner garment family and component
affected vertex region
sparse position/normal deltas
driver references
additive or exclusive composition rule
LOD transfer identity
source motion-fit residual
```

대표 corrective owner:

```text
shoulder raise
cross-body underarm
deep elbow fold
forward-bend waist
seated pelvis
squat crotch/thigh
walking stride hem split
```

## 6. 생성 방식

1. canonical skeletal deformation을 실행한다.
2. R1A accepted Warp pose 결과와 비교한다.
3. 잔차를 semantic region과 feature boundary로 제한한다.
4. 저차원 driver basis에 fitting한다.
5. unseen intermediate angle에서 interpolation을 검사한다.
6. full-pose morph보다 작은 sparse corrective임을 확인한다.

Corrective는 패턴 결함이나 잘못된 skin weight를 숨기는 용도로 사용하지 않는다.
