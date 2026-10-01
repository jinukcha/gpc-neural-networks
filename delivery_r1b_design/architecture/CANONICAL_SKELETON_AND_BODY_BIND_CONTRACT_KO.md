# Canonical Skeleton / Body Bind Contract

## 1. 목적

캐릭터마다 bone 이름, 축, rest pose와 비율이 달라도 의상 제품이 같은 semantic rig 계약을
소비하도록 한다. 캐릭터별 수기 weight painting과 파일 복제를 제거하는 것이 목표다.

## 2. `CanonicalSkeletonPackage/1`

각 bone record:

```text
semantic_bone_id
parent semantic ID
canonical local rest transform
canonical global rest transform
deform / control disposition
optional mirror bone ID
body region ownership
allowed garment influence categories
```

최소 semantic bone:

```text
ROOT / PELVIS
SPINE_01 / SPINE_02 / CHEST / NECK / HEAD
L/R_CLAVICLE
L/R_UPPER_ARM / FOREARM / HAND
L/R_THIGH / CALF / FOOT / TOE
```

Twist, breast, skirt, coat-tail 같은 보조 bone은 optional extension으로 처리하고 필수 humanoid
semantic bone을 대체하지 않는다.

## 3. `SkeletonAdapterMap/1`

외부 캐릭터 rig와 canonical bone 간 매핑:

```text
source skeleton fingerprint
canonical skeleton fingerprint
semantic ID → source bone path
axis conversion
rest-pose rotation offset
translation and scale normalization
parent-chain verification
missing-bone disposition
optional extension mapping
```

허용 disposition:

```text
EXACT
ADAPTED_WITH_REST_OFFSET
OPTIONAL_MISSING
UNSUPPORTED_REQUIRED_BONE
```

required bone 누락, parent-chain 역전, 좌우 반전 또는 비유한 transform은 호환성 실패다.

## 4. `BodySkinSourcePackage/1`

단순 body mesh가 아니라 garment binding의 직접 정본이다.

```text
body vertices / triangles
canonical skin weights
semantic body-region IDs
body landmarks and section loops
surface normals and thickness envelope
body-profile identity
canonical skeleton identity
maximum influence policy
source mesh and rig hashes
```

각 garment vertex는 가능한 경우 body triangle의 barycentric source와 semantic region을
참조한다. nearest body vertex index만 저장하는 경로는 금지한다.

## 5. Character compatibility

`SkeletonCompatibilityReceipt/1`은 장착 전에 다음을 확인한다.

```text
required semantic bone coverage
parent hierarchy
rest transform finite
handedness and axis
body skin source compatibility
body profile range
optional extension availability
corrective driver availability
```

호환되지 않는 rig에 근사 weight를 강제로 적용하지 않는다.

## 6. Pose normalization

기준 pose는 canonical relaxed A-pose다. T-pose나 특정 캐릭터 bind pose는 adapter가 canonical
pose로 변환한다. 의상마다 서로 다른 기준 pose를 갖지 않는다.

## 7. 좌표계

```text
canonical world     right-handed Z-up
distance            metre
angular data        radian
transforms          finite 4×4 or TRS
scale               positive and explicit
```

Godot Y-up 변환은 export boundary에서만 수행하며 내부 패턴·simulation 정본의 좌표계를
변경하지 않는다.
