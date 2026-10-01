# Godot Rigged Garment Product Contract

## 1. 제품 구성

`GodotGarmentEquipPackage/1`:

```text
imported GLB or native scene reference
RiggedGarmentProduct receipt
canonical skeleton contract hash
supported adapter versions
Skin resource fingerprint
corrective driver resource
secondary-motion resource
LOD resources
body occlusion mask
outfit compatibility metadata
product provenance
```

GLB는 neutral-gray geometry, skin, joints, inverse bind matrices, UV, normals, material zones와 필요한
sparse morph deltas를 포함한다. 고급 runtime 규칙은 sidecar 또는 native Resource가 소유한다.

## 2. Equip transaction

```text
resolve CharacterRigHandle
→ validate SkeletonAdapterMap
→ validate body profile and skin-source compatibility
→ instantiate garment scene off-tree
→ bind Skin to target Skeleton3D
→ register corrective drivers
→ register secondary-motion domains
→ compile body occlusion
→ outfit compatibility check
→ atomic attach and publish receipt
```

실패 시 새 node, body mask와 runtime registration을 모두 폐기하고 기존 outfit을 유지한다.

## 3. Runtime API 개념

```gdscript
var preview := garment_service.preview_equip(character, garment_product, outfit)
var receipt := garment_service.commit_equip(preview)
garment_service.unequip(character, receipt.instance_id)
```

API 이름은 구현 시 조정할 수 있으나 preview/commit 경계와 원자성은 유지한다.

## 4. Character swap

같은 canonical skeleton contract를 만족하는 캐릭터는 adapter를 교체해 같은 garment product를
사용할 수 있다. Body profile 차이가 topology variant 허용 범위를 넘으면 새 sized topology가
필요하며 3D mesh scaling으로 대체하지 않는다.

## 5. Read-only consumer 원칙

Godot에서 금지:

```text
pattern fitting
seam repair
weight painting
topology weld/snap
vertex shrinkwrap
product-scale size generation
```

허용:

```text
import
skeleton binding
corrective evaluation
LOD selection
secondary-motion update
equip/unequip
evidence capture
```

## 6. Fresh-process acceptance

```text
exact Godot version identity
clean import
fresh-process reopen
Skeleton3D / Skin fingerprint
surface, vertex, triangle, bone and morph counts
semantic bone mapping
neutral-gray material zones
equip → motion suite → unequip
re-equip idempotence
character swap admission/rejection
no import-time geometry mutation
```
