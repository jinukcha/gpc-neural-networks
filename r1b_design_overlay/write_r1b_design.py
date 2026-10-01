#!/usr/bin/env python3
"""Write the GARMENT-CAD-PRO-R1B rig-aware garment design authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path


PROGRAM_ID = "GARMENT_CAD_PRO_R1B"
PROGRAM_TITLE = "GARMENT-CAD-PRO-R1B — RIG-AWARE GAME GARMENT PLATFORM / MULTI-GARMENT PRODUCTIZATION"


def _clean(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_clean(text), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main_design() -> str:
    return r'''
    # GARMENT-CAD-PRO-R1B — 리그 인식 게임 의상 플랫폼 설계 정본

    ## 0. 문서 상태

    ```text
    program                  GARMENT-CAD-PRO-R1B
    predecessor              GARMENT-CAD-PRO-R1A / CP6-R1
    predecessor status       COMPLETE / immutable
    document role            architecture authority
    implementation status    NOT STARTED
    target engine            Godot 4.7.2 Linux
    primary runtime          game humanoid Skeleton3D
    ```

    R1A는 신체 계측, 파라메트릭 패턴, 제조 construction, 물성 calibration,
    motion-fit, 제조 2D와 neutral-gray 3D 제품까지 닫았다. R1B는 이 정본을
    수정하지 않고 실제 게임 캐릭터 Skeleton에 장착 가능한 의상 제품 계층을 추가한다.

    R1B의 목표는 한 벌을 리깅하는 것이 아니다. 앞으로 추가될 다수의 상의, 하의,
    로브, 외투, 치마, 장갑, 신발, 머리 장식과 방어구 하부 의복이 같은 계약을 소비하고,
    의상별 특수 구조는 해당 family 커널이 소유하게 만드는 것이다.

    ---

    ## 1. 제품 목표

    R1B가 완료되면 다음 흐름이 성립해야 한다.

    ```text
    R1A Pattern / Construction / Material authority
        ↓
    Feature-complete garment topology
        ↓
    Canonical humanoid skeleton + body skin source
        ↓
    Garment-specific bind plan
        ↓
    Skin weights + sparse local corrective deformation
        ↓
    Optional secondary-motion domains
        ↓
    Rig-aware LOD products
        ↓
    Outfit layer composition
        ↓
    Godot import / character-rig adapter / atomic equip
    ```

    완료 제품은 다음을 지원한다.

    - 캐릭터 Skeleton 호환성 사전 판정
    - 체형별 garment rest topology와 skin binding
    - Skeleton animation 기반 primary deformation
    - 어깨, 겨드랑이, 팔꿈치, 골반, 샅, 무릎의 국소 corrective
    - hem, loose sleeve, skirt, pleat free edge의 선택적 secondary motion
    - shell, lining, facing, interfacing, closure hardware의 서로 다른 bind 정책
    - 여러 의상의 동시 장착과 layer/coverage/occlusion 해결
    - LOD 전환 시 bone identity, silhouette와 corrective 의미론 보존
    - Godot에서 topology 수정 없는 read-only 소비

    ---

    ## 2. 비목표

    R1B는 다음을 하지 않는다.

    - 별도 범용 게임 엔진 또는 별도 의류 런타임 프레임워크를 만들지 않는다.
    - Blender나 Godot에서 수기 weight painting을 정본으로 삼지 않는다.
    - 캐릭터마다 복제된 의상 소스를 만들지 않는다.
    - 애니메이션 pose마다 전체 의상을 교체하는 full-body morph 방식을 기본으로 사용하지 않는다.
    - Rig 오류를 GLB import 후 vertex snap, weld, shrinkwrap 또는 mesh scaling으로 수정하지 않는다.
    - 모든 의상을 하나의 거대 `garment.py`, `rig_utils.py`, `common` 폴더에 넣지 않는다.
    - R1A 제조 정본이나 이미 수락된 CP6-R1 제품을 덮어쓰지 않는다.

    ---

    ## 3. 세 제품 계층

    전문 시스템에서는 제조 패턴, simulation 제품과 게임 제품을 구분한다.

    ### 3.1 Manufacturing authority

    ```text
    PatternDocument
    ConstructionPackage
    MaterialMeasurementSet
    ManufacturingPatternPackage
    ```

    치수, 패턴 곡선, seam, 시접, notch, lining, interfacing, 여밈과 공정 순서의 정본이다.

    ### 3.2 Simulation garment product

    ```text
    FeatureCompleteTopologyPackage
    WarpGarmentModelPackage
    MotionFitQualificationReceipt
    ```

    rest topology, component/layer ownership, 물성, seam과 contact의 정본이다.

    ### 3.3 Rigged game garment product

    ```text
    RiggedGarmentProduct
    SkinWeightField
    CorrectiveDeformationSet
    SecondaryMotionProfile
    RiggedGarmentLODSet
    GodotGarmentEquipPackage
    ```

    Skeleton animation과 runtime 장착에 필요한 파생 제품이다. 게임 제품은 제조 정본을
    역으로 수정하지 않는다.

    ---

    ## 4. 권위 방향과 변경 전파

    ```text
    Body measurements changed
    → size/block/alteration stale
    → pattern stale
    → topology stale
    → bind plan stale
    → skin/corrective/LOD/runtime product stale

    Skeleton rest pose changed
    → skeleton adapter stale
    → bind plan and all rig products stale
    → pattern and manufacturing authority remain valid

    Material changed
    → simulation and secondary-motion profile stale
    → skeleton weights remain valid unless thickness/layer topology changes

    Camera or evidence style changed
    → evidence only stale
    ```

    최소 cache identity:

    ```text
    canonical skeleton hash
    external rig adapter hash
    body skin-source hash
    body profile hash
    garment design and construction hash
    feature topology hash
    material and layer-stack hash
    bind profile hash
    skin-weight solver profile hash
    corrective driver profile hash
    LOD profile hash
    outfit composition hash
    Godot runtime lock hash
    ```

    ---

    ## 5. 공통 계약

    R1B의 공유 계약은 실제로 여러 garment family가 소비하는 것만 포함한다.

    ```text
    CanonicalSkeletonPackage/1
    SkeletonAdapterMap/1
    SkeletonCompatibilityReceipt/1

    BodySkinSourcePackage/1
    BodySurfaceSemanticMap/1

    GarmentFamilyDefinition/1
    GarmentVariantDefinition/1
    GarmentBindProfile/1
    RigBindPlan/1
    SkinWeightField/1

    CorrectiveDriverSet/1
    CorrectiveDeformationSet/1
    SecondaryMotionProfile/1

    RiggedGarmentLODSet/1
    RiggedGarmentProduct/1

    OutfitAssemblyPlan/1
    BodyOcclusionMask/1
    OutfitCompatibilityReceipt/1

    GodotGarmentEquipPackage/1
    RuntimeGarmentReceipt/1
    ```

    각 계약은 version, canonical hash, upstream source hash와 loss report를 갖는다.

    ---

    ## 6. 다수 의상군을 위한 소유 구조

    ```text
    source/wuxia_garment_oss/
    ├── rig/
    │   ├── skeleton_contract/
    │   ├── skeleton_adapter/
    │   ├── body_skin_source/
    │   └── qualification/
    ├── outfit/
    │   ├── registry/
    │   ├── layering/
    │   ├── occlusion/
    │   ├── compatibility/
    │   └── assembly/
    ├── runtime/
    │   └── godot/
    │       ├── equip_contract/
    │       ├── skeleton_binding/
    │       ├── corrective_driver/
    │       ├── secondary_motion/
    │       └── lod_runtime/
    └── garments/
        ├── sleeveless_tunic/
        │   ├── rig_binding/
        │   ├── corrective_deformation/
        │   ├── secondary_motion/
        │   ├── rig_lod/
        │   └── runtime_product/
        ├── trousers/
        │   └── ...
        ├── sleeved_tunic/
        │   └── ...
        ├── straight_sleeve_robe/
        │   └── ...
        ├── skirt/
        │   └── ...
        └── coat/
            └── ...
    ```

    `rig/`는 semantic bone과 body skin-source 같은 공통 계약을 소유한다.
    실제 weight field, gusset 영향, sleeve cap 보정, crotch와 knee corrective는 해당 의상
    family가 소유한다. 공통 폴더가 의상별 예외를 흡수하는 구조를 금지한다.

    수기 소스는 원칙적으로 파일당 500 LOC, 함수당 80 LOC 이하를 유지한다. 단순 facade나
    이름만 다른 helper로 제한을 회피하지 않고 데이터 소유권과 알고리즘 단계에 따라 분리한다.

    ---

    ## 7. Garment family 확장 모델

    `GarmentFamilyDefinition/1`은 코드 plugin framework가 아니라 데이터 계약과 소유 커널의
    등록 단위다.

    필수 항목:

    ```text
    family_id
    semantic category
    manufacturing authority resolver
    required body measurements
    required canonical bones
    body coverage regions
    outfit slots
    layer class
    thickness/compression profile
    bind regions
    corrective regions
    secondary-motion domains
    supported LOD profile
    compatible closures and accessories
    incompatibility rules
    product builders
    qualification suite
    ```

    권장 category:

    ```text
    torso_inner
    torso_mid
    torso_outer
    full_body_inner
    full_body_outer
    lower_body_inner
    lower_body_outer
    arm_accessory
    hand
    foot
    head_neck
    armor_underlayer
    accessory
    ```

    같은 family에서 style, material, size와 trim만 다른 경우는
    `GarmentVariantDefinition/1`로 표현한다. topology와 bind 정책이 질적으로 바뀌면 새 family
    또는 명시적 topology variant가 필요하다.

    ---

    ## 8. Outfit composition

    여러 의상을 장착할 때 단순 slot 독점만 사용하지 않는다. 긴 로브처럼 torso와 lower body를
    동시에 덮는 의상이 있기 때문이다.

    `OutfitAssemblyPlan/1`은 다음을 소유한다.

    ```text
    ordered garment product IDs
    semantic body coverage
    layer order
    compressed thickness envelope
    shell-to-shell clearance
    body occlusion mask
    collision pair policy
    secondary-motion priority
    compatible and incompatible feature pairs
    runtime budget profile
    ```

    기본 layer order:

    ```text
    BODY
    BASE
    INNER
    MID
    OUTER
    ARMOR
    ACCESSORY
    ```

    같은 order에서도 semantic region별로 국소 우선순위를 가질 수 있다. Outfit compile은 기존
    garment topology를 수정하지 않고 hide mask, collision admission과 runtime plan만 발행한다.

    ---

    ## 9. 리깅 제품의 기본 원칙

    1. Skeleton semantic ID가 bone name보다 우선한다.
    2. 외부 캐릭터 rig는 adapter를 통해 canonical skeleton에 매핑한다.
    3. body surface의 skin weight와 semantic region은 직접 정본으로 고정한다.
    4. garment weight는 surface barycentric transfer, region admission과 seam/layer 규칙으로 계산한다.
    5. full-pose morph는 진단 fixture로만 보존하고 runtime 기본 경로는 skeletal skinning이다.
    6. skinning 잔차는 작은 국소 corrective로 처리한다.
    7. loose 영역만 secondary motion을 허용한다.
    8. bonded interfacing, button과 rigid hardware는 별도 bind policy를 사용한다.
    9. Godot는 import와 장착 소비자이며 topology authoring 도구가 아니다.
    10. 실패 시 equip transaction 전체를 원자적으로 취소한다.

    ---

    ## 10. 완료 정의

    R1B COMPLETE는 다음이 모두 성립해야 한다.

    ```text
    canonical humanoid skeleton and adapter contract
    body skin-source package
    automatic bind for accepted R1A tunic and trousers
    local corrective system
    Godot equip/unequip and character swap
    outfit layer and occlusion compiler
    sleeved tunic and straight-sleeve robe generalization
    rig-aware LOD and secondary motion
    at least six garment families admitted by one registry
    fixed skeletal motion suite PASS
    fresh-process GLB/Godot reopen PASS
    immutable R1A predecessor preserved
    ```

    R1B가 끝나기 전까지 새 의상군을 단순 복제 방식으로 대량 추가하지 않는다.
    '''


def skeleton_design() -> str:
    return r'''
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
    '''


def binding_design() -> str:
    return r'''
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
    '''


def outfit_design() -> str:
    return r'''
    # Multi-Garment Library / Outfit Composition 설계

    ## 1. 목적

    의상 종류가 늘어도 각 의상이 독립적으로 build되고, 호환되는 제품은 전체 재시뮬레이션 없이
    outfit으로 조합되도록 한다.

    ## 2. Registry

    `GarmentLibraryRegistry/1`은 다음 index만 소유한다.

    ```text
    garment family ID
    available variant IDs
    current product version
    semantic slots and coverage
    required skeleton contract
    supported body-profile range
    layer class
    build product references
    qualification status
    license/source provenance
    ```

    Registry는 의상 구현 코드를 소유하지 않는다. 각 family 커널의 product receipt를 참조한다.

    ## 3. Slots와 coverage

    단일 slot 독점 모델을 피한다.

    ```text
    slots
    torso_inner / torso_mid / torso_outer
    lower_inner / lower_outer
    arm_inner / arm_outer
    hand / foot / head_neck
    full_body_inner / full_body_outer
    armor_underlayer / accessory

    coverage regions
    chest_front / chest_back
    waist / pelvis
    left/right upper arm / forearm
    left/right thigh / calf
    neck / head / hand / foot
    ```

    긴 robe는 여러 coverage region과 slot을 동시에 소유할 수 있다.

    ## 4. Layering

    `LayerStackProfile/1`:

    ```text
    layer class
    nominal thickness
    compressible thickness range
    minimum clearance
    friction pair class
    collision priority
    hide-body policy
    secondary-motion priority
    ```

    Outfit compile은 다음을 판정한다.

    ```text
    skeleton compatibility
    slot and coverage compatibility
    total local thickness budget
    shell intersection risk
    closure obstruction
    hem and sleeve motion-domain conflict
    corrective driver conflict
    runtime budget
    ```

    ## 5. Body occlusion

    `BodyOcclusionMask/1`은 body triangle semantic regions로 발행한다. 의상 내부의 보이지 않는 body
    surface만 숨기며 관절 주변 safety band를 유지한다. Texture alpha에 종속된 수기 mask를 정본으로
    삼지 않는다.

    ## 6. Atomic outfit transaction

    ```text
    resolve products
    → validate rig adapters
    → validate layer stack
    → compile body hide mask
    → compile collision and secondary-motion plan
    → preview
    → atomic equip commit
    ```

    중간 실패 시 기존 outfit과 body mask를 유지한다. Equip과 unequip은 idempotent해야 한다.

    ## 7. Variant 확장

    같은 family의 다음 차이는 variant로 관리할 수 있다.

    ```text
    color/material assignment
    trim and closure option
    bounded length option
    accepted size/body profile
    compatible lining option
    ```

    panel 수, seam graph, sleeve topology나 bind-region ownership이 달라지는 경우는 topology variant
    또는 새 family다.

    ## 8. Library scale acceptance

    ```text
    duplicate family ID                 0
    missing product receipt             0
    unresolved skeleton contract        0
    incompatible outfit admitted        0
    body mask outside coverage          0
    layer order cycle                   0
    registry → product hash mismatch    0
    ```
    '''


def runtime_design() -> str:
    return r'''
    # Rig-aware LOD / Secondary Motion 설계

    ## 1. Primary와 secondary deformation 분리

    ```text
    skeletal skinning
    + sparse local corrective
    + optional region-local secondary motion
    ```

    모든 정점을 runtime cloth 대상으로 만들지 않는다.

    ## 2. Secondary-motion domain

    허용 예:

    ```text
    loose hem
    skirt panel
    robe tail
    loose sleeve lower section
    pleat free edge
    closure loop
    hanging accessory
    ```

    기본 제외:

    ```text
    bonded interfacing
    button / buckle
    reinforced facing
    waistband core
    tight upper torso
    ```

    `SecondaryMotionProfile/1`:

    ```text
    domain ID and owner vertices
    anchor vertices / semantic bones
    calibrated mass and damping
    stretch and bending response
    collision subset
    outfit interaction priority
    update frequency tier
    deterministic fallback
    LOD disposition
    ```

    Backend는 교체 가능하지만 계약은 공통이다. Godot runtime에서 제품 topology를 수정하지 않는다.

    ## 3. LOD ownership

    `RiggedGarmentLODSet/1`은 LOD별 독립 mesh와 다음 transfer를 소유한다.

    ```text
    LOD0 → LODn surface map
    bone-weight transfer map
    corrective transfer map
    component and feature retention map
    secondary-motion domain reduction
    material-zone map
    silhouette-critical edge set
    ```

    ### LOD0

    full 14-component 또는 family-complete 제품, 모든 corrective와 feature silhouette 유지.

    ### LOD1

    facing/interfacing의 내부 상세는 병합 가능하나 dart, pleat, gather, gusset과 closure silhouette는
    보존한다. Bone semantic identity는 LOD0와 동일하다.

    ### LOD2

    내부 lining topology와 비가시 hardware 상세를 축약할 수 있다. 주요 hem, sleeve, collar와 closure
    silhouette를 보존하며 secondary motion을 축약한다.

    ## 4. 전환 gate

    ```text
    bone map identity                  exact
    weight normalization              PASS
    corrective semantic owner         preserved
    visible silhouette jump           bounded
    body penetration after switch      0
    material zone loss                 0 for visible zones
    runtime topology mutation          0
    ```

    ## 5. Runtime budget

    절대 수치를 전 의상에 강제하지 않고 `RuntimeBudgetProfile/1`로 관리한다.

    ```text
    target platform
    maximum active garments
    maximum skinned vertices by LOD
    maximum corrective channels
    maximum secondary-motion particles
    update frequency
    memory and upload budget
    ```

    성능을 맞추기 위해 garment feature authority를 삭제하지 않고 더 낮은 LOD product를 선택한다.
    '''


def godot_design() -> str:
    return r'''
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
    '''


def acceptance_design() -> str:
    return r'''
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
    '''


def roadmap() -> str:
    return r'''
    # GARMENT-CAD-PRO-R1B 구현 로드맵

    ## 프로그램 목표

    R1A CP6-R1 완료본을 immutable predecessor로 유지하고, 제조·simulation authority를 실제 게임
    캐릭터 Skeleton에 장착되는 다수 의상군 제품으로 전환한다.

    ## CP0 — Canonical skeleton / body skin source / compatibility

    ```text
    CanonicalSkeletonPackage/1
    SkeletonAdapterMap/1
    BodySkinSourcePackage/1
    SkeletonCompatibilityReceipt/1
    ```

    기준 humanoid rig와 body skin source를 발행하고 현재 캐릭터/더미 rig를 adapter fixture로
    입장시킨다. 의상 weight transfer는 아직 수행하지 않는다.

    ## CP1 — Automatic bind pilot / tunic + trousers

    수락된 R1A sleeveless tunic과 trousers를 대상으로 body-surface barycentric transfer, semantic region,
    seam/layer regularization과 hardware bind를 구현한다.

    ```text
    SkinWeightField/1
    RigBindPlan/1
    tunic automatic bind
    trousers automatic bind
    ```

    ## CP2 — Skeletal motion / local corrective

    10-pose suite를 skeletal deformation으로 재실행하고 shoulder, underarm, elbow, waist, pelvis, crotch,
    knee와 stride의 sparse local corrective를 구현한다. Full-body pose morph는 진단 비교만 유지한다.

    ## CP3 — Godot equip product / character swap

    ```text
    RiggedGarmentProduct/1
    GodotGarmentEquipPackage/1
    atomic equip / unequip
    Skeleton adapter validation
    body occlusion
    character swap
    ```

    정확한 Godot 4.7.2 Linux import/reopen과 실제 캐릭터 장착을 닫는다.

    ## CP4 — Multi-garment registry / outfit layering

    GarmentLibraryRegistry, OutfitAssemblyPlan, semantic coverage, layer thickness, body hide mask와 collision
    admission을 구현한다. Tunic + trousers 조합과 incompatible fixture를 검증한다.

    ## CP5 — Sleeved tunic / straight-sleeve robe generalization

    직접 팔 치수와 sleeve-cap construction을 소비해 sleeved tunic과 straight-sleeve robe를 실제
    rig product로 추가한다. Shoulder, elbow와 underarm gusset corrective를 일반화한다.

    ## CP6 — Rig-aware LOD / secondary motion

    LOD0/1/2의 bone-weight, corrective와 feature transfer를 구현한다. Hem, loose sleeve, pleat와 robe tail의
    bounded secondary-motion domain을 추가하고 runtime budget profile을 검증한다.

    ## CP7 — Library-scale closeout

    최소 다음 여섯 family를 하나의 registry와 outfit compiler에서 닫는다.

    ```text
    sleeveless tunic
    trousers
    sleeved tunic
    straight-sleeve robe
    skirt or divided skirt
    jacket or coat
    ```

    완료 gate:

    ```text
    all family products have rig receipts
    fixed motion qualification PASS
    at least three multi-garment outfits PASS
    incompatible combinations reject atomically
    LOD and secondary-motion PASS
    Godot import/equip/unequip/character swap PASS
    R1A predecessor unchanged
    ```

    ## 첫 실제 작업

    **`GARMENT-CAD-PRO-R1B / CP0 — CANONICAL HUMANOID SKELETON / BODY SKIN-SOURCE / RIG COMPATIBILITY CONTRACT`**

    R1A의 body, anthropometry와 accepted game products를 read-only 입력으로 사용한다. 기준 semantic
    skeleton, 외부 rig adapter, canonical body weights, surface barycentric source와 compatibility receipt를
    구현한다. Garment skin transfer, corrective, secondary motion과 Godot equip은 아직 수행하지 않는다.
    '''


def index_doc() -> str:
    return r'''
    # GARMENT-CAD-PRO-R1B 설계 정본 인덱스

    ```text
    status          DESIGN_COMPLETE / IMPLEMENTATION_NOT_STARTED
    predecessor     GARMENT-CAD-PRO-R1A COMPLETE
    next checkpoint R1B CP0
    ```

    문서:

    - `GARMENT_CAD_PRO_R1B_GAME_GARMENT_PLATFORM_KO.md` — 전체 authority와 커널 구조
    - `CANONICAL_SKELETON_AND_BODY_BIND_CONTRACT_KO.md` — skeleton, adapter와 body skin source
    - `SKIN_WEIGHT_TRANSFER_AND_CORRECTIVES_KO.md` — 자동 weight와 국소 corrective
    - `MULTI_GARMENT_LIBRARY_AND_OUTFIT_COMPOSITION_KO.md` — 다수 의상 registry와 layer 조합
    - `RUNTIME_SECONDARY_MOTION_AND_LOD_KO.md` — secondary motion과 LOD
    - `GODOT_RIGGED_GARMENT_PRODUCT_CONTRACT_KO.md` — Godot 장착 제품
    - `RIGGED_GARMENT_ACCEPTANCE_MATRIX_KO.md` — 제품 acceptance
    - `../../roadmap/GARMENT_CAD_PRO_R1B_ROADMAP_KO.md` — CP0–CP7 구현 순서

    R1A 문서는 제조·simulation 정본으로 계속 유효하다. R1B 문서는 해당 정본을 게임 Skeleton과
    outfit runtime에 연결하는 후속 authority다.
    '''


def start_here() -> str:
    return f'''
    # R1B Start Here

    ## 현재 상태

    ```text
    predecessor               GARMENT-CAD-PRO-R1A / CP6-R1
    predecessor decision      GARMENT_CAD_PRO_R1A_COMPLETE
    R1B design                COMPLETE
    R1B implementation        NOT STARTED
    next checkpoint           GARMENT_CAD_PRO_R1B_CP0
    ```

    설계 정본은 `docs/architecture/professional_rig/README.md`에서 시작한다.

    다음 실제 작업:

    **`GARMENT-CAD-PRO-R1B / CP0 — CANONICAL HUMANOID SKELETON / BODY SKIN-SOURCE / RIG COMPATIBILITY CONTRACT`**

    CP0는 R1A 완료본을 read-only 입력으로 사용하며 garment skin transfer는 아직 수행하지 않는다.
    '''


def next_task() -> str:
    return r'''
    # Next task

    **`GARMENT-CAD-PRO-R1B / CP0 — CANONICAL HUMANOID SKELETON / BODY SKIN-SOURCE / RIG COMPATIBILITY CONTRACT`**

    `GARMENT-CAD-PRO-R1A / CP6-R1`을 immutable predecessor로 유지한다. 기준 semantic humanoid
    Skeleton, external-rig adapter, body skin weights, body surface semantic regions와 barycentric source,
    compatibility receipt를 실제 구현한다.

    CP0 제외 범위:

    ```text
    garment skin-weight transfer
    local corrective deformation
    secondary motion
    LOD generation
    Godot equip/unequip
    new garment family
    ```
    '''


def write_documents(root: Path) -> list[Path]:
    arch = root / "docs/architecture/professional_rig"
    road = root / "docs/roadmap"
    documents = {
        arch / "README.md": index_doc(),
        arch / "GARMENT_CAD_PRO_R1B_GAME_GARMENT_PLATFORM_KO.md": main_design(),
        arch / "CANONICAL_SKELETON_AND_BODY_BIND_CONTRACT_KO.md": skeleton_design(),
        arch / "SKIN_WEIGHT_TRANSFER_AND_CORRECTIVES_KO.md": binding_design(),
        arch / "MULTI_GARMENT_LIBRARY_AND_OUTFIT_COMPOSITION_KO.md": outfit_design(),
        arch / "RUNTIME_SECONDARY_MOTION_AND_LOD_KO.md": runtime_design(),
        arch / "GODOT_RIGGED_GARMENT_PRODUCT_CONTRACT_KO.md": godot_design(),
        arch / "RIGGED_GARMENT_ACCEPTANCE_MATRIX_KO.md": acceptance_design(),
        road / "GARMENT_CAD_PRO_R1B_ROADMAP_KO.md": roadmap(),
        root / "R1B_START_HERE.md": start_here(),
        root / "NEXT_TASK.md": next_task(),
    }
    for path, content in documents.items():
        _write(path, content)
    return list(documents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    documents = write_documents(root)
    hashes = {path.relative_to(root).as_posix(): _sha256(path) for path in documents}
    status = {
        "program": PROGRAM_ID,
        "title": PROGRAM_TITLE,
        "design_status": "COMPLETE",
        "implementation_status": "NOT_STARTED",
        "immutable_predecessor": "GARMENT_CAD_PRO_R1A_CP6_R1",
        "predecessor_terminal_decision": "GARMENT_CAD_PRO_R1A_COMPLETE",
        "document_count": len(documents),
        "documents": hashes,
        "next_checkpoint": "GARMENT_CAD_PRO_R1B_CP0",
        "next_task": "CANONICAL_HUMANOID_SKELETON_BODY_SKIN_SOURCE_RIG_COMPATIBILITY",
    }
    canonical = json.dumps(status, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    status["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json(root / "R1B_DESIGN_STATUS.json", status)
    print(json.dumps(status, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
