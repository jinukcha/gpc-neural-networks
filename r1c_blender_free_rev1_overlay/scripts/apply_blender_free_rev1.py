#!/usr/bin/env python3
"""Apply the authoritative Blender-free REV1 design amendment to an integrated R1C bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


MARKER = "## Blender-free REV1 authoritative amendment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    return parser.parse_args()


def append_once(path: Path, text: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    current = path.read_text(encoding="utf-8")
    if MARKER in current:
        return
    path.write_text(current.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def update_status(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload["design_revision"] = "BLENDER_FREE_REV1"
    payload["design_revision_status"] = "COMPLETE"
    payload["implementation_status"] = "NOT_STARTED"
    payload["blender_required"] = False
    payload["blender_disposition"] = "OPTIONAL_GUIDED_ARTIST_OVERRIDE_ONLY"
    payload["active_checkpoint"] = "GARMENT_CAD_PRO_R1C_CP4_R2_R1_REV1"
    payload["active_checkpoint_title"] = (
        "COMPONENT_BOUNDARY_METRIC_DECOMPOSITION_ISOMETRIC_ARRANGEMENT_REPAIR_"
        "DIRECT_GLB_GODOT_DIRECT_ART_AUDIT"
    )
    payload["product_acceptance_changed"] = False
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_checks(root: Path, before: dict[str, str], after: dict[str, str]) -> None:
    payload = {
        "checkpoint": "GARMENT_CAD_PRO_R1C_BLENDER_FREE_REV1_DESIGN",
        "design_status": "COMPLETE",
        "implementation_status": "NOT_STARTED",
        "blender_required": False,
        "source_tree_unchanged": before["source"] == after["source"],
        "build_tree_unchanged": before["build"] == after["build"],
        "product_tree_unchanged": before["product"] == after["product"],
        "geometry_generated": False,
        "simulation_executed": False,
        "godot_executed": False,
        "product_acceptance_changed": False,
        "next_checkpoint": "GARMENT_CAD_PRO_R1C_CP4_R2_R1_REV1",
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (root / "R1C_BLENDER_FREE_REV1_CHECKS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        if "__pycache__" in item.parts or item.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def copy_overlay(root: Path, overlay: Path) -> None:
    for relative in (
        Path("docs/architecture/professional_pattern/BLENDER_FREE_PRODUCT_PIPELINE_REV1_KO.md"),
        Path("docs/roadmap/GARMENT_CAD_PRO_R1C_BLENDER_FREE_REV1_ROADMAP_KO.md"),
        Path("R1C_BLENDER_FREE_REV1_STATUS.json"),
        Path("NEXT_TASK.md"),
    ):
        source = overlay / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    overlay = args.overlay.resolve()
    before = {name: tree_hash(root / name) for name in ("source", "build", "product")}
    copy_overlay(root, overlay)

    append_once(
        root / "docs/architecture/professional_pattern/GARMENT_CAD_PRO_R1C_MODULAR_PATTERN_PLATFORM_KO.md",
        f"""
{MARKER}

R1C 자동 제품화의 필수 경로를 `Pattern → Warp → Direct GLB → glTF Validator → Godot 4.7.2`로 변경한다.
Blender는 수기 artist override가 명시적으로 요청된 경우에만 선택적으로 사용하며, 기본 pipeline·terminal gate·패키징 선행조건이 아니다.
CP4-R1의 기존 `.blend`는 accepted predecessor evidence로 보존하지만 REV1 신규 제품은 `.blend`를 발행하지 않는다.
상세 계약은 `BLENDER_FREE_PRODUCT_PIPELINE_REV1_KO.md`를 따른다.
""",
    )
    append_once(
        root / "docs/architecture/professional_pattern/PATTERN_DRIVEN_MATERIALIZATION_PIPELINE_KO.md",
        f"""
{MARKER}

Materialization closeout은 Blender export 대신 deterministic direct GLB writer를 사용한다.
GLB는 fresh-process reopen, exact pinned Khronos glTF Validator와 Godot 4.7.2 fresh import/reopen을 통과해야 한다.
제품 수락 화면은 Godot에서 neutral gray로 생성하며 Blender Workbench 화면을 terminal visual authority로 사용하지 않는다.
""",
    )
    append_once(
        root / "docs/architecture/professional_pattern/VISUAL_PRODUCT_ACCEPTANCE_MATRIX_KO.md",
        f"""
{MARKER}

자동 foreground·luminance 검사는 `AUTOMATED_EVIDENCE_READY`만 발행한다.
`visual_review=PASS`와 `product_acceptance=true`에는 동일 Godot 카메라의 before/after 12뷰를 직접 검토한 `DirectArtReviewReceipt/1`이 필요하다.
검토 대상은 sleeve-cap, underarm, collar, cuff, loose triangle, open boundary, penetration, normal inversion과 전체 garment silhouette이다.
""",
    )
    append_once(
        root / "docs/roadmap/GARMENT_CAD_PRO_R1C_ROADMAP_KO.md",
        f"""
{MARKER}

남은 roadmap은 `GARMENT_CAD_PRO_R1C_BLENDER_FREE_REV1_ROADMAP_KO.md`를 authoritative amendment로 사용한다.
다음 단계는 `CP4-R2-R1-REV1 — COMPONENT–BOUNDARY METRIC DECOMPOSITION / ISOMETRIC ARRANGEMENT REPAIR / DIRECT GLB PRODUCT / GODOT DIRECT ART AUDIT`이다.
CP5 straight robe, CP6 runtime rebind와 CP7 LOD closeout도 Blender-free 경로만 사용한다.
""",
    )
    append_once(
        root / "R1C_START_HERE.md",
        f"""
{MARKER}

자동 제품 pipeline에서 Blender는 필수가 아니다. 새 구현은 `BLENDER_FREE_PRODUCT_PIPELINE_REV1_KO.md`와 Blender-free REV1 roadmap을 먼저 읽는다.
현재 accepted CP4-R1 제품은 immutable predecessor이며 제품 acceptance를 변경하지 않는다.
""",
    )
    update_status(root / "R1C_DESIGN_STATUS.json")
    after = {name: tree_hash(root / name) for name in ("source", "build", "product")}
    write_checks(root, before, after)
    checks = json.loads((root / "R1C_BLENDER_FREE_REV1_CHECKS.json").read_text(encoding="utf-8"))
    if not all(checks[key] for key in ("source_tree_unchanged", "build_tree_unchanged", "product_tree_unchanged")):
        raise RuntimeError("immutable predecessor tree changed during design-only revision")
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
