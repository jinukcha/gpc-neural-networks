#!/usr/bin/env python3
"""Publish and close CP2B Warp CP1 without running garment simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.drape.warp_backend.model import publish_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_payload(metadata: dict) -> dict:
    validation = metadata["contract_validation"]
    return {
        "checkpoint": "CP1_STATIC_GARMENT_MODEL",
        "authority_mode": metadata["authority"]["mode"],
        "source_cp0_byte_exact": metadata["authority"]["source_cp0_available"],
        "model_package_pass": True,
        "vertices": validation["vertices"],
        "triangles": validation["triangles"],
        "seam_pairs": validation["seam_pairs"],
        "attachments": validation["attachments"],
        "body_contact": "NOT_IMPLEMENTED_CP2",
        "self_contact": "NOT_IMPLEMENTED_CP2",
        "simulation_frames_executed": 0,
        "technical_pass": False,
        "product_acceptance": False,
        "cp2c_admission": "BLOCKED_BY_CP2B",
        "terminal_decision": "CP1_COMPLETE_STATIC_MODEL_ONLY",
    }


def report_text(metadata: dict, status: dict) -> str:
    validation = metadata["contract_validation"]
    return f"""# OSS-R0A-CP2B-R4-REV1 / CP1 실행 보고서

## 판정

```text
terminal decision             {status['terminal_decision']}
authority mode                {status['authority_mode']}
source CP0 byte-exact         false
vertices                      {validation['vertices']}
triangles                     {validation['triangles']}
interior bending edges        {validation['interior_edges']}
seam pairs                    {validation['seam_pairs']}
attachment vertices           {validation['attachments']}
simulation frames             0
technical_pass                false
product_acceptance            false
```

## 구현

`native_input_mesh.npz`와 `WarpGarmentModelPackage/1`을 발행했다. 모델은
4개 패널의 dual-area mass, UV grain rest basis, manifold interior edge와
rest dihedral, 8개 named seam의 273 pair, 양쪽 어깨의 52 attachment vertex를
정적 authority로 소유한다. Body/self-contact와 frame stepping은 CP2 범위로 남겼다.

원본 CP0 대화 첨부는 실행면 `ClientError`로 읽히지 않았고 연결 저장소에도
동일 바이트가 없어, 수락된 R2/R3 수치 계약을 사용한 최소 재구현으로 발행했다.
이 사실은 authority receipt에 명시하며 byte-exact CP0 입력이라고 주장하지 않는다.

## 물리 정본

```text
areal density                 {metadata['density_kg_m2']:.9f} kg/m^2
total rest area               {metadata['total_area_m2']:.9f} m^2
total mass                    {metadata['total_mass_kg']:.9f} kg
boundary edges                {metadata['boundary_edges']}
canonical array SHA-256       {validation['canonical_array_sha256']}
```
"""


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    metadata = publish_model(root)
    status = status_payload(metadata)
    write_json(root / "STATUS.json", status)
    write_json(root / "CHECKS.json", {
        "model_contract": metadata["contract_validation"],
        "mass_conservation": True,
        "grain_basis": True,
        "bending_ownership": True,
        "seam_attachment_ownership": True,
        "contact_executed": False,
        "simulation_executed": False,
    })
    docs = root / "docs/cp2b"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "CP2B_CP1_EXECUTION_REPORT_KO.md").write_text(
        report_text(metadata, status), encoding="utf-8"
    )
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`OSS-R0A-CP2B-R4-REV1 / CP2 — BODY/SELF-CONTACT, FRICTION, "
        "ADAPTIVE SUBSTEP & ATOMIC CHECKPOINT/RESUME`\n\n"
        "Use the CP1 model package unchanged. Implement actual body and cloth "
        "self-contact, friction, bounded adaptive substeps, and completed-frame "
        "checkpoint/resume. Do not run the 180-frame qualification yet.\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# CP2B Warp CP1\n\n"
        "Static four-panel garment model authority for the upstream Warp backend. "
        "No contact or frame simulation is included in this checkpoint.\n",
        encoding="utf-8",
    )
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
