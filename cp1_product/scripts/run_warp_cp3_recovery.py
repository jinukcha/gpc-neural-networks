#!/usr/bin/env python3
"""Run the bounded frame 1–180 Warp CP3 recovery and publish evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wuxia_garment_oss.drape.warp_backend.production import (
    ProductionProfile,
    qualify,
    run_production,
    write_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_arrays(root: Path) -> dict[str, np.ndarray]:
    path = root / "build/tunic_pilot/warp_cp1/warp_model_package.npz"
    with np.load(path, allow_pickle=False) as package:
        return {name: package[name] for name in package.files}


def write_outputs(root: Path, arrays: dict, result, qualification: dict) -> dict:
    output = root / "build/tunic_pilot/warp_cp3"
    output.mkdir(parents=True, exist_ok=True)
    final_mesh = output / "final_simulation_mesh.npz"
    np.savez_compressed(
        final_mesh,
        positions=result.positions_final.astype(np.float32),
        triangles=result.triangles.astype(np.int32),
        panel_ids=result.panel_ids.astype(np.int32),
    )
    write_json(output / "frame_metrics.json", {"frames": result.frame_metrics})
    write_json(output / "geometry_qualification.json", qualification)
    evidence, convergence = write_evidence(result, qualification, output)
    profile = {
        "frames": result.profile.frames,
        "substeps": result.profile.substeps,
        "iterations": result.profile.iterations,
        "fps": result.profile.fps,
        "authority_mode": "BOUNDED_CP2_RECOVERY_FROM_CP1_STATIC_MODEL",
        "cp2_source_byte_exact": False,
        "exact_triangle_self_intersection": False,
    }
    write_json(root / "config/profiles/warp_garment/tunic_pilot_cp3_recovery.json", profile)
    receipt = {
        "checkpoint": "CP3_FRAME_1_180_RECOVERY_RUN",
        "production_schedule": profile,
        "model_package_sha256": sha256(root / "build/tunic_pilot/warp_cp1/warp_model_package.npz"),
        "final_mesh_sha256": sha256(final_mesh),
        "evidence_png_sha256": sha256(evidence),
        "convergence_png_sha256": sha256(convergence),
        "qualification": qualification,
        "technical_pass": False,
        "product_acceptance": False,
        "terminal_decision": qualification["terminal_decision"],
    }
    write_json(output / "cp3_receipt.json", receipt)
    return {
        "receipt": receipt,
        "final_mesh": final_mesh,
        "evidence": evidence,
        "convergence": convergence,
    }


def report_text(receipt: dict) -> str:
    q = receipt["qualification"]
    body = q["body_contact"]
    edge = q["edge_strain"]
    seam = q["seam_closure"]
    conv = q["convergence"]
    return f"""# OSS-R0A-CP2B-R4-REV1 / CP3 실행 보고서

## 판정

```text
terminal decision              {receipt['terminal_decision']}
frames                          1–180
substeps / iterations           8 / 12
numeric qualification           {q['numeric_pass']}
exact CP2 source authority       false
exact non-sewn intersection      false
technical_pass                   false
product_acceptance               false
```

## 결과

```text
body penetration max            {body['penetration_max_m']:.9f} m
body penetrated ratio            {body['penetrated_vertex_ratio']:.9f}
edge stretch p99 / max           {edge['edge_stretch_p99']:.6f} / {edge['edge_stretch_max']:.6f}
seam gap mean / p95              {seam['seam_gap_mean_m']:.9f} / {seam['seam_gap_p95_m']:.9f} m
tail final max displacement      {conv['tail_final_max_displacement_m']:.9f} m
```

CP1 static model은 재설계하지 않았다. 현재 대화의 exact CP2 통합본은 실행면
ClientError로 읽을 수 없었고 연결 저장소에도 동일 바이트가 없어, 기존 CP1
정적 모델 위에 production projection path만 제한 재구현했다. 180프레임과 PNG
증거는 실제 실행 결과이지만 exact CP2 byte-equivalence와 exact triangle–triangle
self-intersection gate가 없으므로 CP2B 제품 수락에는 사용하지 않는다.
"""


def publish_status(root: Path, receipt: dict) -> None:
    q = receipt["qualification"]
    status = {
        "checkpoint": "CP3_RECOVERY_RUN_COMPLETE",
        "simulation_frames_executed": 180,
        "substeps": 8,
        "iterations": 12,
        "numeric_pass": q["numeric_pass"],
        "authority_exact_cp2": False,
        "technical_pass": False,
        "product_acceptance": False,
        "cp2c_admission": "BLOCKED_BY_CP2B",
        "terminal_decision": receipt["terminal_decision"],
    }
    checks = {
        "frame_1_180": True,
        "review_tail_161_180": True,
        "geometry_qualification": q,
        "png_evidence": True,
        "exact_cp2_authority": False,
        "exact_nonsewn_self_intersection": False,
    }
    write_json(root / "STATUS.json", status)
    write_json(root / "CHECKS.json", checks)
    report = root / "docs/cp2b/CP2B_CP3_EXECUTION_REPORT_KO.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text(receipt), encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`OSS-R0A-CP2B-R4-REV1 / CP3-R1 — EXACT CP2 SOURCE REBIND / "
        "TRIANGLE SELF-INTERSECTION GATE / NUMERIC DEFECT REPAIR`\n\n"
        "Rebind the materialized exact CP2 source and preserve this recovery run only "
        "as visual evidence. Re-run only failed numeric or authority gates; do not "
        "redraft the R2 pattern or rebuild CP1 ownership.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    arrays = load_arrays(root)
    profile = ProductionProfile()
    result = run_production(arrays, root / "build/tunic_pilot/warp_cp3", profile)
    qualification = qualify(result, arrays)
    outputs = write_outputs(root, arrays, result, qualification)
    publish_status(root, outputs["receipt"])
    print(json.dumps(outputs["receipt"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
