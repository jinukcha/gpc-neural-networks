#!/usr/bin/env python3
"""Publish GARMENT-SIZING-R0A CP1 without meshing or Warp execution."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wuxia_garment_oss.garments.sleeveless_tunic.reference import (
    body_fixtures,
    reference_size_table,
    request_fixtures,
)
from wuxia_garment_oss.garments.sleeveless_tunic.resolver import resolve_tunic_instance
from wuxia_garment_oss.sizing.contracts import contract_schemas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_contracts(root: Path) -> None:
    for name, schema in contract_schemas().items():
        write_json(root / "contracts/sizing" / name, schema)


def execute_fixtures(root: Path) -> tuple[list[dict], dict]:
    table = reference_size_table()
    bodies = body_fixtures()
    build = root / "build/tunic_pilot/sizing_cp1"
    results = []
    write_json(root / "fixtures/sizing/cp2b_tunic_size_table.json", table.to_dict())
    for name, request, body_name in request_fixtures():
        body = bodies.get(body_name) if body_name else None
        instance, receipt = resolve_tunic_instance(request, table, body)
        write_json(build / "selection_receipts" / f"{name}.json", receipt)
        write_json(build / "sized_pattern_instances" / f"{name}.json", instance)
        results.append({"name": name, "receipt": receipt, "instance": instance})
    for name, body in bodies.items():
        write_json(root / "fixtures/sizing/bodies" / f"{name}.json", body.to_dict())
    return results, table.to_dict()


def _score_matrix(results: list[dict], size_ids: list[str]) -> np.ndarray:
    matrix = np.full((len(results), len(size_ids)), np.nan, dtype=np.float64)
    for row, result in enumerate(results):
        for score in result["receipt"]["ranked_sizes"]:
            if score["size_id"] in size_ids:
                matrix[row, size_ids.index(score["size_id"])] = float(score["score"])
    return matrix


def render_evidence(path: Path, results: list[dict], size_ids: list[str]) -> None:
    labels = [result["name"].replace("AUTO_", "") for result in results]
    matrix = _score_matrix(results, size_ids)
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    ax0 = fig.add_subplot(grid[0, 0])
    image = ax0.imshow(np.ma.masked_invalid(matrix), aspect="auto")
    ax0.set_xticks(range(len(size_ids)), size_ids)
    ax0.set_yticks(range(len(labels)), labels)
    ax0.set_title("Normalized multi-dimensional size scores")
    fig.colorbar(image, ax=ax0, label="lower is better")
    ax1 = fig.add_subplot(grid[0, 1])
    ax1.axis("off")
    lines = ["fixture | selected | shape | height | admission", "-" * 72]
    for result in results:
        receipt = result["receipt"]
        lines.append(
            f"{result['name']:<22} {receipt['selected_size_id']:<4} "
            f"{receipt['shape_block']:<16} {receipt['height_block']:<7} "
            f"{receipt['admission']}"
        )
    ax1.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax2 = fig.add_subplot(grid[1, :])
    selected_scores = []
    for result in results:
        receipt = result["receipt"]
        scores = {row["size_id"]: row["score"] for row in receipt["ranked_sizes"]}
        selected_scores.append(float(scores.get(receipt["selected_size_id"], 0.0)))
    ax2.bar(np.arange(len(labels)), selected_scores)
    ax2.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax2.set_ylabel("selected-size normalized score")
    ax2.set_title("Selection and alteration admission evidence")
    fig.suptitle("GARMENT-SIZING-R0A / CP1 — no triangulation, no Warp simulation")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report_text(summary: dict) -> str:
    return f"""# GARMENT-SIZING-R0A / CP1 실행 보고서

## 판정

```text
terminal decision              CP1_COMPLETE_SELECTION_ONLY
fixture count                  {summary['fixture_count']}
size-table cardinality         {summary['size_table_cardinality']}
triangulation executed         false
Warp simulation executed       false
```

## 구현

다차원 normalized size score, shape/height block 선택, standard grade와 custom
alteration 분리, 과도 보정 admission 및 canonical selection receipt를 구현했다.
`REGULAR`, `BROAD_SHOULDER`, `FULL_CHEST`, `FULL_ABDOMEN`, `TALL`, `SHORT`를
실제 fixture로 실행했다. 결과가 지원 범위를 넘으면 자동 mesh scaling을 하지 않고
`ALTERNATE_BLOCK_REQUIRED`, `TOPOLOGY_CHANGE_REQUIRED`, `HOLD`를 발행한다.

## 상태 분포

```json
{json.dumps(summary['admission_counts'], indent=2, sort_keys=True)}
```
"""


def publish_status(root: Path, results: list[dict], table: dict) -> dict:
    counts = Counter(result["receipt"]["admission"] for result in results)
    summary = {
        "checkpoint": "GARMENT_SIZING_R0A_CP1",
        "terminal_decision": "CP1_COMPLETE_SELECTION_ONLY",
        "fixture_count": len(results),
        "size_table_cardinality": len(table["sizes"]),
        "admission_counts": dict(sorted(counts.items())),
        "triangulation_executed": False,
        "warp_simulation_executed": False,
        "mesh_scaling": "FORBIDDEN",
        "next_checkpoint": "GARMENT_SIZING_R0A_CP2",
    }
    write_json(root / "build/tunic_pilot/sizing_cp1/cp1_receipt.json", summary)
    write_json(root / "SIZING_STATUS.json", summary)
    return summary


def append_once(path: Path, marker: str, text: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in current:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(current.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def update_docs(root: Path) -> None:
    append_once(
        root / "docs/architecture/GARMENT_SIZING_SYSTEM_KO.md",
        "## CP1 구현 상태",
        """## CP1 구현 상태

CP1은 튜닉 family의 다차원 size score, shape/height block, standard grade와
custom alteration 분리, excessive-alteration admission을 구현한다. 이 단계는
2D pattern parameter를 변경하지 않고 selection receipt만 발행한다.""",
    )
    append_once(
        root / "docs/roadmap/GARMENT_SIZING_R0A_ROADMAP_KO.md",
        "## CP1 결과",
        """## CP1 결과

- normalized size score와 shape/height block 구현
- grade/custom alteration 분리와 terminal admission 구현
- triangulation/Warp 미실행
- 다음 단계: CP2 parametric tunic POM/landmark resolver""",
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    publish_contracts(root)
    results, table = execute_fixtures(root)
    evidence = root / "build/tunic_pilot/sizing_cp1/cp1_selection_evidence.png"
    render_evidence(evidence, results, [row["size_id"] for row in table["sizes"]])
    summary = publish_status(root, results, table)
    report = root / "docs/cp2b/GARMENT_SIZING_CP1_EXECUTION_REPORT_KO.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text(summary), encoding="utf-8")
    update_docs(root)
    (root / "NEXT_TASK.md").write_text(
        "# Next task\n\n"
        "`GARMENT-SIZING-R0A / CP2 — TUNIC S/M/L + DETAILED-MEASUREMENT RESOLVER / "
        "PARAMETRIC POM & LANDMARK ALTERATION`\n\n"
        "Use CP1 selection receipts unchanged. Resolve actual tunic pattern parameters for "
        "S/M/L and supported custom alterations. Do not triangulate or run Warp yet.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
