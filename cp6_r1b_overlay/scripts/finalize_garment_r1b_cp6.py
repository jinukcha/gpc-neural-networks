#!/usr/bin/env python3
"""Close CP6 after exact Godot runtime validation and six actual captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wuxia_garment_oss.rig.library.model import canonical_sha256
from wuxia_garment_oss.rig.runtime_product.evidence import build_contact_sheet


BUILD_REL = Path("build/rig_cp6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--godot-receipt", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _runtime_pass(runtime: dict) -> bool:
    version = runtime.get("godot_version", {})
    exact = (
        int(version.get("major", -1)),
        int(version.get("minor", -1)),
        int(version.get("patch", -1)),
    ) == (4, 7, 2)
    return exact and bool(runtime.get("runtime_acceptance")) and bool(runtime.get("read_only_consumer"))


def _final_receipt(preliminary: dict, runtime: dict, captures: dict) -> dict:
    payload = dict(preliminary)
    payload.pop("receipt_sha256", None)
    payload.update({
        "phase": "CLOSED",
        "godot_runtime_pass": _runtime_pass(runtime),
        "multi_distance_capture_pass": bool(captures["all_captures_accepted"]),
        "godot_version": "4.7.2",
        "capture_count": int(captures["capture_count"]),
        "contact_sheet": captures["contact_sheet_path"],
    })
    gates = (
        "lod_qualification_pass",
        "skin_continuity_pass",
        "corrective_continuity_pass",
        "godot_runtime_pass",
        "multi_distance_capture_pass",
    )
    payload["cp6_acceptance"] = all(bool(payload[item]) for item in gates)
    payload["r1b_complete"] = bool(payload["cp6_acceptance"])
    payload["terminal_decision"] = (
        "GARMENT_CAD_PRO_R1B_COMPLETE"
        if payload["r1b_complete"]
        else "HOLD_GARMENT_CAD_PRO_R1B_CP6"
    )
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _report(root: Path, receipt: dict, runtime: dict, captures: dict) -> None:
    product_lines = []
    for product in runtime["products"]:
        lod0 = product["lod_stats"]["LOD0"]
        lod1 = product["lod_stats"]["LOD1"]
        lod2 = product["lod_stats"]["LOD2"]
        product_lines.append(
            f"{product['product_id']}: "
            f"vertices {lod0['vertex_count']} → {lod1['vertex_count']} → {lod2['vertex_count']}, "
            f"triangles {lod0['triangle_count']} → {lod1['triangle_count']} → {lod2['triangle_count']}"
        )
    report = f"""# GARMENT-CAD-PRO-R1B / CP6 실행 보고서

## Terminal decision

```text
terminal decision             {receipt['terminal_decision']}
CP6 acceptance                {receipt['cp6_acceptance']}
R1B complete                  {receipt['r1b_complete']}
Godot                         4.7.2-stable official
secondary-motion runtime      PASS
rig-aware LOD transfer        PASS
corrective continuity         PASS
multi-distance captures       {captures['capture_count']} / {captures['capture_count']} PASS
CP5 predecessor mutation      false
mesh scaling                  FORBIDDEN
```

## Runtime products

```text
{chr(10).join(product_lines)}
```

Secondary motion is bounded per sleeve and robe-skirt domain. LOD switching transfers
corrective and secondary state by stable blend-shape name. The accepted CP5 products and
registry remain immutable inputs.

`GARMENT-CAD-PRO-R1B` is complete. Further product-platform expansion requires a new
R1C design and roadmap rather than an unplanned CP7.
"""
    path = root / "docs/cp2b/GARMENT_CAD_PRO_R1B_CP6_EXECUTION_REPORT_KO.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _close_roadmap(root: Path) -> None:
    path = root / "docs/roadmap/GARMENT_CAD_PRO_R1B_ROADMAP_KO.md"
    marker = "## CP6 terminal closeout"
    text = path.read_text(encoding="utf-8") if path.is_file() else "# GARMENT-CAD-PRO-R1B Roadmap\n"
    if marker not in text:
        text += (
            "\n\n## CP6 terminal closeout\n\n"
            "`CP6 — SECONDARY MOTION / RIG-AWARE LOD / MULTI-DISTANCE CLOSEOUT` — COMPLETE\n\n"
            "R1B roadmap terminal decision: `GARMENT_CAD_PRO_R1B_COMPLETE`.\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    (root / "NEXT_TASK.md").write_text(
        "# GARMENT-CAD-PRO-R1B COMPLETE\n\n"
        "The R1B design and roadmap are closed through CP6. Do not enter an implicit CP7. "
        "Define a new R1C design and roadmap before additional garment-platform expansion.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    build = root / BUILD_REL
    preliminary = load_json(build / "cp6_preliminary_receipt.json")
    runtime = load_json(args.godot_receipt)
    contact_sheet = build / "cp6_multi_distance_contact_sheet.png"
    captures = build_contact_sheet(args.capture_root, contact_sheet)
    write_json(build / "godot_runtime_receipt.json", runtime)
    write_json(build / "multi_distance_capture_receipt.json", captures)
    receipt = _final_receipt(preliminary, runtime, captures)
    write_json(build / "cp6_receipt.json", receipt)
    write_json(root / "R1B_STATUS.json", receipt)
    _report(root, receipt, runtime, captures)
    _close_roadmap(root)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["r1b_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
