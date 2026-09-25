#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp1-root", required=True)
    parser.add_argument("--cp0-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--reports-root", required=True)
    args = parser.parse_args()
    cp1 = Path(args.cp1_root).resolve()
    sys.path.insert(0, str(cp1 / "src"))
    from rcf_fortification_cad import Build123dProviderAdapter, CadStatus

    work = Path(args.work_root).resolve()
    reports = Path(args.reports_root).resolve()
    if work.exists() or reports.exists():
        raise SystemExit("work-root and reports-root must both be fresh")
    work.mkdir(parents=True)
    reports.mkdir(parents=True)
    adapter = Build123dProviderAdapter(args.cp0_root)
    fixture_dir = cp1 / "fixtures"

    request = json.loads((fixture_dir / "profile_extrusion.request.json").read_text(encoding="utf-8"))
    successful = []
    for label in ("A", "B"):
        out = work / label
        result = adapter.execute(request, out)
        if result.status is not CadStatus.SUCCEEDED:
            raise SystemExit(f"successful fixture {label} failed: {result}")
        successful.append(out)
        shutil.copytree(out, reports / label)

    compared = ["neutral-mesh.json", "cad-provider-receipt.json", "result.json", "shape.step", "shape.brep"]
    comparisons = {name: {"byte_identical": (successful[0] / name).read_bytes() == (successful[1] / name).read_bytes(), "a_sha256": sha(successful[0] / name), "b_sha256": sha(successful[1] / name)} for name in compared}
    cold = {
        "schema": "royal-capital.fortification.cp1-cold-ab/1",
        "status": "PASS" if all(item["byte_identical"] for item in comparisons.values()) else "FAIL",
        "comparisons": comparisons,
    }
    (reports / "cold_ab.json").write_text(json.dumps(cold, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cold["status"] != "PASS":
        raise SystemExit("cold A/B mismatch")

    expected_failures = {
        "invalid_units.request.json": "UNIT_CONTRACT_MISMATCH",
        "invalid_frame.request.json": "FRAME_CONTRACT_MISMATCH",
        "invalid_tolerance.request.json": "TOLERANCE_OUT_OF_DOMAIN",
        "runtime_mismatch.request.json": "CAD_RUNTIME_VERSION_MISMATCH",
        "budget_exceeded.request.json": "GEOMETRY_BUDGET_EXCEEDED",
    }
    negatives = []
    for filename, expected_code in expected_failures.items():
        request_data = json.loads((fixture_dir / filename).read_text(encoding="utf-8"))
        out = work / filename.replace(".request.json", "")
        result = adapter.execute(request_data, out)
        actual_code = None if result.failure is None else result.failure["code"]
        files = sorted(p.name for p in out.iterdir())
        passed = result.status is CadStatus.REJECTED and actual_code == expected_code and files == ["result.json"]
        negatives.append({"fixture": filename, "expected_code": expected_code, "actual_code": actual_code, "status": result.status.value, "files": files, "pass": passed})
        shutil.copytree(out, reports / out.name)
    qualification = {
        "schema": "royal-capital.fortification.cp1-qualification/1",
        "status": "PASS" if all(item["pass"] for item in negatives) else "FAIL",
        "successful_fixture": {
            "request": "profile_extrusion.request.json",
            "volume_m3": json.loads((successful[0] / "result.json").read_text())["shape"]["volume_m3"],
            "cold_ab": cold["status"],
        },
        "negative_fixtures": negatives,
        "global_install": False,
    }
    (reports / "qualification.json").write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(qualification, indent=2, sort_keys=True))
    return 0 if qualification["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
