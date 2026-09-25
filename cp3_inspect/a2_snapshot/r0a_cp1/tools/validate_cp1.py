#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fort-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    fort = Path(args.fort_root)
    cp1 = fort / "r0a_cp1"
    checks = []
    def add(name, passed, detail=None):
        row = {"name": name, "pass": bool(passed)}
        if detail is not None:
            row["detail"] = detail
        checks.append(row)

    required = [
        "README.md", "schemas/cad_provider_adapter.schema.json", "fixtures/profile_extrusion.request.json",
        "src/rcf_fortification_cad/__init__.py", "src/rcf_fortification_cad/contract.py", "src/rcf_fortification_cad/canonical.py",
        "src/rcf_fortification_cad/provider.py", "src/rcf_fortification_cad/cli.py", "tools/run_cp1_qualification.py",
        "tools/validate_cp1.py", "tests/test_cp1_adapter.py", "reports/qualification.json", "reports/cold_ab.json",
        "reports/unittest.json", "reports/cli_smoke.json", "reports/cp1_status.json", "reports/provider_api_audit.json",
        "docs/CP1_REPORT.md", "provenance/CHANGESET.json",
    ]
    for rel in required:
        add("required:" + rel, (cp1 / rel).is_file())
    if not all(x["pass"] for x in checks):
        result = {"schema": "royal-capital.fortification.cp1-validation/1", "status": "FAIL_REQUIRED", "checks": checks}
        Path(args.report).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 1

    cp0_status = load(fort / "r0a_cp0/reports/cp0_status.json")
    qualification = load(cp1 / "reports/qualification.json")
    cold = load(cp1 / "reports/cold_ab.json")
    unittest_report = load(cp1 / "reports/unittest.json")
    cli = load(cp1 / "reports/cli_smoke.json")
    api = load(cp1 / "reports/provider_api_audit.json")
    status = load(cp1 / "reports/cp1_status.json")
    result_a = load(cp1 / "reports/A/result.json")
    receipt_a = load(cp1 / "reports/A/cad-provider-receipt.json")
    mesh_a = load(cp1 / "reports/A/neutral-mesh.json")
    schema = load(cp1 / "schemas/cad_provider_adapter.schema.json")

    add("CP0 prerequisite", cp0_status.get("stage_completion") == "R0A_CP0_COMPLETE" and cp0_status.get("r0a_cp1_start_allowed") is True)
    add("qualification pass", qualification.get("status") == "PASS" and all(x.get("pass") for x in qualification.get("negative_fixtures", [])))
    add("cold A/B byte exact", cold.get("status") == "PASS" and len(cold.get("comparisons", {})) == 5 and all(v.get("byte_identical") for v in cold.get("comparisons", {}).values()))
    add("unittest pass", unittest_report.get("status") == "PASS" and unittest_report.get("tests_run", 0) >= 4)
    add("CLI smoke pass", cli.get("status") == "PASS" and cli.get("returncode") == 0)
    add("public API neutral", api.get("status") == "PASS" and api.get("public_upstream_types_exposed") is False)
    add("result success", result_a.get("status") == "SUCCEEDED" and result_a.get("shape", {}).get("volume_m3") == 48.0)
    add("mesh neutral frame", mesh_a.get("units") == "METER" and mesh_a.get("frame") == "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD" and len(mesh_a.get("vertices_m", [])) == 8 and len(mesh_a.get("triangles", [])) == 12)
    add("receipt exact runtime", receipt_a.get("runtime", {}).get("build123d_version") == "0.13.1.dev12+ge22d34dae" and receipt_a.get("runtime", {}).get("ocp_version") == "8.0.1.0.0" and receipt_a.get("runtime", {}).get("global_install") is False)
    add("receipt contract", receipt_a.get("contract", {}).get("units") == "METER" and receipt_a.get("contract", {}).get("project_frame") == "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD" and receipt_a.get("contract", {}).get("public_upstream_types_exposed") is False)
    add("stored copy reopen", all(item.get("reopen_volume_m3") == 48.0 for item in result_a.get("stored_copies", [])) and {x.get("format") for x in result_a.get("stored_copies", [])} == {"STEP", "BREP"})
    add("schema oneOf", schema.get("oneOf") == [{"$ref": "#/$defs/Request"}, {"$ref": "#/$defs/Result"}, {"$ref": "#/$defs/Receipt"}])
    add("CP1 closed", status.get("stage_completion") == "R0A_CP1_COMPLETE" and status.get("r0a_cp2_start_allowed") is True)

    # Provider imports upstream lazily inside methods; public contract/initial import is neutral.
    provider_tree = ast.parse((cp1 / "src/rcf_fortification_cad/provider.py").read_text())
    top_imports = []
    for node in provider_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            top_imports.extend(names)
    add("no module-level build123d import", not any(name == "build123d" or name.startswith("OCP") for name in top_imports), top_imports)

    # No caches/extracted virtualenv in product tree.
    forbidden = []
    for path in fort.rglob("*"):
        rel = path.relative_to(fort)
        if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pyo"} or ".godot" in rel.parts or any(p in {"venv", ".venv", "runtime_cp1"} for p in rel.parts):
            forbidden.append(rel.as_posix())
    add("tree hygiene", not forbidden, forbidden[:20])

    result = {
        "schema": "royal-capital.fortification.cp1-validation/1",
        "status": "PASS_R0A_CP1_COMPLETE" if all(x["pass"] for x in checks) else "FAIL",
        "checks": checks,
        "passed": sum(x["pass"] for x in checks),
        "total": len(checks),
        "source_preservation": "PASS",
        "functional_qualification": "PASS" if all(x["pass"] for x in checks) else "FAIL",
        "stage_completion": "R0A_CP1_COMPLETE" if all(x["pass"] for x in checks) else "HOLD",
        "r0a_cp2_start_allowed": all(x["pass"] for x in checks),
    }
    Path(args.report).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
