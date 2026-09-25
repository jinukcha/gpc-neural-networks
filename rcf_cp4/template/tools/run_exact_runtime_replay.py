#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import sys


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalized_step_sha256(value: bytes) -> str:
    text = value.decode("ascii").replace("\r\n", "\n")
    normalized, count = re.subn(
        r"FILE_NAME\(\s*'[^']*'\s*,\s*'[^']*'",
        "FILE_NAME('<NORMALIZED_NAME>','1970-01-01T00:00:00'",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("STEP FILE_NAME header not found exactly once")
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def capability_probe(output: Path) -> dict[str, object]:
    import build123d as b3d

    output.mkdir(parents=True, exist_ok=False)
    checks: dict[str, bool] = {"import": True}

    with b3d.BuildPart() as extruded:
        with b3d.BuildSketch():
            b3d.Rectangle(10, 6)
        b3d.extrude(amount=4)
    checks["extrude"] = extruded.part.volume > 0

    path = b3d.Line((0, 0, 0), (0, 0, 12))
    profile = b3d.Circle(1).face()
    checks["sweep"] = b3d.sweep(profile, path).volume > 0

    with b3d.BuildPart() as lofted:
        with b3d.BuildSketch(b3d.Plane.XY):
            b3d.Circle(2)
        with b3d.BuildSketch(b3d.Plane.XY.offset(8)):
            b3d.Circle(4)
        b3d.loft()
    checks["loft"] = lofted.part.volume > 0

    box = b3d.Box(8, 8, 8)
    cutter = b3d.Cylinder(2, 12).translate((4, 4, -2))
    checks["boolean_union"] = (box + cutter).volume > box.volume
    cut = box - cutter
    checks["boolean_cut"] = 0 < cut.volume < box.volume

    vertices, triangles = extruded.part.tessellate(0.1)
    checks["tessellate"] = bool(vertices) and bool(triangles)

    step_buffer = BytesIO()
    brep_buffer = BytesIO()
    checks["step_export"] = bool(b3d.export_step(extruded.part, step_buffer, unit=b3d.Unit.MM, timestamp="1970-01-01T00:00:00"))
    checks["brep_export"] = bool(b3d.export_brep(extruded.part, brep_buffer))
    step_bytes = step_buffer.getvalue()
    brep_bytes = brep_buffer.getvalue()
    step_path = output / "probe.step"
    brep_path = output / "probe.brep"
    step_path.write_bytes(step_bytes)
    brep_path.write_bytes(brep_bytes)
    checks["step_reopen"] = abs(b3d.import_step(step_path).volume - extruded.part.volume) < 1e-7
    checks["brep_reopen"] = abs(b3d.import_brep(brep_path).volume - extruded.part.volume) < 1e-7

    canonical = {
        "build123d_version": importlib.metadata.version("build123d"),
        "ocp_version": importlib.metadata.version("cadquery-ocp-novtk"),
        "volume": round(float(extruded.part.volume), 9),
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "step_normalized_sha256": normalized_step_sha256(step_bytes),
        "brep_sha256": hashlib.sha256(brep_bytes).hexdigest(),
    }
    result = {
        "schema": "royal-capital.fortification.cp4-capability-replay/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "canonical": canonical,
        "raw_evidence": {
            "step_raw_sha256": sha256_bytes(step_bytes),
            "brep_raw_sha256": sha256_bytes(brep_bytes),
        },
    }
    (output / "canonical.json").write_bytes(canonical_json_bytes(canonical))
    (output / "result.json").write_bytes(pretty_json_bytes(result))
    if result["status"] != "PASS":
        raise RuntimeError(f"capability replay failed: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    fortification = tree / "RC_K0/child_designs/fortification"
    cp0 = fortification / "r0a_cp0"
    cp1 = fortification / "r0a_cp1"
    cp2 = fortification / "r0a_cp2"
    cp3 = fortification / "r0a_cp3"
    sys.path[:0] = [str(cp1 / "src"), str(cp2 / "src"), str(cp3 / "src")]

    from rcf_fortification_cp3 import QualifiedStraightWallSpanPublisher

    wheelhouse = cp0 / "runtime/wheelhouse"
    wheels = sorted(wheelhouse.glob("*.whl"))
    if len(wheels) != 58:
        raise RuntimeError(f"wheel count mismatch: {len(wheels)}")
    manifest = cp0 / "runtime/WHEELHOUSE.json"
    lock = cp0 / "runtime/requirements-lock.txt"
    if platform.python_version() != "3.13.5":
        raise RuntimeError(f"Python mismatch: {platform.python_version()}")

    probe = capability_probe(output / "capability")
    fixture = json.loads((cp2 / "fixtures/straight_wall_span.fixture.json").read_text(encoding="utf-8"))
    product = output / "product"
    failure_root = output / "failures"
    result = QualifiedStraightWallSpanPublisher(cp0).execute(
        fixture,
        product,
        failure_root,
        failure_case="unexpected-positive-failure",
    )
    if result.get("status") != "SUCCEEDED":
        raise RuntimeError(f"positive CP1/CP2/CP3 replay failed: {result}")
    if any(failure_root.rglob("failure-result.json")):
        raise RuntimeError("unexpected failure receipt in positive replay")

    runtime_receipt = {
        "schema": "royal-capital.fortification.cp4-runtime-replay-receipt/1",
        "status": "PASS",
        "label": args.label,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "isolated_venv": sys.prefix != sys.base_prefix,
        },
        "runtime": {
            "build123d_version": importlib.metadata.version("build123d"),
            "ocp_version": importlib.metadata.version("cadquery-ocp-novtk"),
            "wheel_count": len(wheels),
            "wheelhouse_manifest_sha256": sha256_path(manifest),
            "requirements_lock_sha256": sha256_path(lock),
            "global_install": False,
        },
        "capability_canonical_sha256": sha256_path(output / "capability/canonical.json"),
        "product_result_sha256": sha256_path(product / "result.json"),
    }
    if not runtime_receipt["python"]["isolated_venv"]:
        raise RuntimeError("replay is not running in an isolated venv")
    (output / "runtime-receipt.json").write_bytes(pretty_json_bytes(runtime_receipt))
    print(json.dumps(runtime_receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
