from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from rcf_fortification_wall_span import StraightWallSpanProducer
from rcf_fortification_wall_span.model import validate_fixture

from .coverage import (
    Cp3Error, Cp3FailureCode, RECEIPT_SCHEMA, RESULT_SCHEMA, FAILURE_SCHEMA,
    build_source_coverage, canonical_json_bytes, pretty_json_bytes,
    sha256_bytes, sha256_path, tree_digest, validate_source_coverage,
)

EXPECTED_BUILD123D = "0.13.1.dev12+ge22d34dae"
EXPECTED_OCP = "8.0.1.0.0"
EXPECTED_SOURCE_COMMIT = "e22d34dae17111e5b9fdb361317e055d0daae466"
EXPECTED_OCP_SHA = "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f"
PRODUCER_REVISION = "r0a-cp3.1"


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_path(path)}


def _file_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _map_upstream_failure(message: str) -> Cp3FailureCode:
    text = message.upper()
    if "BUDGET" in text:
        return Cp3FailureCode.GEOMETRY_BUDGET_EXCEEDED
    if "RUNTIME" in text or "VERSION" in text or "WHEEL" in text:
        return Cp3FailureCode.RUNTIME_MISMATCH
    if "PART" in text:
        return Cp3FailureCode.REQUIRED_PART_FAILED
    return Cp3FailureCode.PROVIDER_EXECUTION_FAILED


class QualifiedStraightWallSpanPublisher:
    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.cp0_root = Path(cp0_root).resolve()
        self.source = StraightWallSpanProducer(self.cp0_root)

    def execute(
        self,
        fixture: Mapping[str, Any],
        output_dir: str | os.PathLike[str],
        failure_root: str | os.PathLike[str],
        *,
        expected_fixture_digest: str | None = None,
        max_published_bytes: int = 1_000_000_000,
        fault: str | None = None,
        failure_case: str = "failure",
    ) -> dict[str, Any]:
        output = Path(output_dir).resolve()
        failures = Path(failure_root).resolve()
        failures.mkdir(parents=True, exist_ok=True)
        accepted_before = tree_digest(output) if output.exists() else None
        request_digest = sha256_bytes(canonical_json_bytes(fixture))
        staging = output.with_name(f".{output.name}.staging-{failure_case}-{os.getpid()}")
        if staging.exists():
            raise FileExistsError(staging)
        try:
            try:
                spec = validate_fixture(fixture)
            except Exception as exc:
                raise Cp3Error(Cp3FailureCode.INVALID_REQUEST, str(exc)) from exc
            if expected_fixture_digest is not None and expected_fixture_digest != request_digest:
                raise Cp3Error(Cp3FailureCode.STALE_INPUT, f"fixture digest mismatch expected={expected_fixture_digest} observed={request_digest}")
            runtime = spec["runtime"]
            exact = {
                "build123d_version": EXPECTED_BUILD123D,
                "ocp_version": EXPECTED_OCP,
                "build123d_source_commit": EXPECTED_SOURCE_COMMIT,
                "ocp_wheel_sha256": EXPECTED_OCP_SHA,
            }
            if runtime != exact:
                raise Cp3Error(Cp3FailureCode.RUNTIME_MISMATCH, f"runtime identity mismatch: {runtime}")
            staging.mkdir(parents=True)
            source_dir = staging / "source"
            try:
                self.source.execute(spec, source_dir)
            except Exception as exc:
                raise Cp3Error(_map_upstream_failure(str(exc)), str(exc)) from exc
            if fault == "REQUIRED_PART_FAILED":
                raise Cp3Error(Cp3FailureCode.REQUIRED_PART_FAILED, "injected required part failure after source generation")
            mesh = json.loads((source_dir / "neutral-mesh.json").read_text(encoding="utf-8"))
            parts = json.loads((source_dir / "semantic-parts.json").read_text(encoding="utf-8"))
            stored = json.loads((source_dir / "stored-copies.json").read_text(encoding="utf-8"))
            source_result = json.loads((source_dir / "result.json").read_text(encoding="utf-8"))
            if source_result.get("status") != "SUCCEEDED" or source_result.get("part_count") != 5:
                raise Cp3Error(Cp3FailureCode.REQUIRED_PART_FAILED, "source wall span is incomplete")
            coverage = build_source_coverage(mesh, parts, stored, linear_tolerance_m=float(spec["tolerances"]["linear_m"]))
            if fault == "COVERAGE_GAP":
                coverage = copy.deepcopy(coverage)
                coverage["surfaces"][0]["triangle_indices"] = coverage["surfaces"][0]["triangle_indices"][1:]
            elif fault == "COVERAGE_OVERLAP":
                coverage = copy.deepcopy(coverage)
                duplicate = coverage["surfaces"][0]["triangle_indices"][0]
                coverage["surfaces"][1]["triangle_indices"].append(duplicate)
            validate_source_coverage(coverage, mesh)
            coverage_path = staging / "source-coverage-map.json"
            coverage_path.write_bytes(pretty_json_bytes(coverage))
            if _file_bytes(staging) > int(max_published_bytes):
                raise Cp3Error(Cp3FailureCode.GEOMETRY_BUDGET_EXCEEDED, f"published byte budget exceeded: {_file_bytes(staging)} > {max_published_bytes}")
            receipt = {
                "schema": RECEIPT_SCHEMA, "status": "PASS", "producer_revision": PRODUCER_REVISION,
                "fixture_digest": request_digest, "source_result_digest": sha256_path(source_dir / "result.json"),
                "source_provider_receipt_digest": sha256_path(source_dir / "cad-provider-receipt.json"),
                "source_coverage_map_digest": sha256_path(coverage_path),
                "runtime": exact, "source_triangle_count": len(mesh["triangles"]),
                "covered_triangle_count": coverage["summary"]["covered_triangle_count"],
                "surface_count": coverage["summary"]["surface_count"],
                "coverage_policy": "EXCLUSIVE_COMPLETE_ANALYTIC_PLANE_CLASSIFICATION",
                "provider_face_order_trusted": False,
                "publish_policy": "STAGE_VALIDATE_ATOMIC_RENAME",
                "partial_output_allowed": False,
                "capabilities": ["fortification.source_coverage@1", "fortification.negative_gates@1", "artifact.atomic_publish@1"],
            }
            receipt_path = staging / "cp3-qualification-receipt.json"
            receipt_path.write_bytes(pretty_json_bytes(receipt))
            result = {
                "schema": RESULT_SCHEMA, "span_id": spec["span_id"], "status": "SUCCEEDED", "failure": None,
                "source_triangle_count": len(mesh["triangles"]), "covered_triangle_count": coverage["summary"]["covered_triangle_count"],
                "source_surface_count": coverage["summary"]["surface_count"], "gap_count": 0, "overlap_count": 0,
                "source_output_ref": _ref(source_dir / "result.json", staging),
                "source_coverage_map_ref": _ref(coverage_path, staging),
                "qualification_receipt_ref": _ref(receipt_path, staging),
                "partial_output_published": False,
            }
            (staging / "result.json").write_bytes(pretty_json_bytes(result))
            required = {"source", "source-coverage-map.json", "cp3-qualification-receipt.json", "result.json"}
            if not required <= {p.name for p in staging.iterdir()}:
                raise Cp3Error(Cp3FailureCode.PUBLISH_ABORTED, "staging output is incomplete")
            if output.exists():
                raise Cp3Error(Cp3FailureCode.PUBLISH_ABORTED, "accepted output already exists")
            staging.rename(output)
            return result
        except Exception as exc:
            error = exc if isinstance(exc, Cp3Error) else Cp3Error(Cp3FailureCode.PUBLISH_ABORTED, str(exc))
            failed = failures / failure_case
            if failed.exists():
                raise FileExistsError(failed) from exc
            if staging.exists():
                staging.rename(failed)
            else:
                failed.mkdir(parents=True)
            accepted_after = tree_digest(output) if output.exists() else None
            failure = {
                "schema": FAILURE_SCHEMA, "status": "REJECTED", "failure": {"code": error.code, "message": error.message},
                "request_digest": request_digest, "partial_output_published": False,
                "accepted_target_existed_before": accepted_before is not None,
                "accepted_target_unchanged": accepted_before == accepted_after,
                "accepted_target_digest_before": accepted_before,
                "accepted_target_digest_after": accepted_after,
                "preserved_failed_work": True,
            }
            (failed / "failure-result.json").write_bytes(pretty_json_bytes(failure))
            return failure
