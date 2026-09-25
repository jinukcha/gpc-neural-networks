from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping

from .canonical import (
    add_scaled,
    canonical_json_bytes,
    canonicalize_mesh,
    cross,
    normalize,
    normalized_step_sha256,
    pretty_json_bytes,
    project_to_provider,
    sha256_bytes,
    sha256_ref,
    unit_frame_contract,
)
from .contract import (
    ADAPTER_ID,
    ADAPTER_REVISION,
    EXPECTED_BUILD123D_SOURCE_COMMIT,
    EXPECTED_BUILD123D_VERSION,
    EXPECTED_OCP_VERSION,
    EXPECTED_OCP_WHEEL_SHA256,
    RECEIPT_SCHEMA,
    RESULT_SCHEMA,
    CadProviderRequest,
    CadProviderResult,
    CadStatus,
    ContractError,
    FailureCode,
    request_from_dict,
)


class Build123dProviderAdapter:
    """Thin project-owned adapter. Public inputs and outputs contain no upstream types."""

    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.cp0_root = Path(cp0_root).resolve()

    def execute(self, request: CadProviderRequest | Mapping[str, Any], output_dir: str | os.PathLike[str]) -> CadProviderResult:
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"output_dir must be fresh: {output}")
        request_id = str(request.get("request_id", "UNKNOWN")) if isinstance(request, Mapping) else request.request_id
        try:
            parsed = request_from_dict(request) if isinstance(request, Mapping) else request
            self._validate_runtime_expectation(parsed)
        except ContractError as exc:
            return self._publish_failure(output, request_id, CadStatus.REJECTED, exc.code, exc.message)

        temporary = output.with_name(output.name + f".partial-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"temporary output already exists: {temporary}")
        temporary.mkdir(parents=True)
        try:
            result = self._execute_success(parsed, temporary)
            total_bytes = sum(path.stat().st_size for path in temporary.rglob("*") if path.is_file())
            if total_bytes > parsed.budget.max_artifact_bytes:
                raise ContractError(FailureCode.GEOMETRY_BUDGET_EXCEEDED, f"artifact bytes {total_bytes} exceed budget {parsed.budget.max_artifact_bytes}")
            temporary.rename(output)
            return result
        except ContractError as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            return self._publish_failure(output, parsed.request_id, CadStatus.REJECTED, exc.code, exc.message)
        except Exception as exc:  # provider errors are normalized, not leaked
            shutil.rmtree(temporary, ignore_errors=True)
            return self._publish_failure(output, parsed.request_id, CadStatus.FAILED, FailureCode.PROVIDER_EXECUTION_FAILED, f"{type(exc).__name__}: {exc}")

    def _validate_runtime_expectation(self, request: CadProviderRequest) -> None:
        expected = request.runtime
        actual_build123d = importlib.metadata.version("build123d")
        actual_ocp = importlib.metadata.version("cadquery-ocp-novtk")
        manifest_path = self.cp0_root / "runtime/WHEELHOUSE.json"
        lock_path = self.cp0_root / "runtime/requirements-lock.txt"
        if not manifest_path.is_file() or not lock_path.is_file():
            raise ContractError(FailureCode.CAD_RUNTIME_VERSION_MISMATCH, "CP0 runtime evidence is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed = {
            "build123d_version": actual_build123d,
            "ocp_version": actual_ocp,
            "build123d_source_commit": manifest.get("build123d_source_commit"),
            "ocp_wheel_sha256": manifest.get("ocp_expected_sha256"),
        }
        requested = asdict(expected)
        authoritative = {
            "build123d_version": EXPECTED_BUILD123D_VERSION,
            "ocp_version": EXPECTED_OCP_VERSION,
            "build123d_source_commit": EXPECTED_BUILD123D_SOURCE_COMMIT,
            "ocp_wheel_sha256": EXPECTED_OCP_WHEEL_SHA256,
        }
        if requested != authoritative or observed != authoritative:
            raise ContractError(FailureCode.CAD_RUNTIME_VERSION_MISMATCH, f"runtime mismatch requested={requested} observed={observed}")

    def _execute_success(self, request: CadProviderRequest, output: Path) -> CadProviderResult:
        import build123d as b3d

        op = request.operation
        plane = op.plane
        project_points = [add_scaled(plane.origin_m, plane.x_axis, plane.y_axis, point) for point in op.outer_loop_m]
        provider_points = [project_to_provider(point) for point in project_points]
        project_normal = normalize(cross(plane.x_axis, plane.y_axis))
        sign = 1.0 if op.distance_m > 0 else -1.0
        mapped_direction = project_to_provider((project_normal.x * sign, project_normal.y * sign, project_normal.z * sign))
        direction_length = sum(component * component for component in mapped_direction) ** 0.5
        provider_direction = tuple(component / direction_length for component in mapped_direction)
        wire = b3d.Wire.make_polygon(provider_points, close=True)
        face = b3d.Face(wire)
        shape = b3d.extrude(face, amount=abs(op.distance_m) * 1000.0, dir=provider_direction)
        if shape.volume <= 0:
            raise ContractError(FailureCode.PROVIDER_EXECUTION_FAILED, "provider produced a non-positive volume")

        provider_vertices, provider_triangles = shape.tessellate(request.tolerances.tessellation_linear_m * 1000.0, request.tolerances.tessellation_angular_rad)
        mesh = canonicalize_mesh(provider_vertices, provider_triangles)
        vertex_count = len(mesh["vertices_m"])
        triangle_count = len(mesh["triangles"])
        if vertex_count > request.budget.max_vertices or triangle_count > request.budget.max_triangles:
            raise ContractError(FailureCode.GEOMETRY_BUDGET_EXCEEDED, f"mesh {vertex_count} vertices/{triangle_count} triangles exceeds budget")

        mesh_bytes = pretty_json_bytes(mesh)
        mesh_path = output / "neutral-mesh.json"
        mesh_path.write_bytes(mesh_bytes)
        mesh_ref = {
            "path": mesh_path.name,
            "sha256": "sha256:" + sha256_bytes(mesh_bytes),
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
        }

        stored_copies = []
        if "STEP" in request.outputs.stored_copies:
            buffer = BytesIO()
            if not b3d.export_step(shape, buffer, unit=b3d.Unit.MM, timestamp="1970-01-01T00:00:00"):
                raise ContractError(FailureCode.STORED_COPY_EXPORT_FAILED, "STEP export returned false")
            data = buffer.getvalue()
            path = output / "shape.step"
            path.write_bytes(data)
            reopened = b3d.import_step(path)
            if abs(reopened.volume / 1_000_000_000.0 - shape.volume / 1_000_000_000.0) > max(request.tolerances.linear_m ** 3, 1e-9):
                raise ContractError(FailureCode.STORED_COPY_EXPORT_FAILED, "STEP reopen volume mismatch")
            stored_copies.append({
                "format": "STEP",
                "path": path.name,
                "bytes": len(data),
                "raw_sha256": "sha256:" + sha256_bytes(data),
                "canonical_sha256": "sha256:" + normalized_step_sha256(data),
                "reopen_volume_m3": round(float(reopened.volume) / 1_000_000_000.0, 9),
                "stored_numeric_unit": "MILLIMETER",
                "project_unit": "METER",
            })
        if "BREP" in request.outputs.stored_copies:
            buffer = BytesIO()
            if not b3d.export_brep(shape, buffer):
                raise ContractError(FailureCode.STORED_COPY_EXPORT_FAILED, "BREP export returned false")
            data = buffer.getvalue()
            path = output / "shape.brep"
            path.write_bytes(data)
            reopened = b3d.import_brep(path)
            if abs(reopened.volume / 1_000_000_000.0 - shape.volume / 1_000_000_000.0) > max(request.tolerances.linear_m ** 3, 1e-9):
                raise ContractError(FailureCode.STORED_COPY_EXPORT_FAILED, "BREP reopen volume mismatch")
            digest = "sha256:" + sha256_bytes(data)
            stored_copies.append({
                "format": "BREP",
                "path": path.name,
                "bytes": len(data),
                "raw_sha256": digest,
                "canonical_sha256": digest,
                "reopen_volume_m3": round(float(reopened.volume) / 1_000_000_000.0, 9),
                "stored_numeric_unit": "MILLIMETER",
                "project_unit": "METER",
            })

        shape_summary = {
            "solid_count": 1,
            "volume_m3": round(float(shape.volume) / 1_000_000_000.0, 9),
            "bounds_m": mesh["bounds_m"],
        }
        result_core = {
            "schema": RESULT_SCHEMA,
            "request_id": request.request_id,
            "status": CadStatus.SUCCEEDED.value,
            "failure": None,
            "unit_frame_contract": unit_frame_contract(),
            "shape": shape_summary,
            "mesh": mesh_ref,
            "stored_copies": stored_copies,
            "provider_receipt_ref": None,
        }
        request_bytes = canonical_json_bytes(request.to_dict())
        result_core_bytes = canonical_json_bytes(result_core)
        receipt = self._provider_receipt(request, request_bytes, result_core_bytes, mesh_ref, stored_copies)
        receipt_bytes = pretty_json_bytes(receipt)
        receipt_path = output / "cad-provider-receipt.json"
        receipt_path.write_bytes(receipt_bytes)
        receipt_ref = {"path": receipt_path.name, "sha256": sha256_ref(receipt_bytes)}

        final = dict(result_core)
        final["provider_receipt_ref"] = receipt_ref
        result_bytes = pretty_json_bytes(final)
        (output / "result.json").write_bytes(result_bytes)
        return CadProviderResult(
            RESULT_SCHEMA,
            request.request_id,
            CadStatus.SUCCEEDED,
            None,
            final["unit_frame_contract"],
            shape_summary,
            mesh_ref,
            tuple(stored_copies),
            receipt_ref,
        )

    def _provider_receipt(self, request: CadProviderRequest, request_bytes: bytes, result_core_bytes: bytes, mesh_ref: dict[str, Any], stored_copies: list[dict[str, Any]]) -> dict[str, Any]:
        import build123d as _build123d  # local import; never appears in public types

        source_dir = Path(__file__).resolve().parent
        source_files = [source_dir / name for name in ("__init__.py", "contract.py", "canonical.py", "provider.py", "cli.py")]
        source_digest_input = b"".join(path.name.encode("utf-8") + b"\0" + path.read_bytes() for path in source_files)
        manifest_path = self.cp0_root / "runtime/WHEELHOUSE.json"
        lock_path = self.cp0_root / "runtime/requirements-lock.txt"
        return {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS",
            "provider_id": "build123d/OCP",
            "adapter_id": ADAPTER_ID,
            "adapter_revision": ADAPTER_REVISION,
            "adapter_source_digest": sha256_ref(source_digest_input),
            "source_ref": {
                "project": "build123d",
                "commit": EXPECTED_BUILD123D_SOURCE_COMMIT,
                "license": "Apache-2.0",
            },
            "runtime": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_cache_tag": sys.implementation.cache_tag,
                "platform": "linux-x86_64",
                "build123d_version": importlib.metadata.version("build123d"),
                "ocp_version": importlib.metadata.version("cadquery-ocp-novtk"),
                "wheelhouse_manifest_sha256": sha256_ref(manifest_path.read_bytes()),
                "requirements_lock_sha256": sha256_ref(lock_path.read_bytes()),
                "ocp_wheel_sha256": EXPECTED_OCP_WHEEL_SHA256,
                "global_install": False,
            },
            "contract": {
                **unit_frame_contract(),
                "tolerances": asdict(request.tolerances),
                "operation_kind": request.operation.kind,
                "public_upstream_types_exposed": False,
            },
            "request_digest": sha256_ref(request_bytes),
            "result_core_digest": sha256_ref(result_core_bytes),
            "mesh_digest": mesh_ref["sha256"],
            "stored_copy_canonical_digests": {item["format"]: item["canonical_sha256"] for item in stored_copies},
            "capabilities": [
                "cad.profile_extrusion@1",
                "cad.tessellation@1",
                "cad.step_brep_export@1",
            ],
        }

    def _publish_failure(self, output: Path, request_id: str, status: CadStatus, code: FailureCode | str, message: str) -> CadProviderResult:
        output.mkdir(parents=True, exist_ok=False)
        result = CadProviderResult(
            RESULT_SCHEMA,
            request_id,
            status,
            {"code": str(code), "message": message},
            unit_frame_contract(),
            None,
            None,
            tuple(),
            None,
        )
        (output / "result.json").write_bytes(pretty_json_bytes(result.to_dict()))
        return result
