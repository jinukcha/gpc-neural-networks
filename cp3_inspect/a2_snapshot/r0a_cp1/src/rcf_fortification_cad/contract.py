from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import math
from typing import Any, Mapping, Sequence

REQUEST_SCHEMA = "royal-capital.fortification.cad-provider-request/1"
RESULT_SCHEMA = "royal-capital.fortification.cad-provider-result/1"
RECEIPT_SCHEMA = "royal-capital.fortification.cad-provider-receipt/1"
MESH_SCHEMA = "royal-capital.fortification.neutral-indexed-mesh/1"
ADAPTER_ID = "royal-capital/fortification/build123d-adapter"
ADAPTER_REVISION = "r0a-cp1.1"
PROJECT_FRAME_ID = "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"
PROVIDER_FRAME_ID = "RH_X_RIGHT_Y_FORWARD_Z_UP"
SUPPORTED_UNITS = "METER"

EXPECTED_BUILD123D_VERSION = "0.13.1.dev12+ge22d34dae"
EXPECTED_OCP_VERSION = "8.0.1.0.0"
EXPECTED_BUILD123D_SOURCE_COMMIT = "e22d34dae17111e5b9fdb361317e055d0daae466"
EXPECTED_OCP_WHEEL_SHA256 = "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f"


class CadStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class FailureCode(StrEnum):
    INVALID_ADAPTER_REQUEST = "INVALID_ADAPTER_REQUEST"
    UNIT_CONTRACT_MISMATCH = "UNIT_CONTRACT_MISMATCH"
    FRAME_CONTRACT_MISMATCH = "FRAME_CONTRACT_MISMATCH"
    TOLERANCE_OUT_OF_DOMAIN = "TOLERANCE_OUT_OF_DOMAIN"
    GEOMETRY_BUDGET_EXCEEDED = "GEOMETRY_BUDGET_EXCEEDED"
    CAD_RUNTIME_VERSION_MISMATCH = "CAD_RUNTIME_VERSION_MISMATCH"
    PROVIDER_EXECUTION_FAILED = "PROVIDER_EXECUTION_FAILED"
    STORED_COPY_EXPORT_FAILED = "STORED_COPY_EXPORT_FAILED"


class ContractError(ValueError):
    def __init__(self, code: FailureCode | str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = message


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


@dataclass(frozen=True)
class PlaneFrame:
    origin_m: Vec3
    x_axis: Vec3
    y_axis: Vec3

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_m": self.origin_m.to_list(),
            "x_axis": self.x_axis.to_list(),
            "y_axis": self.y_axis.to_list(),
        }


@dataclass(frozen=True)
class ToleranceContract:
    linear_m: float
    angular_rad: float
    tessellation_linear_m: float
    tessellation_angular_rad: float


@dataclass(frozen=True)
class GeometryBudget:
    max_profile_points: int
    max_vertices: int
    max_triangles: int
    max_artifact_bytes: int


@dataclass(frozen=True)
class ProfileExtrusion:
    kind: str
    plane: PlaneFrame
    outer_loop_m: tuple[tuple[float, float], ...]
    distance_m: float


@dataclass(frozen=True)
class OutputRequest:
    indexed_mesh: bool
    stored_copies: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeExpectation:
    build123d_version: str
    ocp_version: str
    build123d_source_commit: str
    ocp_wheel_sha256: str


@dataclass(frozen=True)
class CadProviderRequest:
    schema: str
    request_id: str
    units: str
    frame: str
    runtime: RuntimeExpectation
    tolerances: ToleranceContract
    budget: GeometryBudget
    operation: ProfileExtrusion
    outputs: OutputRequest

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "units": self.units,
            "frame": self.frame,
            "runtime": asdict(self.runtime),
            "tolerances": asdict(self.tolerances),
            "budget": asdict(self.budget),
            "operation": {
                "kind": self.operation.kind,
                "plane": self.operation.plane.to_dict(),
                "outer_loop_m": [list(p) for p in self.operation.outer_loop_m],
                "distance_m": self.operation.distance_m,
            },
            "outputs": {
                "indexed_mesh": self.outputs.indexed_mesh,
                "stored_copies": list(self.outputs.stored_copies),
            },
        }


@dataclass(frozen=True)
class CadProviderResult:
    schema: str
    request_id: str
    status: CadStatus
    failure: dict[str, str] | None
    unit_frame_contract: dict[str, Any]
    shape: dict[str, Any] | None
    mesh: dict[str, Any] | None
    stored_copies: tuple[dict[str, Any], ...]
    provider_receipt_ref: dict[str, str] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "status": self.status.value,
            "failure": self.failure,
            "unit_frame_contract": self.unit_frame_contract,
            "shape": self.shape,
            "mesh": self.mesh,
            "stored_copies": list(self.stored_copies),
            "provider_receipt_ref": self.provider_receipt_ref,
        }


def _strict_keys(value: Mapping[str, Any], required: set[str], optional: set[str] = set()) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise ContractError(
            FailureCode.INVALID_ADAPTER_REQUEST,
            f"key mismatch: missing={sorted(missing)} extra={sorted(extra)}",
        )


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, f"{field} must be finite")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, f"{field} must be an integer")
    return value


def _vec3(value: Any, field: str) -> Vec3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, f"{field} must have exactly 3 numbers")
    return Vec3(*(_number(v, f"{field}[{i}]") for i, v in enumerate(value)))


def _norm(v: Vec3) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def _polygon_area(loop: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        loop[i][0] * loop[(i + 1) % len(loop)][1] - loop[(i + 1) % len(loop)][0] * loop[i][1]
        for i in range(len(loop))
    )


def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d, eps: float) -> bool:
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    return ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))


def _validate_simple_loop(loop: tuple[tuple[float, float], ...], linear_tolerance: float) -> None:
    if len(loop) < 3:
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "outer_loop_m requires at least 3 points")
    if abs(_polygon_area(loop)) <= linear_tolerance * linear_tolerance:
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "outer_loop_m has zero or near-zero area")
    n = len(loop)
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        if math.dist(a, b) <= linear_tolerance:
            raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "outer_loop_m contains a zero-length edge")
        for j in range(i + 1, n):
            if j in {i, (i + 1) % n} or i in {j, (j + 1) % n}:
                continue
            if i == 0 and j == n - 1:
                continue
            c, d = loop[j], loop[(j + 1) % n]
            if _segments_intersect(a, b, c, d, linear_tolerance):
                raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "outer_loop_m self-intersects")


def request_from_dict(value: Mapping[str, Any]) -> CadProviderRequest:
    if not isinstance(value, Mapping):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "request must be an object")
    _strict_keys(value, {"schema", "request_id", "units", "frame", "runtime", "tolerances", "budget", "operation", "outputs"})
    if value["schema"] != REQUEST_SCHEMA:
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, f"unsupported schema {value['schema']!r}")
    request_id = value["request_id"]
    if not isinstance(request_id, str) or not request_id.strip():
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "request_id must be a non-empty string")
    units = value["units"]
    if units != SUPPORTED_UNITS:
        raise ContractError(FailureCode.UNIT_CONTRACT_MISMATCH, f"units must be {SUPPORTED_UNITS}")
    frame = value["frame"]
    if frame != PROJECT_FRAME_ID:
        raise ContractError(FailureCode.FRAME_CONTRACT_MISMATCH, f"frame must be {PROJECT_FRAME_ID}")

    runtime = value["runtime"]
    if not isinstance(runtime, Mapping):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "runtime must be an object")
    _strict_keys(runtime, {"build123d_version", "ocp_version", "build123d_source_commit", "ocp_wheel_sha256"})
    runtime_obj = RuntimeExpectation(**{k: str(runtime[k]) for k in runtime})

    tolerances = value["tolerances"]
    if not isinstance(tolerances, Mapping):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "tolerances must be an object")
    _strict_keys(tolerances, {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad"})
    tolerance_obj = ToleranceContract(**{k: _number(tolerances[k], f"tolerances.{k}") for k in tolerances})
    domains = {
        "linear_m": (1e-9, 1e-3),
        "angular_rad": (1e-9, 1e-3),
        "tessellation_linear_m": (1e-3, 1e-1),
        "tessellation_angular_rad": (1e-2, 3e-1),
    }
    for field, (lower, upper) in domains.items():
        current = getattr(tolerance_obj, field)
        if not (lower <= current <= upper):
            raise ContractError(FailureCode.TOLERANCE_OUT_OF_DOMAIN, f"{field}={current} outside [{lower}, {upper}]")

    budget = value["budget"]
    if not isinstance(budget, Mapping):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "budget must be an object")
    _strict_keys(budget, {"max_profile_points", "max_vertices", "max_triangles", "max_artifact_bytes"})
    budget_obj = GeometryBudget(**{k: _integer(budget[k], f"budget.{k}") for k in budget})
    if not (3 <= budget_obj.max_profile_points <= 4096 and 1 <= budget_obj.max_vertices <= 5_000_000 and 1 <= budget_obj.max_triangles <= 5_000_000 and 1024 <= budget_obj.max_artifact_bytes <= 1_000_000_000):
        raise ContractError(FailureCode.GEOMETRY_BUDGET_EXCEEDED, "budget is outside adapter hard limits")

    operation = value["operation"]
    if not isinstance(operation, Mapping):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "operation must be an object")
    _strict_keys(operation, {"kind", "plane", "outer_loop_m", "distance_m"})
    if operation["kind"] != "PROFILE_EXTRUSION":
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "CP1 adapter supports PROFILE_EXTRUSION only")
    plane = operation["plane"]
    if not isinstance(plane, Mapping):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "operation.plane must be an object")
    _strict_keys(plane, {"origin_m", "x_axis", "y_axis"})
    plane_obj = PlaneFrame(_vec3(plane["origin_m"], "operation.plane.origin_m"), _vec3(plane["x_axis"], "operation.plane.x_axis"), _vec3(plane["y_axis"], "operation.plane.y_axis"))
    axis_error = max(abs(_norm(plane_obj.x_axis) - 1.0), abs(_norm(plane_obj.y_axis) - 1.0), abs(_dot(plane_obj.x_axis, plane_obj.y_axis)))
    if axis_error > tolerance_obj.angular_rad:
        raise ContractError(FailureCode.FRAME_CONTRACT_MISMATCH, f"plane axes are not orthonormal within angular tolerance; error={axis_error}")
    if _norm(_cross(plane_obj.x_axis, plane_obj.y_axis)) <= tolerance_obj.angular_rad:
        raise ContractError(FailureCode.FRAME_CONTRACT_MISMATCH, "plane axes do not define a right-handed normal")

    raw_loop = operation["outer_loop_m"]
    if not isinstance(raw_loop, Sequence) or isinstance(raw_loop, (str, bytes)):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "outer_loop_m must be an array")
    loop: list[tuple[float, float]] = []
    for i, point in enumerate(raw_loop):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, f"outer_loop_m[{i}] must have 2 numbers")
        loop.append((_number(point[0], f"outer_loop_m[{i}][0]"), _number(point[1], f"outer_loop_m[{i}][1]")))
    if len(loop) > budget_obj.max_profile_points:
        raise ContractError(FailureCode.GEOMETRY_BUDGET_EXCEEDED, "profile point count exceeds request budget")
    loop_tuple = tuple(loop)
    _validate_simple_loop(loop_tuple, tolerance_obj.linear_m)
    distance = _number(operation["distance_m"], "operation.distance_m")
    if abs(distance) <= tolerance_obj.linear_m:
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "distance_m must exceed linear tolerance")
    operation_obj = ProfileExtrusion("PROFILE_EXTRUSION", plane_obj, loop_tuple, distance)

    outputs = value["outputs"]
    if not isinstance(outputs, Mapping):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "outputs must be an object")
    _strict_keys(outputs, {"indexed_mesh", "stored_copies"})
    if outputs["indexed_mesh"] is not True:
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "indexed_mesh must be true")
    copies = outputs["stored_copies"]
    if not isinstance(copies, Sequence) or isinstance(copies, (str, bytes)):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "stored_copies must be an array")
    copy_tuple = tuple(str(x) for x in copies)
    if len(set(copy_tuple)) != len(copy_tuple) or any(x not in {"STEP", "BREP"} for x in copy_tuple):
        raise ContractError(FailureCode.INVALID_ADAPTER_REQUEST, "stored_copies must be unique STEP/BREP values")
    output_obj = OutputRequest(True, copy_tuple)

    return CadProviderRequest(REQUEST_SCHEMA, request_id, units, frame, runtime_obj, tolerance_obj, budget_obj, operation_obj, output_obj)
