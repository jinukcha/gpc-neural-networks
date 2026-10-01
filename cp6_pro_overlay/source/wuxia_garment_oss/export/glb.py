"""Minimal deterministic glTF 2.0 binary writer for neutral-gray garments."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..pattern_cad.document.model import canonical_sha256


_COMPONENT_FLOAT = 5126
_COMPONENT_UINT32 = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    remainder = len(data) % 4
    return data if remainder == 0 else data + fill * (4 - remainder)


def _to_gltf_vectors(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    return np.column_stack((source[:, 0], source[:, 2], -source[:, 1])).astype(np.float32)


def _vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float64)
    a = positions[triangles[:, 0]]
    b = positions[triangles[:, 1]]
    c = positions[triangles[:, 2]]
    face = np.cross(b - a, c - a)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face)
    length = np.linalg.norm(normals, axis=1)
    fallback = length <= 1.0e-12
    normals[fallback] = (0.0, 0.0, 1.0)
    length[fallback] = 1.0
    return normals / length[:, None]


def _planar_uv(positions: np.ndarray) -> np.ndarray:
    """Generate stable UV0 from the two largest object-space extents."""
    source = np.asarray(positions, dtype=np.float64)
    if not len(source):
        raise ValueError("primitive positions must not be empty")
    extent = np.ptp(source, axis=0)
    axes = np.argsort(-extent, kind="stable")[:2]
    selected = source[:, axes]
    minimum = np.min(selected, axis=0)
    span = np.ptp(selected, axis=0)
    span = np.where(span > 1.0e-12, span, 1.0)
    uv = (selected - minimum) / span
    return uv.astype(np.float32)


def oriented_triangle_sha256(positions: np.ndarray, triangles: np.ndarray) -> str:
    transformed = _to_gltf_vectors(positions)
    indices = np.asarray(triangles, dtype=np.uint32)
    digest = hashlib.sha256()
    digest.update(transformed.tobytes(order="C"))
    digest.update(indices.tobytes(order="C"))
    return digest.hexdigest()


class _BufferBuilder:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def _append(self, raw: bytes, target: int | None) -> int:
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        self.binary.extend(raw)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1

    def vectors(self, values: np.ndarray, target: int = _ARRAY_BUFFER) -> int:
        array = np.asarray(values, dtype=np.float32)
        view = self._append(array.tobytes(order="C"), target)
        minimum = np.min(array, axis=0).astype(float).tolist()
        maximum = np.max(array, axis=0).astype(float).tolist()
        accessor = {
            "bufferView": view,
            "componentType": _COMPONENT_FLOAT,
            "count": int(len(array)),
            "type": "VEC3",
            "min": minimum,
            "max": maximum,
        }
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def vectors2(self, values: np.ndarray, target: int = _ARRAY_BUFFER) -> int:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("VEC2 accessor values must be Nx2")
        view = self._append(array.tobytes(order="C"), target)
        accessor = {
            "bufferView": view,
            "componentType": _COMPONENT_FLOAT,
            "count": int(len(array)),
            "type": "VEC2",
            "min": np.min(array, axis=0).astype(float).tolist(),
            "max": np.max(array, axis=0).astype(float).tolist(),
        }
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def indices(self, values: np.ndarray) -> int:
        array = np.asarray(values, dtype=np.uint32).reshape(-1)
        view = self._append(array.tobytes(order="C"), _ELEMENT_ARRAY_BUFFER)
        accessor = {
            "bufferView": view,
            "componentType": _COMPONENT_UINT32,
            "count": int(len(array)),
            "type": "SCALAR",
            "min": [int(np.min(array))],
            "max": [int(np.max(array))],
        }
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _primitive_payload(
    builder: _BufferBuilder,
    primitive: Mapping[str, object],
    morph_names: Sequence[str],
) -> tuple[dict, dict]:
    positions = np.asarray(primitive["positions"], dtype=np.float64)
    triangles = np.asarray(primitive["triangles"], dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("primitive positions must be Nx3")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("primitive triangles must be Mx3")
    if triangles.size and (int(triangles.min()) < 0 or int(triangles.max()) >= len(positions)):
        raise ValueError("primitive triangle index is out of bounds")
    normals = _vertex_normals(positions, triangles)
    attributes = {
        "POSITION": builder.vectors(_to_gltf_vectors(positions)),
        "NORMAL": builder.vectors(_to_gltf_vectors(normals)),
        "TEXCOORD_0": builder.vectors2(_planar_uv(positions)),
    }
    targets = []
    morphs = dict(primitive.get("morph_targets", {}))
    for name in morph_names:
        delta = np.asarray(morphs.get(name, np.zeros_like(positions)), dtype=np.float64)
        if delta.shape != positions.shape:
            raise ValueError(f"morph target shape mismatch: {name}")
        targets.append({"POSITION": builder.vectors(_to_gltf_vectors(delta))})
    payload = {
        "attributes": attributes,
        "indices": builder.indices(triangles),
        "material": int(primitive.get("material", 0)),
        "mode": 4,
    }
    if targets:
        payload["targets"] = targets
    receipt = {
        "name": str(primitive.get("name", "primitive")),
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(triangles)),
        "uv0": True,
        "oriented_triangle_sha256": oriented_triangle_sha256(positions, triangles),
    }
    return payload, receipt


def write_glb(
    path: Path,
    product_id: str,
    primitives: Sequence[Mapping[str, object]],
    morph_names: Sequence[str] = (),
    extras: Mapping[str, object] | None = None,
) -> dict:
    if not primitives:
        raise ValueError("at least one GLB primitive is required")
    builder = _BufferBuilder()
    primitive_payloads = []
    primitive_receipts = []
    for primitive in primitives:
        payload, receipt = _primitive_payload(builder, primitive, morph_names)
        primitive_payloads.append(payload)
        primitive_receipts.append(receipt)
    mesh = {
        "name": product_id,
        "primitives": primitive_payloads,
        "weights": [0.0 for _ in morph_names],
        "extras": {"targetNames": list(morph_names)},
    }
    document = {
        "asset": {"version": "2.0", "generator": "WuxiaGarmentOSS CP6 deterministic GLB"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": product_id, "mesh": 0}],
        "meshes": [mesh],
        "materials": [{
            "name": "NeutralGray",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.62, 0.62, 0.62, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.82,
            },
            "doubleSided": True,
        }],
        "buffers": [{"byteLength": len(builder.binary)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
        "extras": dict(extras or {}),
    }
    json_chunk = _pad4(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"), b" ")
    binary_chunk = _pad4(bytes(builder.binary), b"\x00")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    body = struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
    body += struct.pack("<I4s", len(binary_chunk), b"BIN\x00") + binary_chunk
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body)
    receipt = {
        "contract": "GameGarmentProduct/1",
        "product_id": product_id,
        "path": path.name,
        "glb_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_length": path.stat().st_size,
        "primitive_count": len(primitives),
        "morph_target_names": list(morph_names),
        "primitive_receipts": primitive_receipts,
        "coordinate_conversion": "RH_Z_UP_TO_GLTF_RH_Y_UP_X_Z_NEGY",
        "neutral_gray_material": True,
        "uv0_present": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def read_glb_header(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("GLB is too short")
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError("invalid GLB header")
    json_length, json_kind = struct.unpack_from("<I4s", data, 12)
    if json_kind != b"JSON":
        raise ValueError("GLB JSON chunk is missing")
    json_start = 20
    document = json.loads(data[json_start:json_start + json_length].decode("utf-8"))
    offset = json_start + json_length
    binary_length, binary_kind = struct.unpack_from("<I4s", data, offset)
    if binary_kind != b"BIN\x00":
        raise ValueError("GLB binary chunk is missing")
    binary = data[offset + 8:offset + 8 + binary_length]
    return document, binary


def verify_glb(path: Path, expected_morphs: Sequence[str]) -> dict:
    document, binary = read_glb_header(path)
    mesh = document["meshes"][0]
    names = mesh.get("extras", {}).get("targetNames", [])
    primitive_count = len(mesh["primitives"])
    target_counts = [len(item.get("targets", [])) for item in mesh["primitives"]]
    uv0_pass = all("TEXCOORD_0" in item.get("attributes", {}) for item in mesh["primitives"])
    result = {
        "contract": "GLBFreshReopenReceipt/1",
        "path": path.name,
        "glb_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "asset_version": document["asset"]["version"],
        "primitive_count": primitive_count,
        "accessor_count": len(document.get("accessors", [])),
        "buffer_view_count": len(document.get("bufferViews", [])),
        "binary_byte_length": len(binary),
        "morph_target_names": list(names),
        "morph_target_counts": target_counts,
        "morph_identity_pass": list(names) == list(expected_morphs),
        "target_count_pass": all(item == len(expected_morphs) for item in target_counts),
        "uv0_pass": uv0_pass,
    }
    result["fresh_reopen_pass"] = (
        result["asset_version"] == "2.0"
        and result["primitive_count"] > 0
        and result["binary_byte_length"] > 0
        and result["morph_identity_pass"]
        and result["target_count_pass"]
        and result["uv0_pass"]
    )
    result["receipt_sha256"] = canonical_sha256(result)
    return result
