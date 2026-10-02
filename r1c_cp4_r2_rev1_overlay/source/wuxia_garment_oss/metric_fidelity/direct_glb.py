"""Deterministic Blender-free GLB writer and internal fresh-reopen validator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np


_COMPONENT_FLOAT = 5126
_COMPONENT_UINT = 5125
_TARGET_ARRAY = 34962
_TARGET_ELEMENT = 34963


class _BufferBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def accessor(self, values: np.ndarray, component_type: int, kind: str, target: int) -> int:
        array = np.ascontiguousarray(values)
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        raw = array.tobytes(order="C")
        self.data.extend(raw)
        view_index = len(self.views)
        self.views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        count = int(array.shape[0]) if array.ndim > 1 else int(array.size)
        accessor = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": count,
            "type": kind,
        }
        if component_type == _COMPONENT_FLOAT and kind == "VEC3":
            reshaped = array.reshape((-1, 3)).astype(np.float64)
            accessor["min"] = reshaped.min(axis=0).tolist()
            accessor["max"] = reshaped.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def write_direct_glb(
    path: Path,
    components: list[dict],
    body_positions: np.ndarray,
    body_triangles: np.ndarray,
    metadata: dict,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = _BufferBuilder()
    primitives = []
    primitive_receipts = []
    for component in components:
        primitive, receipt = _primitive(builder, component, 0)
        primitives.append(primitive)
        primitive_receipts.append(receipt)
    body_component = {
        "instance_id": "reference_body",
        "positions": np.asarray(body_positions, dtype=np.float32),
        "triangles": np.asarray(body_triangles, dtype=np.uint32),
    }
    primitive, receipt = _primitive(builder, body_component, 1)
    primitives.append(primitive)
    primitive_receipts.append(receipt)
    document = _document(builder, primitives, metadata)
    _write_glb(path, document, bytes(builder.data))
    reopened = validate_direct_glb(path)
    payload = {
        "contract": "DirectGarmentGLBPackage/1",
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "primitive_count": len(primitives),
        "garment_component_count": len(components),
        "body_primitive_count": 1,
        "primitives": primitive_receipts,
        "fresh_reopen": reopened,
        "blender_used": False,
    }
    return payload


def _primitive(builder: _BufferBuilder, component: dict, material: int) -> tuple[dict, dict]:
    positions = np.asarray(component["positions"], dtype=np.float32)
    triangles = np.asarray(component["triangles"], dtype=np.uint32)
    normals = _normals(positions.astype(np.float64), triangles.astype(np.int32)).astype(np.float32)
    position_accessor = builder.accessor(positions, _COMPONENT_FLOAT, "VEC3", _TARGET_ARRAY)
    normal_accessor = builder.accessor(normals, _COMPONENT_FLOAT, "VEC3", _TARGET_ARRAY)
    index_accessor = builder.accessor(triangles.reshape(-1), _COMPONENT_UINT, "SCALAR", _TARGET_ELEMENT)
    name = component["instance_id"]
    primitive = {
        "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
        "indices": index_accessor,
        "material": material,
        "mode": 4,
        "extras": {"componentInstanceId": name},
    }
    receipt = {
        "instance_id": name,
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(triangles)),
        "material_index": material,
    }
    return primitive, receipt


def _document(builder: _BufferBuilder, primitives: list[dict], metadata: dict) -> dict:
    materials = [
        {
            "name": "R1C_NEUTRAL_GRAY_GARMENT",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.56, 0.56, 0.56, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.88,
            },
        },
        {
            "name": "R1C_REFERENCE_BODY",
            "doubleSided": False,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.32, 0.32, 0.32, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.92,
            },
        },
    ]
    return {
        "asset": {"version": "2.0", "generator": "GARMENT-CAD-PRO-R1C CP4-R2-R1-REV1"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "R1C_CP4_R2_PRODUCT", "mesh": 0}],
        "meshes": [{"name": "R1C_CP4_R2_PRODUCT_MESH", "primitives": primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(builder.data)}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
        "extras": metadata,
    }


def _write_glb(path: Path, document: dict, binary: bytes) -> None:
    json_bytes = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    header = struct.pack("<4sII", b"glTF", 2, total)
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(header + json_chunk + bin_chunk)


def validate_direct_glb(path: Path) -> dict:
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total != len(raw):
        raise ValueError("invalid GLB header")
    offset = 12
    document = None
    binary = b""
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \t\r\n\0"))
        elif kind == 0x004E4942:
            binary = payload
        offset += 8 + length
    if document is None or not binary:
        raise ValueError("missing GLB chunks")
    _validate_accessors(document, binary)
    primitives = document["meshes"][0]["primitives"]
    return {
        "pass": True,
        "asset_version": document["asset"]["version"],
        "primitive_count": len(primitives),
        "accessor_count": len(document["accessors"]),
        "binary_bytes": len(binary),
    }


def _validate_accessors(document: dict, binary: bytes) -> None:
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    sizes = {_COMPONENT_FLOAT: 4, _COMPONENT_UINT: 4}
    for accessor in document["accessors"]:
        view = document["bufferViews"][accessor["bufferView"]]
        width = widths[accessor["type"]]
        item_size = sizes[accessor["componentType"]] * width
        start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        end = start + int(accessor["count"]) * item_size
        if start < 0 or end > len(binary):
            raise ValueError("accessor exceeds binary buffer")


def _normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions)
    for triangle in triangles:
        a, b, c = map(int, triangle)
        normal = np.cross(positions[b] - positions[a], positions[c] - positions[a])
        length = np.linalg.norm(normal)
        if length > 1.0e-12:
            normal /= length
            normals[a] += normal
            normals[b] += normal
            normals[c] += normal
    lengths = np.linalg.norm(normals, axis=1)
    fallback = lengths <= 1.0e-12
    normals[~fallback] /= lengths[~fallback, None]
    normals[fallback] = np.array([0.0, 0.0, 1.0])
    return normals


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
