"""Compile accepted garment GLBs into deterministic skinned game products."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Iterable

import numpy as np


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


BONES: tuple[tuple[str, str | None], ...] = (
    ("ROOT", None),
    ("PELVIS", "ROOT"),
    ("SPINE_01", "PELVIS"),
    ("SPINE_02", "SPINE_01"),
    ("CHEST", "SPINE_02"),
    ("NECK", "CHEST"),
    ("HEAD", "NECK"),
    ("L_CLAVICLE", "CHEST"),
    ("L_UPPER_ARM", "L_CLAVICLE"),
    ("L_FOREARM", "L_UPPER_ARM"),
    ("L_HAND", "L_FOREARM"),
    ("R_CLAVICLE", "CHEST"),
    ("R_UPPER_ARM", "R_CLAVICLE"),
    ("R_FOREARM", "R_UPPER_ARM"),
    ("R_HAND", "R_FOREARM"),
    ("L_THIGH", "PELVIS"),
    ("L_CALF", "L_THIGH"),
    ("L_FOOT", "L_CALF"),
    ("L_TOE", "L_FOOT"),
    ("R_THIGH", "PELVIS"),
    ("R_CALF", "R_THIGH"),
    ("R_FOOT", "R_CALF"),
    ("R_TOE", "R_FOOT"),
)
BONE_INDEX = {name: index for index, (name, _) in enumerate(BONES)}


@dataclass(frozen=True)
class GLB:
    document: dict
    binary: bytearray


def read_glb(path: Path) -> GLB:
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError(f"invalid GLB header: {path}")
    offset = 12
    document = None
    binary = bytearray()
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + length]
        offset += length
        if kind == JSON_CHUNK:
            document = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
        elif kind == BIN_CHUNK:
            binary.extend(payload)
    if document is None:
        raise ValueError("GLB has no JSON chunk")
    return GLB(document, binary)


def write_glb(path: Path, glb: GLB) -> None:
    document = json.dumps(glb.document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    document += b" " * ((4 - len(document) % 4) % 4)
    binary = bytes(glb.binary)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(document) + 8 + len(binary)
    payload = bytearray(struct.pack("<4sII", b"glTF", 2, total))
    payload.extend(struct.pack("<II", len(document), JSON_CHUNK))
    payload.extend(document)
    payload.extend(struct.pack("<II", len(binary), BIN_CHUNK))
    payload.extend(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _accessor_array(glb: GLB, accessor_index: int) -> np.ndarray:
    accessor = glb.document["accessors"][accessor_index]
    view = glb.document["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPES[accessor["componentType"]]).newbyteorder("<")
    width = TYPE_COUNTS[accessor["type"]]
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    stride = int(view.get("byteStride", dtype.itemsize * width))
    if stride == dtype.itemsize * width:
        values = np.frombuffer(glb.binary, dtype=dtype, count=count * width, offset=start)
        return values.reshape(count, width).copy()
    result = np.empty((count, width), dtype=dtype)
    for row in range(count):
        offset = start + row * stride
        result[row] = np.frombuffer(glb.binary, dtype=dtype, count=width, offset=offset)
    return result


def _append_blob(glb: GLB, payload: bytes, target: int | None = None) -> int:
    padding = (4 - len(glb.binary) % 4) % 4
    if padding:
        glb.binary.extend(b"\x00" * padding)
    offset = len(glb.binary)
    glb.binary.extend(payload)
    view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
    if target is not None:
        view["target"] = target
    glb.document.setdefault("bufferViews", []).append(view)
    glb.document.setdefault("buffers", [{"byteLength": 0}])[0]["byteLength"] = len(glb.binary)
    return len(glb.document["bufferViews"]) - 1


def _append_accessor(
    glb: GLB,
    values: np.ndarray,
    component_type: int,
    accessor_type: str,
    target: int | None = None,
    include_bounds: bool = False,
) -> int:
    dtype = np.dtype(COMPONENT_DTYPES[component_type]).newbyteorder("<")
    array = np.asarray(values, dtype=dtype)
    view_index = _append_blob(glb, array.tobytes(order="C"), target)
    accessor = {
        "bufferView": view_index,
        "componentType": component_type,
        "count": int(array.shape[0]),
        "type": accessor_type,
    }
    if include_bounds:
        accessor["min"] = np.min(array, axis=0).astype(float).tolist()
        accessor["max"] = np.max(array, axis=0).astype(float).tolist()
    glb.document.setdefault("accessors", []).append(accessor)
    return len(glb.document["accessors"]) - 1


def _all_positions(glb: GLB) -> np.ndarray:
    arrays = []
    for mesh in glb.document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            accessor = primitive.get("attributes", {}).get("POSITION")
            if accessor is not None:
                arrays.append(_accessor_array(glb, accessor).astype(np.float64))
    if not arrays:
        raise ValueError("GLB has no positions")
    return np.concatenate(arrays, axis=0)


def _axis_frame(positions: np.ndarray) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    minimum = np.min(positions, axis=0)
    maximum = np.max(positions, axis=0)
    extent = maximum - minimum
    order = np.argsort(extent)
    depth, lateral, vertical = int(order[0]), int(order[1]), int(order[2])
    return lateral, vertical, depth, minimum, maximum


def _vec(lateral: int, vertical: int, depth: int, x: float, y: float, z: float) -> list[float]:
    result = [0.0, 0.0, 0.0]
    result[lateral] = x
    result[vertical] = y
    result[depth] = z
    return result


def _skeleton_local_translations(positions: np.ndarray) -> list[list[float]]:
    lateral, vertical, depth, minimum, maximum = _axis_frame(positions)
    height = float(maximum[vertical] - minimum[vertical])
    width = float(maximum[lateral] - minimum[lateral])
    bottom = float(minimum[vertical])
    pelvis_y = bottom + 0.49 * height
    unit = max(height, 0.5)
    local: dict[str, list[float]] = {
        "ROOT": _vec(lateral, vertical, depth, 0.0, 0.0, 0.0),
        "PELVIS": _vec(lateral, vertical, depth, 0.0, pelvis_y, 0.0),
        "SPINE_01": _vec(lateral, vertical, depth, 0.0, 0.09 * unit, 0.0),
        "SPINE_02": _vec(lateral, vertical, depth, 0.0, 0.09 * unit, 0.0),
        "CHEST": _vec(lateral, vertical, depth, 0.0, 0.10 * unit, 0.0),
        "NECK": _vec(lateral, vertical, depth, 0.0, 0.09 * unit, 0.0),
        "HEAD": _vec(lateral, vertical, depth, 0.0, 0.13 * unit, 0.0),
        "L_CLAVICLE": _vec(lateral, vertical, depth, -0.08 * width, 0.03 * unit, 0.0),
        "L_UPPER_ARM": _vec(lateral, vertical, depth, -0.20 * width, -0.015 * unit, 0.0),
        "L_FOREARM": _vec(lateral, vertical, depth, -0.26 * width, -0.03 * unit, 0.0),
        "L_HAND": _vec(lateral, vertical, depth, -0.20 * width, -0.02 * unit, 0.0),
        "R_CLAVICLE": _vec(lateral, vertical, depth, 0.08 * width, 0.03 * unit, 0.0),
        "R_UPPER_ARM": _vec(lateral, vertical, depth, 0.20 * width, -0.015 * unit, 0.0),
        "R_FOREARM": _vec(lateral, vertical, depth, 0.26 * width, -0.03 * unit, 0.0),
        "R_HAND": _vec(lateral, vertical, depth, 0.20 * width, -0.02 * unit, 0.0),
        "L_THIGH": _vec(lateral, vertical, depth, -0.10 * width, -0.18 * unit, 0.0),
        "L_CALF": _vec(lateral, vertical, depth, 0.0, -0.22 * unit, 0.0),
        "L_FOOT": _vec(lateral, vertical, depth, 0.0, -0.20 * unit, 0.03 * unit),
        "L_TOE": _vec(lateral, vertical, depth, 0.0, 0.0, 0.10 * unit),
        "R_THIGH": _vec(lateral, vertical, depth, 0.10 * width, -0.18 * unit, 0.0),
        "R_CALF": _vec(lateral, vertical, depth, 0.0, -0.22 * unit, 0.0),
        "R_FOOT": _vec(lateral, vertical, depth, 0.0, -0.20 * unit, 0.03 * unit),
        "R_TOE": _vec(lateral, vertical, depth, 0.0, 0.0, 0.10 * unit),
    }
    return [local[name] for name, _ in BONES]


def _normalize(entries: Iterable[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[int, float] = {}
    for name, value in entries:
        if value > 0.0:
            index = BONE_INDEX[name]
            merged[index] = merged.get(index, 0.0) + float(value)
    ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:4]
    total = sum(value for _, value in ranked)
    if total <= 0.0:
        ranked = [(BONE_INDEX["PELVIS"], 1.0)]
        total = 1.0
    joints = np.zeros(4, dtype=np.uint16)
    weights = np.zeros(4, dtype=np.float32)
    for slot, (index, value) in enumerate(ranked):
        joints[slot] = index
        weights[slot] = value / total
    weights[0] += np.float32(1.0 - float(np.sum(weights, dtype=np.float64)))
    return joints, weights


def _tunic_weight(position: np.ndarray, primitive_index: int, frame) -> tuple[np.ndarray, np.ndarray]:
    lateral, vertical, _, minimum, maximum = frame
    x = float(position[lateral])
    width = max(float(maximum[lateral] - minimum[lateral]), 1.0e-6)
    side = x / (0.5 * width)
    t = float((position[vertical] - minimum[vertical]) / max(maximum[vertical] - minimum[vertical], 1.0e-6))
    if primitive_index == 12:
        return _normalize((("NECK", 1.0),))
    if primitive_index == 13:
        return _normalize((("CHEST", 1.0),))
    if primitive_index in (7, 8, 9, 10, 11):
        return _normalize((("CHEST", 0.70), ("NECK", 0.25), ("SPINE_02", 0.05)))
    if primitive_index == 5:
        return _normalize((("CHEST", 0.38), ("L_CLAVICLE", 0.28), ("L_UPPER_ARM", 0.34)))
    if primitive_index == 6:
        return _normalize((("CHEST", 0.38), ("R_CLAVICLE", 0.28), ("R_UPPER_ARM", 0.34)))
    if t >= 0.78 and abs(side) > 0.45:
        prefix = "L" if side < 0.0 else "R"
        amount = min(1.0, (abs(side) - 0.45) / 0.55)
        return _normalize((("CHEST", 0.50 * (1.0 - amount)), (f"{prefix}_CLAVICLE", 0.35), (f"{prefix}_UPPER_ARM", 0.15 + 0.50 * amount)))
    if t >= 0.63:
        return _normalize((("CHEST", 0.65), ("SPINE_02", 0.25), ("NECK", 0.10 * max(0.0, (t - 0.8) / 0.2))))
    if t >= 0.46:
        blend = (t - 0.46) / 0.17
        return _normalize((("PELVIS", 0.55 * (1.0 - blend)), ("SPINE_01", 0.45), ("SPINE_02", 0.55 * blend)))
    left = max(0.0, min(1.0, 0.5 - 0.5 * side))
    right = 1.0 - left
    lower = max(0.0, (0.46 - t) / 0.46)
    return _normalize((("PELVIS", 0.58 - 0.18 * lower), ("SPINE_01", 0.18), ("L_THIGH", 0.24 * lower * left), ("R_THIGH", 0.24 * lower * right)))


def _trousers_weight(position: np.ndarray, primitive_index: int, frame) -> tuple[np.ndarray, np.ndarray]:
    lateral, vertical, _, minimum, maximum = frame
    side = float(position[lateral] / max(0.5 * (maximum[lateral] - minimum[lateral]), 1.0e-6))
    t = float((position[vertical] - minimum[vertical]) / max(maximum[vertical] - minimum[vertical], 1.0e-6))
    if primitive_index == 1:
        return _normalize((("PELVIS", 0.78), ("SPINE_01", 0.22)))
    if primitive_index == 2:
        return _normalize((("PELVIS", 0.58), ("L_THIGH", 0.21), ("R_THIGH", 0.21)))
    prefix = "L" if side < 0.0 else "R"
    if t >= 0.76:
        return _normalize((("PELVIS", 0.76), ("SPINE_01", 0.14), (f"{prefix}_THIGH", 0.10)))
    if t >= 0.47:
        blend = (t - 0.47) / 0.29
        return _normalize((("PELVIS", 0.20 + 0.32 * blend), (f"{prefix}_THIGH", 0.72 - 0.25 * blend), (f"{prefix}_CALF", 0.08)))
    if t >= 0.18:
        blend = (t - 0.18) / 0.29
        return _normalize(((f"{prefix}_THIGH", 0.18 + 0.42 * blend), (f"{prefix}_CALF", 0.76 - 0.40 * blend), (f"{prefix}_FOOT", 0.06)))
    return _normalize(((f"{prefix}_CALF", 0.35), (f"{prefix}_FOOT", 0.55), (f"{prefix}_TOE", 0.10)))


def _weight_fields(glb: GLB, garment_kind: str) -> list[tuple[np.ndarray, np.ndarray]]:
    positions = _all_positions(glb)
    frame = _axis_frame(positions)
    fields = []
    for mesh in glb.document.get("meshes", []):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            values = _accessor_array(glb, primitive["attributes"]["POSITION"]).astype(np.float64)
            joints = np.empty((len(values), 4), dtype=np.uint16)
            weights = np.empty((len(values), 4), dtype=np.float32)
            for index, position in enumerate(values):
                if garment_kind == "tunic":
                    joint_row, weight_row = _tunic_weight(position, primitive_index, frame)
                else:
                    joint_row, weight_row = _trousers_weight(position, primitive_index, frame)
                joints[index] = joint_row
                weights[index] = weight_row
            fields.append((joints, weights))
    return fields


def _global_matrices(local_translations: list[list[float]]) -> list[np.ndarray]:
    globals_: list[np.ndarray] = []
    for index, (_, parent_name) in enumerate(BONES):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(local_translations[index], dtype=np.float64)
        if parent_name is not None:
            matrix = globals_[BONE_INDEX[parent_name]] @ matrix
        globals_.append(matrix)
    return globals_


def _append_skeleton(glb: GLB, local_translations: list[list[float]]) -> tuple[int, list[int]]:
    nodes = glb.document.setdefault("nodes", [])
    joint_nodes: list[int] = []
    for index, (name, _) in enumerate(BONES):
        node = {"name": name, "translation": [float(v) for v in local_translations[index]], "extras": {"semanticBoneId": name}}
        nodes.append(node)
        joint_nodes.append(len(nodes) - 1)
    children: dict[int, list[int]] = {}
    for index, (_, parent_name) in enumerate(BONES):
        if parent_name is not None:
            children.setdefault(joint_nodes[BONE_INDEX[parent_name]], []).append(joint_nodes[index])
    for node_index, child_indices in children.items():
        nodes[node_index]["children"] = child_indices
    return joint_nodes[0], joint_nodes


def compile_skinned_glb(source: Path, target: Path, garment_kind: str) -> dict:
    glb = read_glb(source)
    positions = _all_positions(glb)
    local = _skeleton_local_translations(positions)
    root_node, joint_nodes = _append_skeleton(glb, local)
    inverse = np.stack([np.linalg.inv(matrix).T.astype(np.float32) for matrix in _global_matrices(local)], axis=0)
    inverse_accessor = _append_accessor(glb, inverse.reshape(len(BONES), 16), 5126, "MAT4")
    fields = _weight_fields(glb, garment_kind)
    field_index = 0
    total_vertices = 0
    max_error = 0.0
    zero_vertices = 0
    for mesh in glb.document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            joints, weights = fields[field_index]
            field_index += 1
            total_vertices += len(joints)
            max_error = max(max_error, float(np.max(np.abs(np.sum(weights, axis=1) - 1.0))))
            zero_vertices += int(np.count_nonzero(np.sum(weights, axis=1) <= 0.0))
            primitive.setdefault("attributes", {})["JOINTS_0"] = _append_accessor(glb, joints, 5123, "VEC4", 34962)
            primitive["attributes"]["WEIGHTS_0"] = _append_accessor(glb, weights, 5126, "VEC4", 34962)
    skin = {
        "name": f"{garment_kind.upper()}_CANONICAL_SKIN",
        "inverseBindMatrices": inverse_accessor,
        "joints": joint_nodes,
        "skeleton": root_node,
        "extras": {"canonicalSkeletonContract": "CanonicalSkeletonPackage/1"},
    }
    glb.document["skins"] = [skin]
    mesh_nodes = [node for node in glb.document.get("nodes", []) if "mesh" in node]
    if not mesh_nodes:
        raise ValueError("GLB has no mesh node")
    for node in mesh_nodes:
        node["skin"] = 0
        node.setdefault("extras", {})["riggedGarmentProduct"] = "RiggedGarmentProduct/1"
    scene_index = int(glb.document.get("scene", 0))
    scenes = glb.document.setdefault("scenes", [{"nodes": []}])
    scenes[scene_index].setdefault("nodes", [])
    if root_node not in scenes[scene_index]["nodes"]:
        scenes[scene_index]["nodes"].append(root_node)
    glb.document.setdefault("asset", {})["generator"] = "GARMENT-CAD-PRO-R1B CP3"
    glb.document.setdefault("extensionsUsed", [])
    glb.document.setdefault("extras", {})["r1bCp3"] = {
        "garmentKind": garment_kind,
        "canonicalBoneCount": len(BONES),
        "correctiveContract": "CorrectiveDeformationSet/1",
        "secondaryMotion": "NOT_EXECUTED",
        "lod": "NOT_EXECUTED",
    }
    write_glb(target, glb)
    return {
        "source": source.name,
        "target": target.name,
        "garment_kind": garment_kind,
        "bone_count": len(BONES),
        "mesh_node_count": len(mesh_nodes),
        "primitive_count": field_index,
        "vertex_count": total_vertices,
        "maximum_weight_sum_error": max_error,
        "zero_weight_vertex_count": zero_vertices,
        "skin_count": 1,
        "inverse_bind_matrix_count": len(BONES),
    }
