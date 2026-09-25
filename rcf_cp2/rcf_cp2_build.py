#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import textwrap
import zipfile
from typing import Any, Iterable

BASELINE_SHA = "6b705952be72036e9d6c6eaa614ddd33abe963ed9d1411f9d839b1f11e7f1cd2"
RCK0_SHA = "7f637daf8e6fd1e055cca31e4d287072cc96ae0bb4028121c5471d34905a38ac"
D0_PATCH_SHA = "15369b830e049937e0d980100197c617285687316bea96339b1ed5dc4cb7c47b"
A1S_SHA = "56d573c5c82bd421a423f82d251e42553298eccb9bae35a7c95278c209ce8b9d"
A1_FULL_SHA = "9f5886497a23b783379bc0938cb3e64134fddbf60130d48498d31053ac2d8b3c"
CP0_ARTIFACT_RUN = 36093161115
PART_ORDER = ["foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet"]
SOCKET_ORDER = [
    "span_start", "span_end", "wall_walk_start", "wall_walk_end",
    "foundation_start", "foundation_end", "tower_start", "tower_end",
    "utility_inside_01", "utility_inside_02",
]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_text(path: Path, value: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    path.chmod(mode)


def write_json(path: Path, value: Any, mode: int = 0o644) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", mode)


def copy_overlay(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        target = destination / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_symlink():
            raise RuntimeError(f"symlink not allowed: {path}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def extract_safe(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = set()
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            parts = Path(name).parts
            if name.startswith("/") or ".." in parts:
                raise RuntimeError(f"unsafe ZIP path: {name}")
            if name in names:
                raise RuntimeError(f"duplicate ZIP entry: {name}")
            names.add(name)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise RuntimeError(f"symlink ZIP entry: {name}")
        zf.extractall(destination)
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {archive}: {bad}")


def file_map(root: Path) -> dict[str, tuple[str, int, int]]:
    result: dict[str, tuple[str, int, int]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        result[rel] = (sha_file(path), path.stat().st_size, stat.S_IMODE(path.stat().st_mode))
    return result


def update_registry(root: Path) -> None:
    registry = root / "provenance/FILES.sha256"
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != registry):
        rows.append(f"{sha_file(path)}  {path.relative_to(root).as_posix()}")
    write_text(registry, "\n".join(rows) + "\n")


def verify_registry(root: Path, expected_count: int | None = None) -> dict[str, Any]:
    registry = root / "provenance/FILES.sha256"
    listed: dict[str, str] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    actual = {p.relative_to(root).as_posix(): sha_file(p) for p in root.rglob("*") if p.is_file() and p != registry}
    missing = sorted(set(listed) - set(actual))
    extra = sorted(set(actual) - set(listed))
    mismatched = sorted(rel for rel in set(listed) & set(actual) if listed[rel] != actual[rel])
    if missing or extra or mismatched:
        raise RuntimeError(f"FILES.sha256 mismatch missing={missing[:5]} extra={extra[:5]} bad={mismatched[:5]}")
    total = len(actual) + 1
    if expected_count is not None and total != expected_count:
        raise RuntimeError(f"tree file count {total} != expected {expected_count}")
    return {"listed": len(listed), "tree_files": total, "missing": missing, "extra": extra, "mismatched": mismatched}


def deterministic_zip(source_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(p for p in source_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(rel, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IMODE(path.stat().st_mode) & 0xFFFF) << 16
            info.create_system = 3
            zf.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(output) as zf:
        if zf.testzip() is not None:
            raise RuntimeError(f"reopen CRC failed: {output}")
        if len(zf.namelist()) != len(set(zf.namelist())):
            raise RuntimeError(f"duplicate entries in {output}")


def compare_trees(a: Path, b: Path) -> tuple[bool, dict[str, Any]]:
    ma, mb = file_map(a), file_map(b)
    missing = sorted(set(ma) - set(mb))
    extra = sorted(set(mb) - set(ma))
    changed = sorted(k for k in set(ma) & set(mb) if ma[k] != mb[k])
    return not (missing or extra or changed), {"missing": missing, "extra": extra, "changed": changed, "a_files": len(ma), "b_files": len(mb)}


def build_patch_payload(baseline_parent: Path, final_parent: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    base = file_map(baseline_parent)
    final = file_map(final_parent)
    added = sorted(set(final) - set(base))
    modified = sorted(k for k in set(final) & set(base) if final[k] != base[k])
    deleted = sorted(set(base) - set(final))
    w = destination / "W"
    for rel in added + modified:
        src = final_parent / rel
        dst = w / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    write_text(destination / "D.txt", "\n".join(deleted) + ("\n" if deleted else ""))
    metadata = {
        "schema": "royal-capital.fortification.cumulative-patch/1",
        "baseline": {"file": "RCF_D0_full.zip", "sha256": BASELINE_SHA},
        "authority": "W_PAYLOAD_EXACT_BYTES",
        "added": len(added), "modified": len(modified), "deleted": len(deleted),
        "added_paths": added, "modified_paths": modified, "deleted_paths": deleted,
    }
    write_json(destination / "PATCH_METADATA.json", metadata)
    return metadata


def apply_patch_payload(baseline_parent: Path, patch: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(baseline_parent, destination, copy_function=shutil.copy2)
    if (patch / "W").exists():
        copy_overlay(patch / "W", destination)
    deletes = (patch / "D.txt").read_text(encoding="utf-8").splitlines()
    for rel in deletes:
        target = destination / rel
        if target.is_file() or target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)


def build_source_payload(baseline_parent: Path, final_parent: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    base, final = file_map(baseline_parent), file_map(final_parent)
    changed = sorted(k for k in set(final) if k not in base or final[k] != base[k])
    included, excluded = [], []
    for rel in changed:
        if "/runtime/wheelhouse/" in f"/{rel}":
            excluded.append(rel)
            continue
        src = final_parent / rel
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        included.append(rel)
    write_text(destination / "SOURCE_PACKAGE.txt", textwrap.dedent(f"""\
        ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP2 cumulative changed source package
        baseline=RCF_D0_full.zip
        baseline_sha256={BASELINE_SHA}
        included_paths={len(included)}
        excluded_exact_wheels={len(excluded)}
        wheel_payload_is_preserved_in_full_and_patch=true
        """))
    return {"included": included, "excluded_wheels": excluded}


def package_set(label: str, baseline_parent: Path, tree_parent: Path, package_root: Path) -> dict[str, Any]:
    package_root.mkdir(parents=True, exist_ok=True)
    patch_dir = package_root / f"{label}_patch_payload"
    source_dir = package_root / f"{label}_source_payload"
    patch_meta = build_patch_payload(baseline_parent, tree_parent, patch_dir)
    source_meta = build_source_payload(baseline_parent, tree_parent, source_dir)
    full_zip = package_root / f"{label}.zip"
    patch_zip = package_root / f"{label}P.zip"
    source_zip = package_root / f"{label}S.zip"
    deterministic_zip(tree_parent, full_zip)
    deterministic_zip(patch_dir, patch_zip)
    deterministic_zip(source_dir, source_zip)
    reconstructed = package_root / f"{label}_reconstructed"
    apply_patch_payload(baseline_parent, patch_dir, reconstructed)
    equal, detail = compare_trees(tree_parent, reconstructed)
    if not equal:
        raise RuntimeError(f"baseline+patch mismatch for {label}: {detail}")
    for z in (full_zip, patch_zip, source_zip):
        with zipfile.ZipFile(z) as zf:
            if zf.testzip() is not None:
                raise RuntimeError(f"CRC failed: {z}")
    return {
        "label": label,
        "full": {"file": full_zip.name, "bytes": full_zip.stat().st_size, "sha256": sha_file(full_zip), "entries": len(file_map(tree_parent))},
        "patch": {"file": patch_zip.name, "bytes": patch_zip.stat().st_size, "sha256": sha_file(patch_zip), "entries": len(file_map(patch_dir)), **{k: patch_meta[k] for k in ("added", "modified", "deleted")}},
        "source": {"file": source_zip.name, "bytes": source_zip.stat().st_size, "sha256": sha_file(source_zip), "entries": len(file_map(source_dir)), "included": len(source_meta["included"]), "excluded_wheels": len(source_meta["excluded_wheels"])},
        "baseline_plus_patch_equals_full": True,
    }


def reconstruct(inputs: Path, wheel_artifact: Path, work: Path) -> tuple[Path, Path, dict[str, Any]]:
    for name, expected in (("RC_K0.zip", RCK0_SHA), ("RCF_D0_patch.zip", D0_PATCH_SHA), ("A1S.zip", A1S_SHA)):
        actual = sha_file(inputs / name)
        if actual != expected:
            raise RuntimeError(f"{name} SHA mismatch {actual}")
    base_extract = work / "base_extract"
    patch_extract = work / "d0_patch_extract"
    a1s_extract = work / "a1s_extract"
    extract_safe(inputs / "RC_K0.zip", base_extract)
    extract_safe(inputs / "RCF_D0_patch.zip", patch_extract)
    extract_safe(inputs / "A1S.zip", a1s_extract)
    baseline_parent = work / "baseline_parent"
    tree_parent = work / "tree_parent"
    baseline_parent.mkdir(parents=True)
    tree_parent.mkdir(parents=True)
    copy_overlay(base_extract, baseline_parent)
    copy_overlay(base_extract, tree_parent)
    w_candidates = [p for p in patch_extract.rglob("W") if p.is_dir()]
    if len(w_candidates) != 1:
        raise RuntimeError(f"expected one W directory, found {w_candidates}")
    copy_overlay(w_candidates[0], baseline_parent)
    copy_overlay(w_candidates[0], tree_parent)
    baseline_root = baseline_parent / "RC_K0"
    verify_registry(baseline_root, 165)
    a1_root = a1s_extract / "RC_K0"
    if not a1_root.is_dir():
        raise RuntimeError("A1S RC_K0 root missing")
    copy_overlay(a1_root, tree_parent / "RC_K0")
    wh_source = wheel_artifact / "wheelhouse"
    wh_target = tree_parent / "RC_K0/child_designs/fortification/r0a_cp0/runtime/wheelhouse"
    if not wh_source.is_dir() or len(list(wh_source.glob("*.whl"))) != 58:
        raise RuntimeError("exact 58-wheel artifact missing")
    wh_target.mkdir(parents=True, exist_ok=True)
    copy_overlay(wh_source, wh_target)
    a1_registry = verify_registry(tree_parent / "RC_K0", 427)
    return baseline_parent, tree_parent, {"baseline": verify_registry(baseline_root, 165), "a1": a1_registry}


def cp2_files() -> dict[str, str]:
    model_py = r'''from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

SPAN_SCHEMA = "royal-capital.fortification.straight-wall-span/1"
PLAN_SCHEMA = "royal-capital.fortification.wall-span-plan/1"
PARTS_SCHEMA = "royal-capital.fortification.semantic-parts/1"
SOCKETS_SCHEMA = "royal-capital.fortification.socket-plan/1"
TESSELLATION_SCHEMA = "royal-capital.fortification.fixed-tessellation/1"
COMBINED_MESH_SCHEMA = "royal-capital.fortification.wall-span-indexed-mesh/1"
RECEIPT_SCHEMA = "royal-capital.fortification.wall-span-provider-receipt/1"
RESULT_SCHEMA = "royal-capital.fortification.wall-span-result/1"
PART_ORDER = ("foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet")
SOCKET_ORDER = (
    "span_start", "span_end", "wall_walk_start", "wall_walk_end",
    "foundation_start", "foundation_end", "tower_start", "tower_end",
    "utility_inside_01", "utility_inside_02",
)

@dataclass(frozen=True)
class PartSpec:
    part_id: str
    semantic_role: str
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    material_slot: str

    @property
    def volume_factor(self) -> float:
        return (self.y_max - self.y_min) * (self.z_max - self.z_min)

PART_SPECS = (
    PartSpec("foundation", "FOUNDATION", -2.0, 0.0, -4.0, 4.0, "stone_foundation"),
    PartSpec("wall_body", "WALL_BODY", 0.0, 12.0, -3.0, 3.0, "stone_wall"),
    PartSpec("wall_walk", "WALL_WALK", 12.0, 12.6, -3.2, 3.2, "stone_walk"),
    PartSpec("inner_parapet", "INNER_PARAPET", 12.6, 14.4, 2.4, 3.2, "stone_parapet"),
    PartSpec("outer_parapet", "OUTER_PARAPET", 12.6, 14.4, -3.2, -2.4, "stone_parapet"),
)


def validate_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "span_id", "length_m", "frame", "inside_side", "outside_side", "tolerances", "budget", "runtime"}
    if set(value) != required:
        raise ValueError(f"fixture keys mismatch missing={sorted(required-set(value))} extra={sorted(set(value)-required)}")
    if value["schema"] != SPAN_SCHEMA:
        raise ValueError("unsupported wall-span schema")
    if not isinstance(value["span_id"], str) or not value["span_id"].strip():
        raise ValueError("span_id required")
    length = float(value["length_m"])
    if not math.isfinite(length) or length <= 0 or length > 200:
        raise ValueError("length_m outside (0,200]")
    if value["frame"] != "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD":
        raise ValueError("project frame mismatch")
    if value["inside_side"] != "POSITIVE_Z" or value["outside_side"] != "NEGATIVE_Z":
        raise ValueError("pilot inside/outside convention mismatch")
    t = value["tolerances"]
    if set(t) != {"linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad", "mesh_round_digits"}:
        raise ValueError("tolerance keys mismatch")
    if float(t["tessellation_linear_m"]) != 0.05 or float(t["tessellation_angular_rad"]) != 0.1 or int(t["mesh_round_digits"]) != 9:
        raise ValueError("CP2 fixed tessellation contract mismatch")
    return dict(value)


def provider_request(fixture: Mapping[str, Any], part: PartSpec) -> dict[str, Any]:
    length = float(fixture["length_m"])
    loop = [[part.y_min, part.z_min], [part.y_max, part.z_min], [part.y_max, part.z_max], [part.y_min, part.z_max]]
    return {
        "schema": "royal-capital.fortification.cad-provider-request/1",
        "request_id": f"cad-request/{fixture['span_id']}/{part.part_id}@1",
        "units": "METER",
        "frame": fixture["frame"],
        "runtime": dict(fixture["runtime"]),
        "tolerances": {k: fixture["tolerances"][k] for k in ("linear_m", "angular_rad", "tessellation_linear_m", "tessellation_angular_rad")},
        "budget": dict(fixture["budget"]),
        "operation": {
            "kind": "PROFILE_EXTRUSION",
            "plane": {"origin_m": [0.0, 0.0, 0.0], "x_axis": [0.0, 1.0, 0.0], "y_axis": [0.0, 0.0, 1.0]},
            "outer_loop_m": loop,
            "distance_m": length,
        },
        "outputs": {"indexed_mesh": True, "stored_copies": ["STEP", "BREP"]},
    }


def expected_bounds(length: float, part: PartSpec) -> dict[str, list[float]]:
    return {"min": [0.0, part.y_min, part.z_min], "max": [length, part.y_max, part.z_max]}


def socket_plan(length: float) -> list[dict[str, Any]]:
    frame = {"x_axis": [1.0, 0.0, 0.0], "y_axis": [0.0, 1.0, 0.0], "z_axis": [0.0, 0.0, 1.0]}
    rows = [
        ("span_start", "SPAN_JOIN", [0.0, 6.0, 0.0]), ("span_end", "SPAN_JOIN", [length, 6.0, 0.0]),
        ("wall_walk_start", "WALL_WALK_CONTINUATION", [0.0, 12.3, 0.0]), ("wall_walk_end", "WALL_WALK_CONTINUATION", [length, 12.3, 0.0]),
        ("foundation_start", "FOUNDATION_INTERFACE", [0.0, -1.0, 0.0]), ("foundation_end", "FOUNDATION_INTERFACE", [length, -1.0, 0.0]),
        ("tower_start", "TOWER_JOIN", [0.0, 6.0, 0.0]), ("tower_end", "TOWER_JOIN", [length, 6.0, 0.0]),
        ("utility_inside_01", "UTILITY_INSIDE", [length / 3.0, 6.0, 2.75]), ("utility_inside_02", "UTILITY_INSIDE", [2.0 * length / 3.0, 6.0, 2.75]),
    ]
    return [{"socket_id": sid, "role": role, "position_m": [round(float(v), 9) for v in pos], "frame": frame, "required": role != "UTILITY_INSIDE"} for sid, role, pos in rows]
'''

    producer_py = r'''from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from rcf_fortification_cad import Build123dProviderAdapter, CadStatus
from rcf_fortification_cad.canonical import canonical_json_bytes, pretty_json_bytes, sha256_ref, unit_frame_contract

from .model import (
    COMBINED_MESH_SCHEMA, PART_ORDER, PART_SPECS, PARTS_SCHEMA, PLAN_SCHEMA,
    RECEIPT_SCHEMA, RESULT_SCHEMA, SOCKETS_SCHEMA, TESSELLATION_SCHEMA,
    expected_bounds, provider_request, socket_plan, validate_fixture,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path, base: Path) -> dict[str, Any]:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": _digest(path)}


def _bounds(vertices: list[list[float]]) -> dict[str, list[float]]:
    return {"min": [min(v[i] for v in vertices) for i in range(3)], "max": [max(v[i] for v in vertices) for i in range(3)]}


class StraightWallSpanProducer:
    def __init__(self, cp0_root: str | os.PathLike[str]):
        self.cp0_root = Path(cp0_root).resolve()
        self.adapter = Build123dProviderAdapter(self.cp0_root)

    def execute(self, fixture: Mapping[str, Any], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
        spec = validate_fixture(fixture)
        output = Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"output must be fresh: {output}")
        temporary = output.with_name(output.name + f".partial-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"temporary exists: {temporary}")
        temporary.mkdir(parents=True)
        try:
            result = self._produce(spec, temporary)
            temporary.rename(output)
            return result
        except Exception:
            # Preserve failed unique work; never overwrite it or publish a partial success.
            failed = output.with_name(output.name + f".failed-{os.getpid()}")
            if temporary.exists():
                temporary.rename(failed)
            raise

    def _produce(self, spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
        length = float(spec["length_m"])
        combined_vertices: list[list[float]] = []
        combined_triangles: list[list[int]] = []
        part_ranges = []
        semantic_parts = []
        stored = []
        part_receipts = []
        total_volume = 0.0
        for part in PART_SPECS:
            part_dir = output / "parts" / part.part_id
            request = provider_request(spec, part)
            result = self.adapter.execute(request, part_dir)
            if result.status is not CadStatus.SUCCEEDED:
                raise RuntimeError(f"part {part.part_id} provider failure: {result.to_dict()}")
            mesh = json.loads((part_dir / "neutral-mesh.json").read_text(encoding="utf-8"))
            provider_result = json.loads((part_dir / "result.json").read_text(encoding="utf-8"))
            receipt = json.loads((part_dir / "cad-provider-receipt.json").read_text(encoding="utf-8"))
            expected_volume = round(length * part.volume_factor, 9)
            expected_box = expected_bounds(length, part)
            if provider_result["shape"]["volume_m3"] != expected_volume:
                raise RuntimeError(f"{part.part_id} volume mismatch")
            if provider_result["shape"]["bounds_m"] != expected_box:
                raise RuntimeError(f"{part.part_id} bounds mismatch {provider_result['shape']['bounds_m']} != {expected_box}")
            v0, t0 = len(combined_vertices), len(combined_triangles)
            combined_vertices.extend(mesh["vertices_m"])
            combined_triangles.extend([[a + v0, b + v0, c + v0] for a, b, c in mesh["triangles"]])
            part_ranges.append({
                "part_id": part.part_id,
                "vertex_offset": v0, "vertex_count": len(mesh["vertices_m"]),
                "triangle_offset": t0, "triangle_count": len(mesh["triangles"]),
            })
            semantic_parts.append({
                "part_id": part.part_id, "semantic_role": part.semantic_role,
                "material_slot": part.material_slot, "bounds_m": expected_box,
                "volume_m3": expected_volume,
                "provider_result_ref": _ref(part_dir / "result.json", output),
                "provider_receipt_ref": _ref(part_dir / "cad-provider-receipt.json", output),
                "surface_coverage": "DEFERRED_TO_R0A_CP3",
            })
            for item in provider_result["stored_copies"]:
                copy = dict(item)
                copy["part_id"] = part.part_id
                copy["path"] = (Path("parts") / part.part_id / item["path"]).as_posix()
                stored.append(copy)
            part_receipts.append({"part_id": part.part_id, "receipt": _ref(part_dir / "cad-provider-receipt.json", output), "adapter_revision": receipt["adapter_revision"]})
            total_volume += expected_volume

        mesh = {
            "schema": COMBINED_MESH_SCHEMA, "units": "METER",
            "frame": spec["frame"], "part_order": list(PART_ORDER),
            "vertices_m": combined_vertices, "triangles": combined_triangles,
            "bounds_m": _bounds(combined_vertices), "part_ranges": part_ranges,
        }
        plan = {
            "schema": PLAN_SCHEMA, "span_id": spec["span_id"], "span_family": "STRAIGHT_SPAN",
            "length_m": length, "axis": {"start_m": [0.0, 0.0, 0.0], "end_m": [length, 0.0, 0.0]},
            "inside_side": spec["inside_side"], "outside_side": spec["outside_side"],
            "part_order": list(PART_ORDER), "assembly_mode": "BOUNDED_NON_UNIONED_PARTS",
            "source_coverage_stage": "R0A_CP3",
        }
        parts_doc = {"schema": PARTS_SCHEMA, "span_id": spec["span_id"], "parts": semantic_parts}
        sockets_doc = {"schema": SOCKETS_SCHEMA, "span_id": spec["span_id"], "sockets": socket_plan(length)}
        tess_doc = {
            "schema": TESSELLATION_SCHEMA,
            "linear_deflection_m": float(spec["tolerances"]["tessellation_linear_m"]),
            "angular_deflection_rad": float(spec["tolerances"]["tessellation_angular_rad"]),
            "mesh_round_digits": int(spec["tolerances"]["mesh_round_digits"]),
            "part_order": list(PART_ORDER), "vertex_policy": "PER_PART_LEXICOGRAPHIC_UNIQUE",
            "triangle_policy": "CYCLIC_MIN_PRESERVE_WINDING_THEN_SORT",
        }
        stored_doc = {"schema": "royal-capital.fortification.wall-span-stored-copies/1", "span_id": spec["span_id"], "copies": stored}
        documents = {
            "wall-span-plan.json": plan, "semantic-parts.json": parts_doc,
            "sockets.json": sockets_doc, "fixed-tessellation.json": tess_doc,
            "neutral-mesh.json": mesh, "stored-copies.json": stored_doc,
        }
        refs = {}
        for name, value in documents.items():
            path = output / name
            path.write_bytes(pretty_json_bytes(value))
            refs[name] = _ref(path, output)

        source_dir = Path(__file__).resolve().parent
        source_names = ("__init__.py", "model.py", "producer.py")
        source_digest_input = b"".join(name.encode() + b"\0" + (source_dir / name).read_bytes() for name in source_names)
        receipt = {
            "schema": RECEIPT_SCHEMA, "status": "PASS",
            "producer_id": "royal-capital/fortification/straight-wall-span",
            "producer_revision": "r0a-cp2.1",
            "producer_source_digest": sha256_ref(source_digest_input),
            "fixture_digest": sha256_ref(canonical_json_bytes(spec)),
            "provider_adapter": {"id": "royal-capital/fortification/build123d-adapter", "revision": "r0a-cp1.1"},
            "runtime": dict(spec["runtime"]), "unit_frame_contract": unit_frame_contract(),
            "fixed_tessellation": tess_doc,
            "part_receipts": part_receipts,
            "output_digests": {name: ref["sha256"] for name, ref in refs.items()},
            "capabilities": ["fortification.straight_span@1", "fortification.semantic_parts@1", "fortification.sockets@1", "cad.fixed_tessellation@1"],
            "surface_source_coverage": "NOT_AUTHORED_CP3",
        }
        receipt_path = output / "cad-provider-receipt.json"
        receipt_path.write_bytes(pretty_json_bytes(receipt))
        receipt_ref = _ref(receipt_path, output)
        result = {
            "schema": RESULT_SCHEMA, "span_id": spec["span_id"], "status": "SUCCEEDED",
            "failure": None, "solid_count": len(PART_ORDER),
            "volume_m3": round(total_volume, 9), "bounds_m": mesh["bounds_m"],
            "vertex_count": len(combined_vertices), "triangle_count": len(combined_triangles),
            "part_count": len(semantic_parts), "socket_count": len(sockets_doc["sockets"]),
            "artifacts": {**refs, "cad-provider-receipt.json": receipt_ref},
        }
        (output / "result.json").write_bytes(pretty_json_bytes(result))
        return result
'''

    init_py = r'''"""Straight fortification wall-span pilot built on the project-owned CP1 CAD adapter."""
from .model import PART_ORDER, SOCKET_ORDER, SPAN_SCHEMA, validate_fixture
from .producer import StraightWallSpanProducer
__all__ = ["PART_ORDER", "SOCKET_ORDER", "SPAN_SCHEMA", "StraightWallSpanProducer", "validate_fixture"]
'''

    run_py = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--cp0-root", required=True)
p.add_argument("--cp1-root", required=True)
p.add_argument("--cp2-root", required=True)
p.add_argument("--fixture", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
sys.path.insert(0, str(Path(a.cp1_root).resolve() / "src"))
sys.path.insert(0, str(Path(a.cp2_root).resolve() / "src"))
from rcf_fortification_wall_span import StraightWallSpanProducer
fixture = json.loads(Path(a.fixture).read_text(encoding="utf-8"))
result = StraightWallSpanProducer(a.cp0_root).execute(fixture, a.output)
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "SUCCEEDED" else 1)
'''

    validate_py = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

PARTS = ["foundation", "wall_body", "wall_walk", "inner_parapet", "outer_parapet"]
SOCKETS = ["span_start", "span_end", "wall_walk_start", "wall_walk_end", "foundation_start", "foundation_end", "tower_start", "tower_end", "utility_inside_01", "utility_inside_02"]
EXPECTED = {
 "foundation": ({"min":[0.0,-2.0,-4.0],"max":[24.0,0.0,4.0]},384.0),
 "wall_body": ({"min":[0.0,0.0,-3.0],"max":[24.0,12.0,3.0]},1728.0),
 "wall_walk": ({"min":[0.0,12.0,-3.2],"max":[24.0,12.6,3.2]},92.16),
 "inner_parapet": ({"min":[0.0,12.6,2.4],"max":[24.0,14.4,3.2]},34.56),
 "outer_parapet": ({"min":[0.0,12.6,-3.2],"max":[24.0,14.4,-2.4]},34.56),
}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def files(root): return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}

p=argparse.ArgumentParser(); p.add_argument('--a',required=True); p.add_argument('--b',required=True); p.add_argument('--report',required=True); a=p.parse_args()
ra,rb=Path(a.a),Path(a.b); checks=[]
def ck(name, ok, detail=None): checks.append({"name":name,"pass":bool(ok),"detail":detail});
required={"wall-span-plan.json","semantic-parts.json","sockets.json","fixed-tessellation.json","neutral-mesh.json","stored-copies.json","cad-provider-receipt.json","result.json"}
ck('root_required_A', required <= {p.name for p in ra.iterdir()}, sorted(required-{p.name for p in ra.iterdir()}))
ck('root_required_B', required <= {p.name for p in rb.iterdir()}, sorted(required-{p.name for p in rb.iterdir()}))
result=json.loads((ra/'result.json').read_text()); mesh=json.loads((ra/'neutral-mesh.json').read_text()); parts=json.loads((ra/'semantic-parts.json').read_text()); sockets=json.loads((ra/'sockets.json').read_text()); tess=json.loads((ra/'fixed-tessellation.json').read_text()); receipt=json.loads((ra/'cad-provider-receipt.json').read_text()); stored=json.loads((ra/'stored-copies.json').read_text())
ck('status',result['status']=='SUCCEEDED',result['status'])
ck('solid_count',result['solid_count']==5,result['solid_count'])
ck('volume',result['volume_m3']==2273.28,result['volume_m3'])
ck('bounds',result['bounds_m']=={'min':[0.0,-2.0,-4.0],'max':[24.0,14.4,4.0]},result['bounds_m'])
ck('mesh_counts',result['vertex_count']==40 and result['triangle_count']==60,[result['vertex_count'],result['triangle_count']])
ck('part_order',mesh['part_order']==PARTS,mesh['part_order'])
ck('part_ranges',len(mesh['part_ranges'])==5 and sum(x['vertex_count'] for x in mesh['part_ranges'])==40 and sum(x['triangle_count'] for x in mesh['part_ranges'])==60,mesh['part_ranges'])
ck('indices_valid',all(0<=i<len(mesh['vertices_m']) for tri in mesh['triangles'] for i in tri))
ck('finite_vertices',all(isinstance(x,(int,float)) and abs(x)<1e9 for v in mesh['vertices_m'] for x in v))
by_part={x['part_id']:x for x in parts['parts']}
ck('semantic_parts',list(by_part)==PARTS,list(by_part))
for part,(bounds,volume) in EXPECTED.items():
    ck(f'{part}_bounds',by_part[part]['bounds_m']==bounds,by_part[part]['bounds_m']); ck(f'{part}_volume',by_part[part]['volume_m3']==volume,by_part[part]['volume_m3']); ck(f'{part}_coverage_deferred',by_part[part]['surface_coverage']=='DEFERRED_TO_R0A_CP3')
ids=[x['socket_id'] for x in sockets['sockets']]; ck('socket_order',ids==SOCKETS,ids); ck('socket_unique',len(ids)==len(set(ids))==10)
ck('tess_linear',tess['linear_deflection_m']==0.05,tess); ck('tess_angular',tess['angular_deflection_rad']==0.1,tess); ck('mesh_round',tess['mesh_round_digits']==9,tess)
ck('stored_copy_count',len(stored['copies'])==10,len(stored['copies']))
ck('part_receipts',len(receipt['part_receipts'])==5,len(receipt['part_receipts']))
ck('cp3_not_claimed',receipt['surface_source_coverage']=='NOT_AUTHORED_CP3')
fa,fb=files(ra),files(rb); ck('file_set_equal',set(fa)==set(fb),{'missing':sorted(set(fa)-set(fb)),'extra':sorted(set(fb)-set(fa))}); ck('clean_replay_byte_identical',fa==fb,[k for k in fa if fb.get(k)!=fa[k]])
report={"schema":"royal-capital.fortification.cp2-validation/1","status":"PASS" if all(x['pass'] for x in checks) else "FAIL","checks":checks,"summary":{"passed":sum(x['pass'] for x in checks),"failed":sum(not x['pass'] for x in checks),"files_compared":len(fa)},"canonical_output_sha256":hashlib.sha256((ra/'result.json').read_bytes()).hexdigest()}
Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True)); raise SystemExit(0 if report['status']=='PASS' else 1)
'''

    test_py = r'''from __future__ import annotations
import json, os, unittest
from pathlib import Path

class WallSpanTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=Path(os.environ['RCF_CP2_ACCEPTED_OUTPUT']); cls.result=json.loads((cls.root/'result.json').read_text()); cls.mesh=json.loads((cls.root/'neutral-mesh.json').read_text()); cls.parts=json.loads((cls.root/'semantic-parts.json').read_text()); cls.sockets=json.loads((cls.root/'sockets.json').read_text())
 def test_total_envelope(self): self.assertEqual(self.result['bounds_m'],{'min':[0.0,-2.0,-4.0],'max':[24.0,14.4,4.0]}); self.assertEqual(self.result['volume_m3'],2273.28)
 def test_semantic_parts(self): self.assertEqual([p['part_id'] for p in self.parts['parts']],['foundation','wall_body','wall_walk','inner_parapet','outer_parapet'])
 def test_mesh(self): self.assertEqual(len(self.mesh['vertices_m']),40); self.assertEqual(len(self.mesh['triangles']),60); self.assertTrue(all(0<=i<40 for t in self.mesh['triangles'] for i in t))
 def test_sockets(self): self.assertEqual(len(self.sockets['sockets']),10); self.assertEqual(len({s['socket_id'] for s in self.sockets['sockets']}),10)
 def test_no_cp3_claim(self): self.assertTrue(all(p['surface_coverage']=='DEFERRED_TO_R0A_CP3' for p in self.parts['parts']))
if __name__=='__main__': unittest.main()
'''

    fixture = {
        "schema": "royal-capital.fortification.straight-wall-span/1",
        "span_id": "fortification/caelmere/pilot/straight-span-001",
        "length_m": 24.0,
        "frame": "RH_X_RIGHT_Y_UP_NEG_Z_FORWARD",
        "inside_side": "POSITIVE_Z",
        "outside_side": "NEGATIVE_Z",
        "tolerances": {"linear_m": 1e-7, "angular_rad": 1e-7, "tessellation_linear_m": 0.05, "tessellation_angular_rad": 0.1, "mesh_round_digits": 9},
        "budget": {"max_profile_points": 32, "max_vertices": 128, "max_triangles": 256, "max_artifact_bytes": 2_000_000},
        "runtime": {"build123d_version": "0.13.1.dev12+ge22d34dae", "ocp_version": "8.0.1.0.0", "build123d_source_commit": "e22d34dae17111e5b9fdb361317e055d0daae466", "ocp_wheel_sha256": "1702dfc8f5bdcb916ce65ea0c6314fe453cc78a6a3d49a044fb3062d47b0ef9f"},
    }

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "royal-capital.fortification.straight-wall-span/1",
        "title": "Straight fortification wall span pilot",
        "type": "object", "additionalProperties": False,
        "required": ["schema","span_id","length_m","frame","inside_side","outside_side","tolerances","budget","runtime"],
        "properties": {"schema":{"const":"royal-capital.fortification.straight-wall-span/1"},"span_id":{"type":"string","minLength":1},"length_m":{"type":"number","exclusiveMinimum":0,"maximum":200},"frame":{"const":"RH_X_RIGHT_Y_UP_NEG_Z_FORWARD"},"inside_side":{"const":"POSITIVE_Z"},"outside_side":{"const":"NEGATIVE_Z"},"tolerances":{"type":"object"},"budget":{"type":"object"},"runtime":{"type":"object"}},
    }

    return {
        "src/rcf_fortification_wall_span/model.py": model_py,
        "src/rcf_fortification_wall_span/producer.py": producer_py,
        "src/rcf_fortification_wall_span/__init__.py": init_py,
        "tools/run_cp2_once.py": run_py,
        "tools/validate_cp2.py": validate_py,
        "tests/test_cp2_wall_span.py": test_py,
        "fixtures/straight_wall_span.fixture.json": json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        "schemas/straight_wall_span.schema.json": json.dumps(schema, indent=2, sort_keys=True) + "\n",
        "data/SEMANTIC_PARTS.csv": "part_order,part_id,semantic_role,material_slot,coverage_stage\n0,foundation,FOUNDATION,stone_foundation,R0A-CP3\n1,wall_body,WALL_BODY,stone_wall,R0A-CP3\n2,wall_walk,WALL_WALK,stone_walk,R0A-CP3\n3,inner_parapet,INNER_PARAPET,stone_parapet,R0A-CP3\n4,outer_parapet,OUTER_PARAPET,stone_parapet,R0A-CP3\n",
        "data/SOCKETS.csv": "socket_order,socket_id,role,required\n0,span_start,SPAN_JOIN,true\n1,span_end,SPAN_JOIN,true\n2,wall_walk_start,WALL_WALK_CONTINUATION,true\n3,wall_walk_end,WALL_WALK_CONTINUATION,true\n4,foundation_start,FOUNDATION_INTERFACE,true\n5,foundation_end,FOUNDATION_INTERFACE,true\n6,tower_start,TOWER_JOIN,true\n7,tower_end,TOWER_JOIN,true\n8,utility_inside_01,UTILITY_INSIDE,false\n9,utility_inside_02,UTILITY_INSIDE,false\n",
        "data/FIXED_TESSELLATION.json": json.dumps({"linear_deflection_m":0.05,"angular_deflection_rad":0.1,"mesh_round_digits":9,"part_order":PART_ORDER},indent=2,sort_keys=True)+"\n",
    }


def implement_cp2(tree_root: Path) -> Path:
    cp2 = tree_root / "child_designs/fortification/r0a_cp2"
    if cp2.exists():
        raise RuntimeError("CP2 directory already exists")
    for rel, content in cp2_files().items():
        mode = 0o755 if rel.startswith("tools/") else 0o644
        write_text(cp2 / rel, content, mode)
    write_text(cp2 / "README.md", textwrap.dedent("""\
        # R0A-CP2 — Straight wall-span pilot

        This checkpoint consumes the qualified CP1 neutral provider adapter. It creates one bounded, non-unioned straight wall span with five stable semantic parts, ten named sockets, fixed tessellation and independent clean replay. Surface-to-triangle source coverage and negative/stale gates remain owned by R0A-CP3.

        Pre-validation state: `IMPLEMENTATION_COMPLETE_VALIDATION_PENDING`.
        """))
    write_text(cp2 / "docs/CP2_REPORT.md", "# R0A-CP2 report\n\nImplementation preserved before validation. Functional result pending.\n")
    write_json(cp2 / "provenance/CHANGESET.json", {"schema":"royal-capital.fortification.cp2-changeset/1","stage":"R0A-CP2","status":"IMPLEMENTATION_COMPLETE_VALIDATION_PENDING","baseline":"RCF_D0_full.zip","baseline_sha256":BASELINE_SHA,"resume_checkpoint":"A1.zip","resume_sha256":A1_FULL_SHA,"new_directory":"child_designs/fortification/r0a_cp2","cp3_scope_deferred":["surface-to-triangle source coverage","negative gates","stale/budget rejection","no-partial-output expansion"]})
    return cp2


def set_pre_status(root: Path) -> None:
    write_text(root / "child_designs/fortification/docs/00_STATUS.md", textwrap.dedent("""\
        # 상태

        ```text
        task                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP2
        resume checkpoint            A1.zip
        initial implementation base  RCF_D0_full.zip
        CP0 runtime                  COMPLETE / ADMITTED
        CP1 provider adapter         COMPLETE / QUALIFIED
        straight wall-span source    IMPLEMENTED
        semantic parts/sockets       IMPLEMENTED
        fixed tessellation           IMPLEMENTED
        clean replay                 VALIDATION_PENDING
        stage completion             HOLD_VALIDATION_PENDING
        next                         NONE UNTIL CP2 QUALIFICATION
        ```

        Source preservation, functional qualification and stage completion remain separate.
        """))


def set_final_status(root: Path, validation: Mapping[str, Any]) -> None:
    cp_status = root / "child_designs/fortification/data/CP_STATUS.csv"
    rows = list(csv.DictReader(cp_status.open(newline="", encoding="utf-8")))
    fields = list(rows[0])
    rows = [r for r in rows if not (r["stage"] == "R0A" and r["checkpoint"] == "CP2")]
    rows.append({"stage":"R0A","checkpoint":"CP2","source_preservation":"PASS","remote_self_admission":"NOT_APPLICABLE","local_runtime_admission":"REUSE_CP0_FRESH_PASS_X2","functional_qualification":"PASS","stage_completion":"R0A_CP2_COMPLETE","next":"START_R0A_CP3","blocking_reason":"NONE"})
    with cp_status.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)
    child = root / "data/CHILD_DESIGN_STATUS.csv"
    crows=list(csv.DictReader(child.open(newline="",encoding="utf-8"))); cfields=list(crows[0])
    for row in crows:
        if row["packet_id"]=="RC-FORT-R0-DESIGN": row["implementation_state"]="R0A_CP2_COMPLETE"; row["next"]="RC-FORT-R0A-CP3"
    with child.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cfields,lineterminator="\n"); w.writeheader(); w.writerows(crows)
    write_text(root / "child_designs/fortification/docs/00_STATUS.md", textwrap.dedent("""\
        # 상태

        ```text
        task                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP2
        resume checkpoint            A1.zip
        initial implementation base  RCF_D0_full.zip
        CP0 runtime                  COMPLETE / ADMITTED
        CP1 provider adapter         COMPLETE / QUALIFIED
        straight wall-span           PASS
        semantic parts               5 / 5 PASS
        sockets                      10 / 10 PASS
        fixed tessellation           PASS
        clean A/B                    PASS / BYTE-IDENTICAL
        surface source coverage      DEFERRED TO R0A-CP3
        stage completion             R0A_CP2_COMPLETE / CLOSED
        next                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP3 — SOURCE COVERAGE & NEGATIVE GATES
        R0A-CP3 start                ALLOWED
        ```

        ```text
        source preservation          PASS
        functional qualification     PASS
        stage completion             COMPLETE
        Godot product                NOT_STARTED
        ```
        """))
    write_text(root / "docs/00_STATUS.md", textwrap.dedent("""\
        # ROYAL-CAPITAL-KERNEL 상태

        ```text
        reference_fixture             capital/caelmere-reference@1
        architecture_design          COMPLETE
        finite_roadmap               COMPLETE
        RC-FORT child design         COMPLETE
        RC-FORT implementation       R0A-CP2 COMPLETE
        CAD runtime admission        ADMITTED — CPython 3.13.5 / build123d e22d34d / OCP 8.0.1
        provider adapter             QUALIFIED / NEUTRAL PUBLIC CONTRACT
        straight wall-span pilot     QUALIFIED
        Godot product                NOT_CREATED
        stage completion             RCK_DESIGN_R0_COMPLETE / IMPLEMENTATION ROADMAP OPEN
        next                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP3 — SOURCE COVERAGE & NEGATIVE GATES
        terminal                     ROYAL-CAPITAL-KERNEL-R0F
        ```

        ```text
        RC-FORT-R0-DESIGN            COMPLETE
        RC-FORT-R0A-CP0              COMPLETE / CLOSED
        RC-FORT-R0A-CP1              COMPLETE / CLOSED
        RC-FORT-R0A-CP2              COMPLETE / CLOSED
        RC-FORT accepted CPs         3 / 25
        R0A progress                 3 / 5
        R0A-CP3                      START ALLOWED
        RC-BUILD / RC-ARCH design    NOT_STARTED
        ```
        """))
    report = root / "child_designs/fortification/r0a_cp2/docs/CP2_REPORT.md"
    write_text(report, textwrap.dedent(f"""\
        # ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP2 report

        ```text
        source preservation          PASS
        functional qualification     PASS
        stage completion             R0A_CP2_COMPLETE / CLOSED
        straight span                24.0 m
        semantic parts               5
        named sockets                10
        vertices / triangles         40 / 60
        total volume                 2273.28 m3
        clean replay                 BYTE-IDENTICAL
        focused checks               {validation['summary']['passed']} / {validation['summary']['passed']} PASS
        CP3 coverage                 NOT_AUTHORED / DEFERRED
        ```

        The five semantic solids remain independent bounded modules. No ring-wide union, surface-to-triangle coverage, stale rejection, collision, LOD or Godot product is claimed in CP2.
        """))
    readme = root / "child_designs/fortification/README.md"
    with readme.open("a", encoding="utf-8") as f:
        f.write("\n## R0A-CP2 implementation\n\nStraight wall-span pilot accepted: 5 semantic parts, 10 sockets, fixed tessellation and byte-identical clean replay. Next: R0A-CP3 source coverage and negative gates.\n")
    change = root / "child_designs/fortification/r0a_cp2/provenance/CHANGESET.json"
    data=json.loads(change.read_text()); data["status"]="R0A_CP2_COMPLETE"; data["validation_summary"]=validation["summary"]; write_json(change,data)


def materialize_runtime(python: Path, runtime: Path, cp0: Path, log: Path) -> None:
    if runtime.exists():
        raise RuntimeError(f"runtime destination not fresh: {runtime}")
    subprocess.run([str(python), "-m", "venv", str(runtime)], check=True)
    cmd=[str(runtime/"bin/python"),"-m","pip","install","--no-index","--find-links",str(cp0/"runtime/wheelhouse"),"--require-hashes","-r",str(cp0/"runtime/requirements-lock.txt")]
    with log.open("wb") as f: subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
    subprocess.run([str(runtime/"bin/python"),"-m","pip","check"],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)


def run_validation(tree_root: Path, work: Path) -> dict[str, Any]:
    cp0=tree_root/"child_designs/fortification/r0a_cp0"; cp1=tree_root/"child_designs/fortification/r0a_cp1"; cp2=tree_root/"child_designs/fortification/r0a_cp2"
    validation=work/"validation"; validation.mkdir(parents=True)
    python=Path(sys.executable)
    runtimes=[]
    for label in ("A","B"):
        rt=work/f"runtime_{label}"; materialize_runtime(python,rt,cp0,validation/f"runtime_{label}.log"); runtimes.append(rt)
        env=os.environ.copy(); env["PYTHONPATH"]=os.pathsep.join([str(cp1/"src"),str(cp2/"src")])
        if label=="A":
            audit='import sys,rcf_fortification_wall_span; assert "build123d" not in sys.modules; print("PASS")'
            subprocess.run([str(rt/"bin/python"),"-c",audit],check=True,env=env,stdout=(validation/"public_api_audit.log").open("wb"),stderr=subprocess.STDOUT)
        cmd=[str(rt/"bin/python"),str(cp2/"tools/run_cp2_once.py"),"--cp0-root",str(cp0),"--cp1-root",str(cp1),"--cp2-root",str(cp2),"--fixture",str(cp2/"fixtures/straight_wall_span.fixture.json"),"--output",str(validation/label)]
        with (validation/f"producer_{label}.log").open("wb") as f: subprocess.run(cmd,check=True,env=env,stdout=f,stderr=subprocess.STDOUT)
    env=os.environ.copy(); env["PYTHONPATH"]=os.pathsep.join([str(cp1/"src"),str(cp2/"src")])
    report=validation/"cp2_validation.json"
    subprocess.run([str(runtimes[0]/"bin/python"),str(cp2/"tools/validate_cp2.py"),"--a",str(validation/"A"),"--b",str(validation/"B"),"--report",str(report)],check=True,env=env,stdout=(validation/"cp2_validation.log").open("wb"),stderr=subprocess.STDOUT)
    env["RCF_CP2_ACCEPTED_OUTPUT"]=str(validation/"A")
    test_run=subprocess.run([str(runtimes[0]/"bin/python"),"-m","unittest","discover","-s",str(cp2/"tests"),"-v"],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (validation/"unittest.log").write_bytes(test_run.stdout)
    if test_run.returncode: raise RuntimeError("CP2 unittest failed")
    validation_data=json.loads(report.read_text())
    write_json(validation/"unittest.json",{"schema":"royal-capital.fortification.cp2-unittest/1","status":"PASS","tests":5})
    accepted=cp2/"outputs/reference"; shutil.copytree(validation/"A",accepted)
    reports=cp2/"reports"; reports.mkdir(parents=True,exist_ok=True)
    for name in ("cp2_validation.json","cp2_validation.log","unittest.json","unittest.log","public_api_audit.log"):
        shutil.copy2(validation/name,reports/name)
    compare={"schema":"royal-capital.fortification.cp2-clean-replay/1","status":"PASS","byte_identical":True,"files_compared":validation_data["summary"]["files_compared"],"a_result_sha256":sha_file(validation/"A/result.json"),"b_result_sha256":sha_file(validation/"B/result.json")}
    write_json(reports/"clean_replay.json",compare)
    return validation_data


def scan_hygiene(parent: Path) -> dict[str, Any]:
    unsafe=[]; symlinks=[]; caches=[]; nested=[]
    for p in parent.rglob("*"):
        rel=p.relative_to(parent).as_posix()
        if p.is_symlink(): symlinks.append(rel)
        if any(x in {"__pycache__",".pytest_cache",".mypy_cache",".godot"} for x in p.parts) or p.suffix in {".pyc",".pyo"}: caches.append(rel)
        if p.is_file() and p.suffix.lower()==".zip": nested.append(rel)
        if rel.startswith("/") or ".." in Path(rel).parts: unsafe.append(rel)
    return {"unsafe":unsafe,"symlinks":symlinks,"caches":caches,"nested_zip":nested}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',required=True); ap.add_argument('--wheel-artifact',required=True); ap.add_argument('--work',required=True); ap.add_argument('--packages',required=True); args=ap.parse_args()
    inputs=Path(args.inputs).resolve(); wheel=Path(args.wheel_artifact).resolve(); work=Path(args.work).resolve(); packages=Path(args.packages).resolve()
    if work.exists() or packages.exists(): raise SystemExit('work and packages must be fresh')
    work.mkdir(parents=True); packages.mkdir(parents=True)
    baseline_parent,tree_parent,reconstruction=reconstruct(inputs,wheel,work)
    root=tree_parent/'RC_K0'; implement_cp2(root); set_pre_status(root); update_registry(root); pre_registry=verify_registry(root)
    pre=package_set('A2_pre',baseline_parent,tree_parent,packages)
    write_json(packages/'A2_pre_check.json',{"schema":"royal-capital.fortification.cp2-prevalidation-package/1","status":"PASS","reconstruction":reconstruction,"registry":pre_registry,"packages":pre})
    validation=run_validation(root,work)
    set_final_status(root,validation); update_registry(root); final_registry=verify_registry(root)
    hygiene=scan_hygiene(tree_parent)
    if hygiene['unsafe'] or hygiene['symlinks'] or hygiene['caches'] or hygiene['nested_zip']:
        raise RuntimeError(f"bundle hygiene failure: {hygiene}")
    final=package_set('A2',baseline_parent,tree_parent,packages)
    package_check={"schema":"royal-capital.fortification.cp2-package-verification/1","status":"PASS","baseline":{"file":"RCF_D0_full.zip","sha256":BASELINE_SHA,"files":165},"resume":{"file":"A1.zip","sha256":A1_FULL_SHA,"files":427},"source_preservation":"PASS","functional_qualification":"PASS","stage_completion":"R0A_CP2_COMPLETE","validation":validation,"registry":final_registry,"hygiene":hygiene,"packages":final,"prevalidation_packages":pre,"next":"ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP3"}
    write_json(packages/'A2.json',package_check)
    report=textwrap.dedent(f"""\
        # ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP2 completion report

        ```text
        source preservation          PASS
        functional qualification     PASS
        stage completion             R0A_CP2_COMPLETE / CLOSED
        straight span length         24.0 m
        semantic parts               5 / 5 PASS
        named sockets                10 / 10 PASS
        fixed tessellation           0.05 m / 0.1 rad / 9 digits
        vertices / triangles         40 / 60
        total volume                 2273.28 m3
        clean A/B                    BYTE-IDENTICAL
        focused checks               {validation['summary']['passed']} / {validation['summary']['passed']} PASS
        R0A progress                 3 / 5
        RC-FORT accepted CPs         3 / 25
        next                         ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP3
        ```

        Surface-to-triangle source coverage, stale/budget rejection expansion and generalized no-partial-output gates remain CP3 scope. Collision, LOD, navigation and Godot product work were not run.

        ## Packages

        ```text
        A2.zip   {final['full']['bytes']} bytes  SHA-256 {final['full']['sha256']}
        A2P.zip  {final['patch']['bytes']} bytes  SHA-256 {final['patch']['sha256']}
        A2S.zip  {final['source']['bytes']} bytes  SHA-256 {final['source']['sha256']}
        baseline + cumulative patch = full  PASS
        ```
        """)
    write_text(packages/'A2.md',report)
    write_text(packages/'A2.txt',f"source_preservation=PASS\nfunctional_qualification=PASS\nstage_completion=R0A_CP2_COMPLETE\nchecks={validation['summary']['passed']}/{validation['summary']['passed']}_PASS\nnext=ROYAL-CAPITAL-FORTIFICATION-CAD-R0A-CP3\n")
    sidecars=['A2.zip','A2P.zip','A2S.zip','A2_pre.zip','A2_preP.zip','A2_preS.zip','A2.md','A2.json','A2.txt','A2_pre_check.json']
    write_text(packages/'A2.sha256','\n'.join(f"{sha_file(packages/name)}  {name}" for name in sidecars)+'\n')
    print(json.dumps(package_check,indent=2,sort_keys=True))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
