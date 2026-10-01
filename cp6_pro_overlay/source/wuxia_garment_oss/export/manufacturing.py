"""Deterministic 1:1 manufacturing package, SVG, and ASCII DXF export."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..pattern_cad.document.model import canonical_sha256


Point2 = tuple[float, float]


def _lerp(a: Point2, b: Point2, t: float) -> Point2:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def curve_point(curve: Mapping[str, object], t: float) -> Point2:
    rows = [(float(item[0]), float(item[1])) for item in curve["points"]]
    kind = str(curve["curve_type"])
    if kind == "LINE":
        return _lerp(rows[0], rows[1], t)
    if kind == "QUADRATIC_BEZIER":
        return _lerp(_lerp(rows[0], rows[1], t), _lerp(rows[1], rows[2], t), t)
    if kind == "CUBIC_BEZIER":
        a = _lerp(rows[0], rows[1], t)
        b = _lerp(rows[1], rows[2], t)
        c = _lerp(rows[2], rows[3], t)
        return _lerp(_lerp(a, b, t), _lerp(b, c, t), t)
    raise ValueError(f"unsupported curve type: {kind}")


def sample_curve(curve: Mapping[str, object], count: int = 48) -> list[Point2]:
    if count < 2:
        raise ValueError("curve sample count must be >= 2")
    return [curve_point(curve, index / count) for index in range(count + 1)]


def polyline_length(points: Sequence[Point2]) -> float:
    return sum(math.dist(points[index - 1], points[index]) for index in range(1, len(points)))


def _normal(points: Sequence[Point2], index: int, centroid: Point2) -> Point2:
    left = points[max(index - 1, 0)]
    right = points[min(index + 1, len(points) - 1)]
    dx, dy = right[0] - left[0], right[1] - left[1]
    length = math.hypot(dx, dy)
    if length <= 1.0e-12:
        return 0.0, 0.0
    nx, ny = -dy / length, dx / length
    rx, ry = points[index][0] - centroid[0], points[index][1] - centroid[1]
    if nx * rx + ny * ry < 0.0:
        nx, ny = -nx, -ny
    return nx, ny


def offset_polyline(points: Sequence[Point2], centroid: Point2, distance: float) -> list[Point2]:
    return [
        (point[0] + _normal(points, index, centroid)[0] * distance,
         point[1] + _normal(points, index, centroid)[1] * distance)
        for index, point in enumerate(points)
    ]


def _allowances(construction: Mapping[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for seam in construction.get("seam_specs", []):
        values[str(seam["side_a"]["curve_id"])] = float(seam["allowance_a_m"])
        values[str(seam["side_b"]["curve_id"])] = float(seam["allowance_b_m"])
    for finish_row in construction.get("edge_finishes", []):
        finish = finish_row["finish"]
        values[str(finish["boundary"]["curve_id"])] = float(finish["allowance_m"])
    return values


def _panel_centroid(curves: Sequence[Mapping[str, object]]) -> Point2:
    points = [point for curve in curves for point in sample_curve(curve, 12)]
    return (
        sum(point[0] for point in points) / max(len(points), 1),
        sum(point[1] for point in points) / max(len(points), 1),
    )


def _curve_record(curve: Mapping[str, object], centroid: Point2, allowance: float) -> dict:
    stitch = sample_curve(curve)
    cut = offset_polyline(stitch, centroid, allowance)
    return {
        "curve_id": str(curve["curve_id"]),
        "boundary_role": str(curve["boundary_role"]),
        "disposition": str(curve["disposition"]),
        "allowance_m": allowance,
        "stitch_line": [list(item) for item in stitch],
        "cut_line": [list(item) for item in cut],
        "stitch_length_m": polyline_length(stitch),
        "cut_length_m": polyline_length(cut),
    }


def _panel_records(resolved: Mapping[str, object], construction: Mapping[str, object]) -> list[dict]:
    by_panel: dict[str, list[Mapping[str, object]]] = {}
    for curve in resolved["curves"].values():
        if str(curve["disposition"]) == "INTERNAL":
            continue
        by_panel.setdefault(str(curve["panel_id"]), []).append(curve)
    allowances = _allowances(construction)
    panels = []
    for panel_id, curves in sorted(by_panel.items()):
        centroid = _panel_centroid(curves)
        rows = [
            _curve_record(curve, centroid, allowances.get(str(curve["curve_id"]), 0.012))
            for curve in sorted(curves, key=lambda item: str(item["curve_id"]))
        ]
        panel_points = [point for row in rows for point in row["cut_line"]]
        panels.append({
            "panel_id": panel_id,
            "grainline": [[centroid[0], min(point[1] for point in panel_points)],
                          [centroid[0], max(point[1] for point in panel_points)]],
            "label_position": list(centroid),
            "curves": rows,
            "cut_bbox_m": [
                min(point[0] for point in panel_points),
                min(point[1] for point in panel_points),
                max(point[0] for point in panel_points),
                max(point[1] for point in panel_points),
            ],
        })
    return panels


def compile_manufacturing_package(
    garment_id: str,
    resolved: Mapping[str, object],
    construction: Mapping[str, object],
    source_authority: Mapping[str, object],
) -> dict:
    panels = _panel_records(resolved, construction)
    payload = {
        "contract": "ManufacturingPatternPackage/1",
        "garment_id": garment_id,
        "unit": "metre",
        "export_unit": "millimetre",
        "scale": "1:1",
        "panels": panels,
        "notches": list(construction.get("notch_correspondence", [])),
        "facings": list(construction.get("facings", [])),
        "closures": list(construction.get("closures", [])),
        "layer_pieces": list(construction.get("layer_pieces", [])),
        "turn_of_cloth": list(construction.get("turn_of_cloth", [])),
        "assembly_plan": construction.get("assembly_plan", {}),
        "bill_of_materials": list(construction.get("bill_of_materials", [])),
        "source_authority": dict(source_authority),
        "standards_claim": "OPEN_EXCHANGE_PILOT_NOT_AAMA_DXF_CERTIFIED",
    }
    payload["package_sha256"] = canonical_sha256(payload)
    return payload


def _bounds(panels: Sequence[Mapping[str, object]]) -> tuple[float, float, float, float]:
    boxes = [panel["cut_bbox_m"] for panel in panels]
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _layout(panels: Sequence[Mapping[str, object]]) -> dict[str, Point2]:
    offsets: dict[str, Point2] = {}
    cursor_x = 0.0
    row_y = 0.0
    row_height = 0.0
    for index, panel in enumerate(panels):
        box = panel["cut_bbox_m"]
        width = float(box[2]) - float(box[0])
        height = float(box[3]) - float(box[1])
        if index and index % 3 == 0:
            cursor_x = 0.0
            row_y += row_height + 0.08
            row_height = 0.0
        offsets[str(panel["panel_id"])] = (cursor_x - float(box[0]), row_y - float(box[1]))
        cursor_x += width + 0.08
        row_height = max(row_height, height)
    return offsets


def _svg_polyline(points: Iterable[Sequence[float]], offset: Point2) -> str:
    return " ".join(
        f"{(float(point[0]) + offset[0]) * 1000.0:.3f},{-(float(point[1]) + offset[1]) * 1000.0:.3f}"
        for point in points
    )


def write_svg(package: Mapping[str, object], path: Path) -> None:
    panels = package["panels"]
    offsets = _layout(panels)
    laid_out = []
    for panel in panels:
        ox, oy = offsets[str(panel["panel_id"])]
        box = panel["cut_bbox_m"]
        laid_out.append((float(box[2]) + ox, float(box[3]) + oy))
    width = max(item[0] for item in laid_out) * 1000.0 + 40.0
    height = max(item[1] for item in laid_out) * 1000.0 + 40.0
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}mm" height="{height:.3f}mm" viewBox="0 {-height:.3f} {width:.3f} {height:.3f}">',
        '<g fill="none" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for panel in panels:
        panel_id = str(panel["panel_id"])
        offset = offsets[panel_id]
        rows.append(f'<g id="{html.escape(panel_id)}">')
        for curve in panel["curves"]:
            rows.append(f'<polyline points="{_svg_polyline(curve["cut_line"], offset)}" stroke="#111" stroke-width="0.45"/>')
            rows.append(f'<polyline points="{_svg_polyline(curve["stitch_line"], offset)}" stroke="#666" stroke-width="0.25" stroke-dasharray="3 2"/>')
        rows.append(f'<polyline points="{_svg_polyline(panel["grainline"], offset)}" stroke="#333" stroke-width="0.25" marker-end="url(#arrow)"/>')
        label = panel["label_position"]
        x = (float(label[0]) + offset[0]) * 1000.0
        y = -(float(label[1]) + offset[1]) * 1000.0
        rows.append(f'<text x="{x:.3f}" y="{y:.3f}" font-size="7" fill="#111">{html.escape(panel_id)}</text>')
        rows.append("</g>")
    rows.extend(("</g>", "</svg>"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _dxf_line(layer: str, a: Sequence[float], b: Sequence[float]) -> list[str]:
    return ["0", "LINE", "8", layer, "10", f"{float(a[0]) * 1000.0:.6f}", "20", f"{float(a[1]) * 1000.0:.6f}",
            "30", "0.0", "11", f"{float(b[0]) * 1000.0:.6f}", "21", f"{float(b[1]) * 1000.0:.6f}", "31", "0.0"]


def _segments(points: Sequence[Sequence[float]]) -> Iterable[tuple[Sequence[float], Sequence[float]]]:
    return zip(points[:-1], points[1:])


def write_dxf(package: Mapping[str, object], path: Path) -> None:
    rows = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for panel in package["panels"]:
        for curve in panel["curves"]:
            for a, b in _segments(curve["cut_line"]):
                rows.extend(_dxf_line("CUT", a, b))
            for a, b in _segments(curve["stitch_line"]):
                rows.extend(_dxf_line("STITCH", a, b))
        rows.extend(_dxf_line("GRAIN", panel["grainline"][0], panel["grainline"][1]))
        label = panel["label_position"]
        rows.extend(("0", "TEXT", "8", "LABEL", "10", f"{float(label[0]) * 1000.0:.6f}", "20", f"{float(label[1]) * 1000.0:.6f}", "30", "0.0", "40", "7.0", "1", str(panel["panel_id"])))
    for notch in package.get("notches", []):
        for key in ("side_a_position", "side_b_position"):
            point = notch.get(key)
            if point:
                a = [float(point[0]) - 0.003, float(point[1])]
                b = [float(point[0]) + 0.003, float(point[1])]
                rows.extend(_dxf_line("NOTCH", a, b))
    rows.extend(("0", "ENDSEC", "0", "EOF"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def write_package(package: Mapping[str, object], directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "manufacturing_pattern_package.json"
    svg_path = directory / "manufacturing_pattern_1to1.svg"
    dxf_path = directory / "manufacturing_pattern_r12.dxf"
    json_path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_svg(package, svg_path)
    write_dxf(package, dxf_path)
    return {"json": json_path, "svg": svg_path, "dxf": dxf_path}
