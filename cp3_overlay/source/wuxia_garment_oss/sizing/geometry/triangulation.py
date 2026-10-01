"""Deterministic ear clipping and conforming long-edge refinement."""
from __future__ import annotations

from math import hypot

from .polygon import Point, Triangle, cross, ensure_ccw, point_in_triangle, self_intersections, signed_area, triangle_area


def _edge_length(points: list[Point], first: int, second: int) -> float:
    a, b = points[first], points[second]
    return hypot(b[0] - a[0], b[1] - a[1])


def _oriented(points: list[Point], triangle: Triangle) -> Triangle:
    a, b, c = triangle
    return triangle if cross(points[a], points[b], points[c]) > 0.0 else (a, c, b)


def ear_clip(points: list[Point]) -> tuple[list[Point], list[Triangle]]:
    polygon = ensure_ccw(points)
    if len(polygon) < 3:
        raise ValueError("panel polygon has fewer than three vertices")
    intersections = self_intersections(polygon)
    if intersections:
        raise ValueError(f"panel polygon self-intersects: {intersections[:4]}")
    remaining = list(range(len(polygon)))
    triangles: list[Triangle] = []
    guard = len(remaining) * len(remaining)
    while len(remaining) > 3 and guard > 0:
        guard -= 1
        ear = _find_ear(polygon, remaining)
        if ear is None:
            raise ValueError("ear clipping stalled on panel polygon")
        position, triangle = ear
        triangles.append(triangle)
        remaining.pop(position)
    if len(remaining) != 3:
        raise ValueError("ear clipping did not close")
    triangles.append(_oriented(polygon, tuple(remaining)))
    return polygon, triangles


def _find_ear(points: list[Point], remaining: list[int]) -> tuple[int, Triangle] | None:
    for position, current in enumerate(remaining):
        previous = remaining[position - 1]
        following = remaining[(position + 1) % len(remaining)]
        if cross(points[previous], points[current], points[following]) <= 1.0e-12:
            continue
        triangle = (previous, current, following)
        if any(
            point_in_triangle(points[index], points[previous], points[current], points[following])
            for index in remaining
            if index not in triangle
        ):
            continue
        return position, triangle
    return None


def _long_edges(points: list[Point], triangles: list[Triangle], maximum: float) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for a, b, c in triangles:
        for first, second in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((first, second)))
            if _edge_length(points, *edge) > maximum:
                result.add(edge)
    return result


def _midpoints(points: list[Point], edges: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    indices: dict[tuple[int, int], int] = {}
    for edge in sorted(edges):
        first, second = edge
        a, b = points[first], points[second]
        indices[edge] = len(points)
        points.append(((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5))
    return indices


def _split_one(a: int, b: int, c: int, edge: int) -> list[Triangle]:
    if edge == 0:
        return [(a, -1, c), (-1, b, c)]
    if edge == 1:
        return [(b, -1, a), (-1, c, a)]
    return [(c, -1, b), (-1, a, b)]


def _split_two(a: int, b: int, c: int, mids: dict[int, int]) -> list[Triangle]:
    if set(mids) == {0, 1}:
        ab, bc = mids[0], mids[1]
        return [(b, bc, ab), (a, ab, c), (ab, bc, c)]
    if set(mids) == {1, 2}:
        bc, ca = mids[1], mids[2]
        return [(c, ca, bc), (a, b, ca), (b, bc, ca)]
    ab, ca = mids[0], mids[2]
    return [(a, ab, ca), (ab, b, c), (ab, c, ca)]


def _split_triangle(triangle: Triangle, midpoint_indices: dict[tuple[int, int], int]) -> list[Triangle]:
    a, b, c = triangle
    edges = ((a, b), (b, c), (c, a))
    mids = {
        index: midpoint_indices[tuple(sorted(edge))]
        for index, edge in enumerate(edges)
        if tuple(sorted(edge)) in midpoint_indices
    }
    if not mids:
        return [triangle]
    if len(mids) == 1:
        edge, midpoint = next(iter(mids.items()))
        return [tuple(midpoint if value == -1 else value for value in tri) for tri in _split_one(a, b, c, edge)]
    if len(mids) == 2:
        return _split_two(a, b, c, mids)
    ab, bc, ca = mids[0], mids[1], mids[2]
    return [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]


def refine_long_edges(
    points: list[Point], triangles: list[Triangle], maximum_edge_m: float, maximum_rounds: int = 12
) -> tuple[list[Point], list[Triangle], int]:
    vertices = list(points)
    faces = list(triangles)
    rounds = 0
    while rounds < maximum_rounds:
        edges = _long_edges(vertices, faces, maximum_edge_m)
        if not edges:
            break
        midpoint_indices = _midpoints(vertices, edges)
        faces = [
            _oriented(vertices, child)
            for triangle in faces
            for child in _split_triangle(triangle, midpoint_indices)
        ]
        rounds += 1
    if _long_edges(vertices, faces, maximum_edge_m):
        raise ValueError("triangulation refinement did not meet maximum edge gate")
    return vertices, faces, rounds


def triangulate_panel(points: list[Point], maximum_edge_m: float) -> tuple[list[Point], list[Triangle], dict]:
    polygon, initial = ear_clip(points)
    vertices, triangles, rounds = refine_long_edges(polygon, initial, maximum_edge_m)
    polygon_area = abs(signed_area(polygon))
    mesh_area = sum(triangle_area(vertices, triangle) for triangle in triangles)
    degenerate = sum(triangle_area(vertices, triangle) <= 1.0e-12 for triangle in triangles)
    maximum_edge = max(
        _edge_length(vertices, first, second)
        for triangle in triangles
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
    )
    return vertices, triangles, {
        "polygon_vertex_count": len(polygon),
        "mesh_vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "polygon_area_m2": polygon_area,
        "mesh_area_m2": mesh_area,
        "area_coverage_ratio": mesh_area / polygon_area,
        "degenerate_triangle_count": degenerate,
        "maximum_edge_m": maximum_edge,
        "refinement_rounds": rounds,
        "self_intersection_count": 0,
    }
