"""Avatar-aware component arrangement before seam-map compilation."""
from __future__ import annotations

import math
import numpy as np

from .body import BodyProfile, arm_path, arm_radius, local_frame, torso_radii
from .model import ComponentMesh


def arrange_components(meshes: dict[str, ComponentMesh], profile: BodyProfile) -> dict:
    for mesh in meshes.values():
        if mesh.component_id.startswith("BODICE_"):
            _arrange_bodice(mesh, profile)
        elif mesh.component_id == "SET_IN_SLEEVE_BASIC":
            _arrange_sleeve(mesh, profile)
        elif mesh.component_id == "COLLAR_STAND_BASIC":
            _arrange_collar(mesh, profile)
        elif mesh.component_id == "CUFF_STRAIGHT_BASIC":
            _arrange_cuff(mesh, profile)
        else:
            raise ValueError(f"unsupported CP4-R1 component: {mesh.component_id}")
        mesh.validate_3d()
    return {
        "contract": "AvatarArrangementReceipt/1",
        "component_count": len(meshes),
        "arrangement": "BODY_SECTION_WRAP_PLUS_SLEEVE_TUBE_CAP_PATCH_PRESEED",
        "body_profile": profile.to_dict(),
        "components": [mesh.to_summary() for mesh in meshes.values()],
    }


def _arrange_bodice(mesh: ComponentMesh, profile: BodyProfile) -> None:
    points = mesh.vertices_2d
    lower, upper = points.min(axis=0), points.max(axis=0)
    front = mesh.component_id == "BODICE_FRONT_BASIC"
    result = np.empty((len(points), 3), dtype=np.float64)
    for index, point in enumerate(points):
        vertical = (point[1] - lower[1]) / max(upper[1] - lower[1], 1.0e-9)
        y = 0.80 + vertical * 0.66
        half_width = max(abs(lower[0]), abs(upper[0]), 1.0e-6)
        angle = min(max(point[0] / half_width, -1.0), 1.0) * math.pi * 0.5
        rx, rz = torso_radii(profile, y)
        clearance = 0.014
        x = (rx + clearance) * math.sin(angle)
        z = (rz + clearance) * math.cos(angle)
        result[index] = (x, y, z if front else -z)
    mesh.vertices_3d = result
    mesh.fixed_mask = _boundary_mask(mesh)
    mesh.metadata.update({"arrangement_owner": "BODICE_SECTION_WRAP", "body_clearance_m": 0.014})


def _arrange_sleeve(mesh: ComponentMesh, profile: BodyProfile) -> None:
    points = mesh.vertices_2d
    lower, upper = points.min(axis=0), points.max(axis=0)
    side = "LEFT" if mesh.instance_id.endswith("left") else "RIGHT"
    result = np.empty((len(points), 3), dtype=np.float64)
    cap_height = max(0.08, upper[1] - _sleeve_underarm_y(mesh))
    for index, point in enumerate(points):
        shoulder_to_wrist = (upper[1] - point[1]) / max(upper[1] - lower[1], 1.0e-9)
        t = min(max(shoulder_to_wrist, 0.0), 1.0)
        centre, tangent = arm_path(profile, side, t)
        first, second = local_frame(tangent)
        bounds = _sleeve_width_at_y(mesh, float(point[1]))
        u = (point[0] - bounds[0]) / max(bounds[1] - bounds[0], 1.0e-9)
        angle = 2.0 * math.pi * min(max(u, 0.0), 1.0)
        radius = arm_radius(profile, t) + 0.014
        if point[1] > upper[1] - cap_height:
            cap_t = (point[1] - (upper[1] - cap_height)) / cap_height
            radius *= 1.0 - 0.45 * cap_t
            centre = centre + np.array([0.0, 0.018 * cap_t, 0.0])
        result[index] = centre + radius * (first * math.cos(angle) + second * math.sin(angle))
    mesh.vertices_3d = result
    mesh.fixed_mask = _boundary_mask(mesh)
    mesh.metadata.update({"arrangement_owner": "SLEEVE_TUBE_CAP_PATCH_PRESEED", "side": side})


def _arrange_collar(mesh: ComponentMesh, profile: BodyProfile) -> None:
    points = mesh.vertices_2d
    lower, upper = points.min(axis=0), points.max(axis=0)
    total = max(upper[0] - lower[0], 1.0e-9)
    neck_radius = profile.neck_circumference_m / (2.0 * math.pi) + 0.006
    result = np.empty((len(points), 3), dtype=np.float64)
    for index, point in enumerate(points):
        angle = 2.0 * math.pi * (point[0] - lower[0]) / total
        height = 1.455 + (point[1] - lower[1])
        result[index] = (neck_radius * math.sin(angle), height, neck_radius * math.cos(angle))
    mesh.vertices_3d = result
    mesh.fixed_mask = _boundary_mask(mesh)
    mesh.metadata["arrangement_owner"] = "COLLAR_NECK_WRAP"


def _arrange_cuff(mesh: ComponentMesh, profile: BodyProfile) -> None:
    side = "LEFT" if mesh.instance_id.endswith("left") else "RIGHT"
    points = mesh.vertices_2d
    lower, upper = points.min(axis=0), points.max(axis=0)
    width = max(upper[0] - lower[0], 1.0e-9)
    centre, tangent = arm_path(profile, side, 1.0)
    first, second = local_frame(tangent)
    radius = profile.wrist_circumference_m / (2.0 * math.pi) + 0.009
    result = np.empty((len(points), 3), dtype=np.float64)
    for index, point in enumerate(points):
        angle = 2.0 * math.pi * (point[0] - lower[0]) / width
        axial = (point[1] - lower[1]) - 0.5 * (upper[1] - lower[1])
        result[index] = centre + radius * (first * math.cos(angle) + second * math.sin(angle)) + tangent * axial
    mesh.vertices_3d = result
    mesh.fixed_mask = _boundary_mask(mesh)
    mesh.metadata.update({"arrangement_owner": "CUFF_WRIST_WRAP", "side": side})


def _boundary_mask(mesh: ComponentMesh) -> np.ndarray:
    mask = np.zeros(len(mesh.vertices_2d), dtype=np.bool_)
    for indices in mesh.boundary_indices.values():
        mask[indices] = True
    return mask


def _sleeve_underarm_y(mesh: ComponentMesh) -> float:
    indices = np.concatenate((mesh.boundary_indices["underarm_front"], mesh.boundary_indices["underarm_back"]))
    return float(mesh.vertices_2d[indices, 1].max())


def _sleeve_width_at_y(mesh: ComponentMesh, y: float) -> tuple[float, float]:
    points = mesh.vertices_2d
    tolerance = max((points[:, 1].max() - points[:, 1].min()) * 0.035, 0.005)
    selected = points[np.abs(points[:, 1] - y) <= tolerance]
    if len(selected) < 2:
        selected = points
    return float(selected[:, 0].min()), float(selected[:, 0].max())
