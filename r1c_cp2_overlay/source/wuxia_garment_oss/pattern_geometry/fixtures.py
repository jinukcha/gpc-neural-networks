"""Bounded CP2 rejection fixtures."""
from __future__ import annotations

from wuxia_garment_oss.pattern_geometry.curve import boundary_length, point_at_arc
from wuxia_garment_oss.pattern_geometry.model import ComponentGeometryAuthority, NotchPlacement


def shifted_front_pitch(authority: ComponentGeometryAuthority, delta_normalized: float = 0.12) -> ComponentGeometryAuthority:
    segment_map = authority.segment_map()
    changed = []
    for notch in authority.notches:
        if notch.semantic_role != "FRONT_PITCH":
            changed.append(notch)
            continue
        boundary = authority.boundary(notch.boundary_id)
        length = boundary_length(boundary, segment_map)
        normalized = min(max(notch.normalized_arc + delta_normalized, 0.0), 1.0)
        arc = length * normalized
        point, actual = point_at_arc(boundary, segment_map, arc)
        changed.append(NotchPlacement(notch.notch_id, notch.semantic_role, notch.boundary_id, arc, actual, point))
    return ComponentGeometryAuthority(
        instance_id=authority.instance_id,
        component_id=authority.component_id,
        component_revision=authority.component_revision,
        parameter_set_sha256=authority.parameter_set_sha256,
        mirror_source_instance_id=authority.mirror_source_instance_id,
        segments=authority.segments,
        boundaries=authority.boundaries,
        notches=tuple(changed),
        landmarks=authority.landmarks,
        internal_segment_ids=authority.internal_segment_ids,
    )
