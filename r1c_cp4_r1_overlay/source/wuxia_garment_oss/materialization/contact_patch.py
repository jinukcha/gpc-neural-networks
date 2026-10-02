"""Apply the CP4-R1 collar contact-owner correction before pipeline import."""
from __future__ import annotations

from pathlib import Path


def apply_contact_owner_patch(package_dir: Path) -> None:
    _patch_settle(package_dir / "settle.py")
    _patch_qualification(package_dir / "qualification.py")


def _replace_once(text: str, before: str, after: str, owner: str) -> str:
    if after in text:
        return text
    if text.count(before) != 1:
        raise RuntimeError(f"unexpected source shape while patching {owner}")
    return text.replace(before, after, 1)


def _patch_settle(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        '        if instance_id.startswith("bodice") or instance_id == "collar":\n'
        '            count += _project_torso(result, indices, profile, clearance)\n',
        '        if instance_id.startswith("bodice"):\n'
        '            count += _project_torso(result, indices, profile, clearance)\n'
        '        elif instance_id == "collar":\n'
        '            count += _project_neck(result, indices, profile, clearance)\n',
        "settle contact dispatch",
    )
    helper = '''\n\ndef _project_neck(positions, indices, profile, clearance):
    radius = profile.neck_circumference_m / (2.0 * np.pi) + clearance
    count = 0
    for index in indices:
        radial = positions[index, (0, 2)]
        distance = float(np.linalg.norm(radial))
        if distance < radius:
            direction = radial / distance if distance > 1.0e-9 else np.array([0.0, 1.0])
            positions[index, 0] = direction[0] * radius
            positions[index, 2] = direction[1] * radius
            count += 1
    return count
'''
    text = _replace_once(text, "\n\ndef _project_arm(positions, indices, profile, side, clearance):", helper + "\n\ndef _project_arm(positions, indices, profile, side, clearance):", "settle neck projector")
    path.write_text(text, encoding="utf-8")


def _patch_qualification(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        '        if instance_id.startswith("bodice") or instance_id == "collar":\n'
        '            penetration[indices] = _torso_penetration(positions[indices], profile)\n',
        '        if instance_id.startswith("bodice"):\n'
        '            penetration[indices] = _torso_penetration(positions[indices], profile)\n'
        '        elif instance_id == "collar":\n'
        '            penetration[indices] = _neck_penetration(positions[indices], profile)\n',
        "qualification contact dispatch",
    )
    helper = '''\n\ndef _neck_penetration(points, profile):
    radius = profile.neck_circumference_m / (2.0 * np.pi)
    radial = np.linalg.norm(points[:, (0, 2)], axis=1)
    return np.maximum(0.0, radius - radial)
'''
    text = _replace_once(text, "\n\ndef _arm_penetration(points, profile, side):", helper + "\n\ndef _arm_penetration(points, profile, side):", "qualification neck metric")
    path.write_text(text, encoding="utf-8")
