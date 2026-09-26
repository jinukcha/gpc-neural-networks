"""Project-owned bounded tower/span join producers for RC-FORT R0C-CP2."""

from collections.abc import Mapping

from . import fx01 as _fx01


def _canonical_frame_axes(socket: Mapping[str, object]):
    """Normalize accepted R0C axis keys and accepted R0B semantic frame keys."""
    frame = socket.get("frame")
    if not isinstance(frame, Mapping):
        raise _fx01.JoinError(_fx01.JoinFailureCode.INVALID_REQUEST, "socket frame is missing")
    if all(key in frame for key in ("tangent", "up", "inside")):
        return (
            _fx01._vec(frame.get("tangent"), "frame.tangent"),
            _fx01._vec(frame.get("up"), "frame.up"),
            _fx01._vec(frame.get("inside"), "frame.inside"),
        )
    if all(key in frame for key in ("x_axis", "y_axis", "z_axis")):
        return (
            _fx01._vec(frame.get("x_axis"), "frame.x_axis"),
            _fx01._vec(frame.get("y_axis"), "frame.y_axis"),
            _fx01._vec(frame.get("z_axis"), "frame.z_axis"),
        )
    raise _fx01.JoinError(
        _fx01.JoinFailureCode.INVALID_REQUEST,
        "socket frame must provide tangent/up/inside or x_axis/y_axis/z_axis",
    )


# Accepted R0B and R0C use two equivalent socket-frame spellings.
_fx01._frame_axes = _canonical_frame_axes

Fx01TowerJoinProducer = _fx01.Fx01TowerJoinProducer
JoinError = _fx01.JoinError
JoinFailureCode = _fx01.JoinFailureCode
validate_fx01_fixture = _fx01.validate_fixture

from .fx23 import Fx23TowerJoinProducer, validate_fixture as validate_fx23_fixture

__all__ = [
    "Fx01TowerJoinProducer",
    "Fx23TowerJoinProducer",
    "JoinError",
    "JoinFailureCode",
    "validate_fx01_fixture",
    "validate_fx23_fixture",
]
