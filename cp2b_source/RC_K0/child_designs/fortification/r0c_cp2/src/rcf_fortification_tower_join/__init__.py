"""Project-owned bounded tower/span join producer for RC-FORT R0C-CP2."""

from collections.abc import Mapping

from . import fx01 as _impl


def _canonical_frame_axes(socket: Mapping[str, object]):
    """Normalize accepted R0C axis keys and accepted R0B semantic frame keys."""
    frame = socket.get("frame")
    if not isinstance(frame, Mapping):
        raise _impl.JoinError(_impl.JoinFailureCode.INVALID_REQUEST, "socket frame is missing")
    if all(key in frame for key in ("tangent", "up", "inside")):
        return (
            _impl._vec(frame.get("tangent"), "frame.tangent"),
            _impl._vec(frame.get("up"), "frame.up"),
            _impl._vec(frame.get("inside"), "frame.inside"),
        )
    if all(key in frame for key in ("x_axis", "y_axis", "z_axis")):
        return (
            _impl._vec(frame.get("x_axis"), "frame.x_axis"),
            _impl._vec(frame.get("y_axis"), "frame.y_axis"),
            _impl._vec(frame.get("z_axis"), "frame.z_axis"),
        )
    raise _impl.JoinError(
        _impl.JoinFailureCode.INVALID_REQUEST,
        "socket frame must provide tangent/up/inside or x_axis/y_axis/z_axis",
    )


# R0B and R0C are already accepted with two equivalent socket-frame spellings.
# Keep the producer implementation single-path by normalizing only at this boundary.
_impl._frame_axes = _canonical_frame_axes

Fx01TowerJoinProducer = _impl.Fx01TowerJoinProducer
JoinError = _impl.JoinError
JoinFailureCode = _impl.JoinFailureCode
validate_fixture = _impl.validate_fixture

__all__ = [
    "Fx01TowerJoinProducer",
    "JoinError",
    "JoinFailureCode",
    "validate_fixture",
]
