"""Named professional motion-fit pose suite and pose-specific acceptance policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ...pattern_cad.document.model import canonical_sha256


@dataclass(frozen=True)
class MotionPose:
    pose_id: str
    display_name: str
    arm_forward: float = 0.0
    arm_raise: float = 0.0
    cross_body: float = 0.0
    elbow_bend: float = 0.0
    torso_twist: float = 0.0
    forward_bend: float = 0.0
    hip_flexion: float = 0.0
    squat: float = 0.0
    stride: float = 0.0
    strain_p99_limit: float = 0.36
    pressure_p99_kpa_limit: float = 18.0
    seam_tension_p95_n_m_limit: float = 850.0
    mobility_restriction_p95_limit: float = 0.38

    def validate(self) -> None:
        if not self.pose_id or not self.display_name:
            raise ValueError("pose identity and name are required")
        amplitudes = (
            self.arm_forward,
            self.arm_raise,
            self.cross_body,
            self.elbow_bend,
            self.torso_twist,
            self.forward_bend,
            self.hip_flexion,
            self.squat,
            self.stride,
        )
        if any(not -1.0 <= value <= 1.0 for value in amplitudes):
            raise ValueError(f"pose amplitude outside [-1, 1]: {self.pose_id}")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class MotionFitSuite:
    suite_id: str
    poses: tuple[MotionPose, ...]
    frames_per_pose: int = 12
    substeps_per_frame: int = 2
    projection_iterations: int = 5

    def validate(self) -> None:
        if not self.suite_id or len(self.poses) < 2:
            raise ValueError("motion suite identity and poses are required")
        ids = [item.pose_id for item in self.poses]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate motion pose identity")
        if self.frames_per_pose < 4 or self.substeps_per_frame < 1:
            raise ValueError("motion schedule is too short")
        if not 2 <= self.projection_iterations <= 12:
            raise ValueError("invalid projection iteration count")
        for pose in self.poses:
            pose.validate()

    def to_dict(self) -> dict:
        self.validate()
        payload = {
            "contract": "MotionFitSuite/1",
            "suite_id": self.suite_id,
            "poses": [item.to_dict() for item in self.poses],
            "frames_per_pose": self.frames_per_pose,
            "substeps_per_frame": self.substeps_per_frame,
            "projection_iterations": self.projection_iterations,
        }
        payload["suite_sha256"] = canonical_sha256(payload)
        return payload


def professional_tunic_suite() -> MotionFitSuite:
    return MotionFitSuite(
        "TUNIC_PROFESSIONAL_MOTION_SUITE_V1",
        (
            MotionPose("NEUTRAL_A", "Neutral A-pose", strain_p99_limit=0.25, mobility_restriction_p95_limit=0.25),
            MotionPose("ARMS_FORWARD", "Arms forward", arm_forward=0.72),
            MotionPose("ARMS_OVERHEAD", "Arms overhead", arm_raise=0.90, strain_p99_limit=0.42, pressure_p99_kpa_limit=22.0),
            MotionPose("CROSS_BODY_REACH", "Cross-body reach", cross_body=0.82, torso_twist=0.18, strain_p99_limit=0.42),
            MotionPose("DEEP_ELBOW_BEND", "Deep elbow bend", arm_forward=0.38, elbow_bend=0.95),
            MotionPose("TORSO_TWIST", "Torso twist", torso_twist=0.82, strain_p99_limit=0.40),
            MotionPose("FORWARD_BEND", "Forward bend", forward_bend=0.78, hip_flexion=0.30, pressure_p99_kpa_limit=22.0),
            MotionPose("SEATED", "Seated", hip_flexion=0.82, strain_p99_limit=0.43, pressure_p99_kpa_limit=24.0),
            MotionPose("SQUAT", "Squat", hip_flexion=0.68, squat=0.88, strain_p99_limit=0.46, pressure_p99_kpa_limit=25.0),
            MotionPose("WALK_STRIDE", "Walk stride", stride=0.92, hip_flexion=0.22, strain_p99_limit=0.42),
        ),
    )
