"""Skeleton Solver - Convert joint positions to UE5 Mannequin bone rotations"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from src.core import get_logger, Config
from src.pose.reconstructor_3d import Pose3D, Joint3D
from .root_motion import RootTransform


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector, return zero vector if length is zero."""
    length = np.linalg.norm(v)
    if length < 1e-8:
        return np.zeros_like(v)
    return v / length


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create quaternion from axis and angle."""
    axis = normalize(axis)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quaternion_from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Create quaternion that rotates v1 to v2.
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    
    dot = np.dot(v1, v2)
    
    if dot > 0.9999:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    
    if dot < -0.9999:
        ortho = np.array([1, 0, 0], dtype=np.float32)
        if abs(v1[0]) > 0.9:
            ortho = np.array([0, 1, 0], dtype=np.float32)
        axis = normalize(np.cross(v1, ortho))
        return np.array([0, axis[0], axis[1], axis[2]], dtype=np.float32)
    
    axis = np.cross(v1, v2)
    s = np.sqrt((1 + dot) * 2)
    invs = 1 / s
    
    return np.array([
        s * 0.5,
        axis[0] * invs,
        axis[1] * invs,
        axis[2] * invs
    ], dtype=np.float32)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (XYZ order) in radians."""
    w, x, y, z = q
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw], dtype=np.float32)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z], dtype=np.float32)


class UE5Bone(Enum):
    """UE5 Mannequin skeleton bone names."""
    ROOT = "root"
    PELVIS = "pelvis"
    SPINE_01 = "spine_01"
    SPINE_02 = "spine_02"
    SPINE_03 = "spine_03"
    NECK_01 = "neck_01"
    HEAD = "head"
    CLAVICLE_L = "clavicle_l"
    UPPERARM_L = "upperarm_l"
    LOWERARM_L = "lowerarm_l"
    HAND_L = "hand_l"
    CLAVICLE_R = "clavicle_r"
    UPPERARM_R = "upperarm_r"
    LOWERARM_R = "lowerarm_r"
    HAND_R = "hand_r"
    THIGH_L = "thigh_l"
    CALF_L = "calf_l"
    FOOT_L = "foot_l"
    BALL_L = "ball_l"
    THIGH_R = "thigh_r"
    CALF_R = "calf_r"
    FOOT_R = "foot_r"
    BALL_R = "ball_r"


MEDIAPIPE_TO_UE5_MAPPING = {
    "left_hip": UE5Bone.THIGH_L,
    "right_hip": UE5Bone.THIGH_R,
    "left_knee": UE5Bone.CALF_L,
    "right_knee": UE5Bone.CALF_R,
    "left_ankle": UE5Bone.FOOT_L,
    "right_ankle": UE5Bone.FOOT_R,
    "left_shoulder": UE5Bone.UPPERARM_L,
    "right_shoulder": UE5Bone.UPPERARM_R,
    "left_elbow": UE5Bone.LOWERARM_L,
    "right_elbow": UE5Bone.LOWERARM_R,
    "left_wrist": UE5Bone.HAND_L,
    "right_wrist": UE5Bone.HAND_R,
    "nose": UE5Bone.HEAD,
}

UE5_BONE_HIERARCHY = {
    UE5Bone.ROOT: None,
    UE5Bone.PELVIS: UE5Bone.ROOT,
    UE5Bone.SPINE_01: UE5Bone.PELVIS,
    UE5Bone.SPINE_02: UE5Bone.SPINE_01,
    UE5Bone.SPINE_03: UE5Bone.SPINE_02,
    UE5Bone.NECK_01: UE5Bone.SPINE_03,
    UE5Bone.HEAD: UE5Bone.NECK_01,
    UE5Bone.CLAVICLE_L: UE5Bone.SPINE_03,
    UE5Bone.UPPERARM_L: UE5Bone.CLAVICLE_L,
    UE5Bone.LOWERARM_L: UE5Bone.UPPERARM_L,
    UE5Bone.HAND_L: UE5Bone.LOWERARM_L,
    UE5Bone.CLAVICLE_R: UE5Bone.SPINE_03,
    UE5Bone.UPPERARM_R: UE5Bone.CLAVICLE_R,
    UE5Bone.LOWERARM_R: UE5Bone.UPPERARM_R,
    UE5Bone.HAND_R: UE5Bone.LOWERARM_R,
    UE5Bone.THIGH_L: UE5Bone.PELVIS,
    UE5Bone.CALF_L: UE5Bone.THIGH_L,
    UE5Bone.FOOT_L: UE5Bone.CALF_L,
    UE5Bone.BALL_L: UE5Bone.FOOT_L,
    UE5Bone.THIGH_R: UE5Bone.PELVIS,
    UE5Bone.CALF_R: UE5Bone.THIGH_R,
    UE5Bone.FOOT_R: UE5Bone.CALF_R,
    UE5Bone.BALL_R: UE5Bone.FOOT_R,
}

UE5_REST_POSE_DIRECTIONS = {
    UE5Bone.SPINE_01: np.array([0, 1, 0], dtype=np.float32),
    UE5Bone.SPINE_02: np.array([0, 1, 0], dtype=np.float32),
    UE5Bone.SPINE_03: np.array([0, 1, 0], dtype=np.float32),
    UE5Bone.NECK_01: np.array([0, 1, 0], dtype=np.float32),
    UE5Bone.HEAD: np.array([0, 1, 0], dtype=np.float32),
    UE5Bone.UPPERARM_L: np.array([-1, 0, 0], dtype=np.float32),
    UE5Bone.LOWERARM_L: np.array([-1, 0, 0], dtype=np.float32),
    UE5Bone.HAND_L: np.array([-1, 0, 0], dtype=np.float32),
    UE5Bone.UPPERARM_R: np.array([1, 0, 0], dtype=np.float32),
    UE5Bone.LOWERARM_R: np.array([1, 0, 0], dtype=np.float32),
    UE5Bone.HAND_R: np.array([1, 0, 0], dtype=np.float32),
    UE5Bone.THIGH_L: np.array([0, -1, 0], dtype=np.float32),
    UE5Bone.CALF_L: np.array([0, -1, 0], dtype=np.float32),
    UE5Bone.FOOT_L: np.array([0, 0, 1], dtype=np.float32),
    UE5Bone.THIGH_R: np.array([0, -1, 0], dtype=np.float32),
    UE5Bone.CALF_R: np.array([0, -1, 0], dtype=np.float32),
    UE5Bone.FOOT_R: np.array([0, 0, 1], dtype=np.float32),
}


@dataclass
class BoneTransform:
    """Transform for a single bone."""
    bone: UE5Bone
    rotation: np.ndarray  # Quaternion (w, x, y, z)
    local_rotation: np.ndarray  # Local rotation relative to parent
    
    @property
    def euler(self) -> np.ndarray:
        """Get rotation as Euler angles in degrees."""
        return np.degrees(quaternion_to_euler(self.rotation))
    
    @property
    def local_euler(self) -> np.ndarray:
        """Get local rotation as Euler angles in degrees."""
        return np.degrees(quaternion_to_euler(self.local_rotation))


@dataclass
class SkeletonPose:
    """Complete skeleton pose for a single frame."""
    frame_number: int
    timestamp: float
    root_position: np.ndarray  # (3,) world position
    root_rotation: np.ndarray  # Quaternion
    bone_transforms: Dict[UE5Bone, BoneTransform] = field(default_factory=dict)
    
    def get_bone(self, bone: UE5Bone) -> Optional[BoneTransform]:
        return self.bone_transforms.get(bone)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for export."""
        return {
            "frame": self.frame_number,
            "timestamp": self.timestamp,
            "root_position": self.root_position.tolist(),
            "root_rotation": self.root_rotation.tolist(),
            "bones": {
                bone.value: {
                    "rotation": transform.rotation.tolist(),
                    "local_rotation": transform.local_rotation.tolist()
                }
                for bone, transform in self.bone_transforms.items()
            }
        }


class SkeletonSolver:
    """
    Convert 3D joint positions to UE5 Mannequin bone rotations.
    
    Features:
    - Maps MediaPipe joints to UE5 Mannequin bones
    - Computes local bone rotations using quaternions
    - Preserves rest pose offsets
    - Outputs animation-ready transform data
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("motion.skeleton")
        self.config = config or Config()
        
        skeleton_config = self.config.skeleton
        
        self._target = skeleton_config.get("target", "ue5_mannequin")
        self._apply_rest_pose = skeleton_config.get("apply_rest_pose", True)
        
        self._pose_history: List[SkeletonPose] = []
        self._max_history = 300
        self._frame_count = 0
        
        self.logger.info(f"Initialized skeleton solver (target={self._target})")
    
    def solve(
        self,
        pose_3d: Pose3D,
        root_transform: Optional[RootTransform] = None
    ) -> SkeletonPose:
        """
        Solve skeleton pose from 3D joint positions.
        
        Args:
            pose_3d: 3D pose with joint positions
            root_transform: Optional root motion data
        
        Returns:
            SkeletonPose with bone rotations
        """
        if root_transform is not None:
            root_position = root_transform.position.copy()
            root_rotation = self._compute_root_rotation(root_transform.facing_direction)
        else:
            pelvis = pose_3d.pelvis
            root_position = pelvis if pelvis is not None else np.zeros(3, dtype=np.float32)
            root_rotation = np.array([1, 0, 0, 0], dtype=np.float32)
        
        bone_transforms = {}
        
        pelvis_rot = self._solve_pelvis(pose_3d)
        bone_transforms[UE5Bone.PELVIS] = pelvis_rot
        
        spine_rots = self._solve_spine(pose_3d)
        bone_transforms.update(spine_rots)
        
        left_arm_rots = self._solve_arm(pose_3d, "left")
        bone_transforms.update(left_arm_rots)
        
        right_arm_rots = self._solve_arm(pose_3d, "right")
        bone_transforms.update(right_arm_rots)
        
        left_leg_rots = self._solve_leg(pose_3d, "left")
        bone_transforms.update(left_leg_rots)
        
        right_leg_rots = self._solve_leg(pose_3d, "right")
        bone_transforms.update(right_leg_rots)
        
        skeleton_pose = SkeletonPose(
            frame_number=self._frame_count,
            timestamp=pose_3d.timestamp,
            root_position=root_position,
            root_rotation=root_rotation,
            bone_transforms=bone_transforms
        )
        
        self._pose_history.append(skeleton_pose)
        if len(self._pose_history) > self._max_history:
            self._pose_history.pop(0)
        
        self._frame_count += 1
        
        return skeleton_pose
    
    def _compute_root_rotation(self, facing: np.ndarray) -> np.ndarray:
        """Compute root rotation from facing direction."""
        forward = np.array([0, 0, 1], dtype=np.float32)
        facing_xz = normalize(np.array([facing[0], 0, facing[2]], dtype=np.float32))
        
        return quaternion_from_two_vectors(forward, facing_xz)
    
    def _solve_pelvis(self, pose: Pose3D) -> BoneTransform:
        """Solve pelvis rotation."""
        left_hip = pose.joints.get("left_hip")
        right_hip = pose.joints.get("right_hip")
        
        if left_hip and right_hip:
            hip_vec = right_hip.position - left_hip.position
            hip_vec = normalize(hip_vec)
            
            rest_hip = np.array([1, 0, 0], dtype=np.float32)
            rotation = quaternion_from_two_vectors(rest_hip, hip_vec)
        else:
            rotation = np.array([1, 0, 0, 0], dtype=np.float32)
        
        return BoneTransform(
            bone=UE5Bone.PELVIS,
            rotation=rotation,
            local_rotation=rotation
        )
    
    def _solve_spine(self, pose: Pose3D) -> Dict[UE5Bone, BoneTransform]:
        """Solve spine chain rotations."""
        result = {}
        
        pelvis = pose.pelvis
        chest = pose.chest
        
        if pelvis is not None and chest is not None:
            spine_dir = normalize(chest - pelvis)
            rest_dir = UE5_REST_POSE_DIRECTIONS[UE5Bone.SPINE_01]
            
            spine_rot = quaternion_from_two_vectors(rest_dir, spine_dir)
            
            for bone in [UE5Bone.SPINE_01, UE5Bone.SPINE_02, UE5Bone.SPINE_03]:
                result[bone] = BoneTransform(
                    bone=bone,
                    rotation=spine_rot,
                    local_rotation=spine_rot
                )
        else:
            identity = np.array([1, 0, 0, 0], dtype=np.float32)
            for bone in [UE5Bone.SPINE_01, UE5Bone.SPINE_02, UE5Bone.SPINE_03]:
                result[bone] = BoneTransform(
                    bone=bone,
                    rotation=identity,
                    local_rotation=identity
                )
        
        nose = pose.joints.get("nose")
        if chest is not None and nose:
            neck_dir = normalize(nose.position - chest)
            rest_dir = UE5_REST_POSE_DIRECTIONS[UE5Bone.NECK_01]
            neck_rot = quaternion_from_two_vectors(rest_dir, neck_dir)
            
            result[UE5Bone.NECK_01] = BoneTransform(
                bone=UE5Bone.NECK_01,
                rotation=neck_rot,
                local_rotation=neck_rot
            )
            result[UE5Bone.HEAD] = BoneTransform(
                bone=UE5Bone.HEAD,
                rotation=neck_rot,
                local_rotation=np.array([1, 0, 0, 0], dtype=np.float32)
            )
        
        return result
    
    def _solve_arm(self, pose: Pose3D, side: str) -> Dict[UE5Bone, BoneTransform]:
        """Solve arm chain rotations."""
        result = {}
        
        shoulder = pose.joints.get(f"{side}_shoulder")
        elbow = pose.joints.get(f"{side}_elbow")
        wrist = pose.joints.get(f"{side}_wrist")
        
        if side == "left":
            upperarm_bone = UE5Bone.UPPERARM_L
            lowerarm_bone = UE5Bone.LOWERARM_L
            hand_bone = UE5Bone.HAND_L
            clavicle_bone = UE5Bone.CLAVICLE_L
        else:
            upperarm_bone = UE5Bone.UPPERARM_R
            lowerarm_bone = UE5Bone.LOWERARM_R
            hand_bone = UE5Bone.HAND_R
            clavicle_bone = UE5Bone.CLAVICLE_R
        
        identity = np.array([1, 0, 0, 0], dtype=np.float32)
        result[clavicle_bone] = BoneTransform(
            bone=clavicle_bone,
            rotation=identity,
            local_rotation=identity
        )
        
        if shoulder and elbow:
            upper_dir = normalize(elbow.position - shoulder.position)
            rest_dir = UE5_REST_POSE_DIRECTIONS[upperarm_bone]
            upper_rot = quaternion_from_two_vectors(rest_dir, upper_dir)
            
            result[upperarm_bone] = BoneTransform(
                bone=upperarm_bone,
                rotation=upper_rot,
                local_rotation=upper_rot
            )
        else:
            result[upperarm_bone] = BoneTransform(
                bone=upperarm_bone,
                rotation=identity,
                local_rotation=identity
            )
        
        if elbow and wrist:
            lower_dir = normalize(wrist.position - elbow.position)
            rest_dir = UE5_REST_POSE_DIRECTIONS[lowerarm_bone]
            lower_rot = quaternion_from_two_vectors(rest_dir, lower_dir)
            
            if upperarm_bone in result:
                parent_rot = result[upperarm_bone].rotation
                local_rot = quaternion_multiply(
                    quaternion_inverse(parent_rot),
                    lower_rot
                )
            else:
                local_rot = lower_rot
            
            result[lowerarm_bone] = BoneTransform(
                bone=lowerarm_bone,
                rotation=lower_rot,
                local_rotation=local_rot
            )
        else:
            result[lowerarm_bone] = BoneTransform(
                bone=lowerarm_bone,
                rotation=identity,
                local_rotation=identity
            )
        
        result[hand_bone] = BoneTransform(
            bone=hand_bone,
            rotation=identity,
            local_rotation=identity
        )
        
        return result
    
    def _solve_leg(self, pose: Pose3D, side: str) -> Dict[UE5Bone, BoneTransform]:
        """Solve leg chain rotations."""
        result = {}
        
        hip = pose.joints.get(f"{side}_hip")
        knee = pose.joints.get(f"{side}_knee")
        ankle = pose.joints.get(f"{side}_ankle")
        foot_index = pose.joints.get(f"{side}_foot_index")
        
        if side == "left":
            thigh_bone = UE5Bone.THIGH_L
            calf_bone = UE5Bone.CALF_L
            foot_bone = UE5Bone.FOOT_L
            ball_bone = UE5Bone.BALL_L
        else:
            thigh_bone = UE5Bone.THIGH_R
            calf_bone = UE5Bone.CALF_R
            foot_bone = UE5Bone.FOOT_R
            ball_bone = UE5Bone.BALL_R
        
        identity = np.array([1, 0, 0, 0], dtype=np.float32)
        
        if hip and knee:
            thigh_dir = normalize(knee.position - hip.position)
            rest_dir = UE5_REST_POSE_DIRECTIONS[thigh_bone]
            thigh_rot = quaternion_from_two_vectors(rest_dir, thigh_dir)
            
            result[thigh_bone] = BoneTransform(
                bone=thigh_bone,
                rotation=thigh_rot,
                local_rotation=thigh_rot
            )
        else:
            result[thigh_bone] = BoneTransform(
                bone=thigh_bone,
                rotation=identity,
                local_rotation=identity
            )
        
        if knee and ankle:
            calf_dir = normalize(ankle.position - knee.position)
            rest_dir = UE5_REST_POSE_DIRECTIONS[calf_bone]
            calf_rot = quaternion_from_two_vectors(rest_dir, calf_dir)
            
            if thigh_bone in result:
                parent_rot = result[thigh_bone].rotation
                local_rot = quaternion_multiply(
                    quaternion_inverse(parent_rot),
                    calf_rot
                )
            else:
                local_rot = calf_rot
            
            result[calf_bone] = BoneTransform(
                bone=calf_bone,
                rotation=calf_rot,
                local_rotation=local_rot
            )
        else:
            result[calf_bone] = BoneTransform(
                bone=calf_bone,
                rotation=identity,
                local_rotation=identity
            )
        
        if ankle and foot_index:
            foot_dir = normalize(foot_index.position - ankle.position)
            rest_dir = UE5_REST_POSE_DIRECTIONS[foot_bone]
            foot_rot = quaternion_from_two_vectors(rest_dir, foot_dir)
            
            if calf_bone in result:
                parent_rot = result[calf_bone].rotation
                local_rot = quaternion_multiply(
                    quaternion_inverse(parent_rot),
                    foot_rot
                )
            else:
                local_rot = foot_rot
            
            result[foot_bone] = BoneTransform(
                bone=foot_bone,
                rotation=foot_rot,
                local_rotation=local_rot
            )
        else:
            result[foot_bone] = BoneTransform(
                bone=foot_bone,
                rotation=identity,
                local_rotation=identity
            )
        
        result[ball_bone] = BoneTransform(
            bone=ball_bone,
            rotation=identity,
            local_rotation=identity
        )
        
        return result
    
    def get_animation_data(self) -> List[SkeletonPose]:
        """Get all skeleton poses for animation export."""
        return self._pose_history.copy()
    
    def reset(self) -> None:
        """Reset solver state."""
        self._pose_history.clear()
        self._frame_count = 0
    
    @property
    def history(self) -> List[SkeletonPose]:
        return self._pose_history
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
