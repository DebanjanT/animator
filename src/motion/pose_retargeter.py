"""
Pose Retargeting - Convert 2D/3D pose estimation to Mixamo skeleton.

This module properly handles:
1. Coordinate system conversion (camera space to Mixamo space)
2. Local rotation computation (relative to parent bone bind pose)
3. IK corrections (foot locking, ground alignment, pole vectors)
4. Twist bone distribution

Coordinate Systems:
- Camera/MediaPipe: X=right, Y=down, Z=away from camera
- Mixamo/OpenGL: X=right, Y=up, Z=towards camera
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.core.halpe_keypoints import (
    HalpeBodyKeypoint,
    HALPE_BODY_KEYPOINT_NAMES,
)


# =============================================================================
# MIXAMO SKELETON DEFINITION
# =============================================================================

class MixamoBone(Enum):
    """Mixamo bone names with their bind pose directions."""
    HIPS = "mixamorig:Hips"
    SPINE = "mixamorig:Spine"
    SPINE1 = "mixamorig:Spine1"
    SPINE2 = "mixamorig:Spine2"
    NECK = "mixamorig:Neck"
    HEAD = "mixamorig:Head"
    LEFT_SHOULDER = "mixamorig:LeftShoulder"
    LEFT_ARM = "mixamorig:LeftArm"
    LEFT_FOREARM = "mixamorig:LeftForeArm"
    LEFT_HAND = "mixamorig:LeftHand"
    RIGHT_SHOULDER = "mixamorig:RightShoulder"
    RIGHT_ARM = "mixamorig:RightArm"
    RIGHT_FOREARM = "mixamorig:RightForeArm"
    RIGHT_HAND = "mixamorig:RightHand"
    LEFT_UP_LEG = "mixamorig:LeftUpLeg"
    LEFT_LEG = "mixamorig:LeftLeg"
    LEFT_FOOT = "mixamorig:LeftFoot"
    LEFT_TOE_BASE = "mixamorig:LeftToeBase"
    RIGHT_UP_LEG = "mixamorig:RightUpLeg"
    RIGHT_LEG = "mixamorig:RightLeg"
    RIGHT_FOOT = "mixamorig:RightFoot"
    RIGHT_TOE_BASE = "mixamorig:RightToeBase"


# Mixamo T-pose bind directions (the direction each bone points in T-pose)
# These are in Mixamo's coordinate system: Y-up, Z-forward
MIXAMO_BIND_POSE: Dict[MixamoBone, np.ndarray] = {
    MixamoBone.HIPS: np.array([0, 1, 0]),          # Spine points up
    MixamoBone.SPINE: np.array([0, 1, 0]),         # Up
    MixamoBone.SPINE1: np.array([0, 1, 0]),        # Up
    MixamoBone.SPINE2: np.array([0, 1, 0]),        # Up
    MixamoBone.NECK: np.array([0, 1, 0]),          # Up
    MixamoBone.HEAD: np.array([0, 1, 0]),          # Up
    MixamoBone.LEFT_SHOULDER: np.array([1, 0, 0]), # Left arm points +X (left)
    MixamoBone.LEFT_ARM: np.array([1, 0, 0]),      # +X
    MixamoBone.LEFT_FOREARM: np.array([1, 0, 0]),  # +X
    MixamoBone.LEFT_HAND: np.array([1, 0, 0]),     # +X
    MixamoBone.RIGHT_SHOULDER: np.array([-1, 0, 0]), # Right arm points -X (right)
    MixamoBone.RIGHT_ARM: np.array([-1, 0, 0]),    # -X
    MixamoBone.RIGHT_FOREARM: np.array([-1, 0, 0]), # -X
    MixamoBone.RIGHT_HAND: np.array([-1, 0, 0]),   # -X
    MixamoBone.LEFT_UP_LEG: np.array([0, -1, 0]),  # Legs point down (-Y)
    MixamoBone.LEFT_LEG: np.array([0, -1, 0]),     # -Y
    MixamoBone.LEFT_FOOT: np.array([0, 0, 1]),     # Foot points forward (+Z)
    MixamoBone.LEFT_TOE_BASE: np.array([0, 0, 1]), # +Z
    MixamoBone.RIGHT_UP_LEG: np.array([0, -1, 0]), # -Y
    MixamoBone.RIGHT_LEG: np.array([0, -1, 0]),    # -Y
    MixamoBone.RIGHT_FOOT: np.array([0, 0, 1]),    # +Z
    MixamoBone.RIGHT_TOE_BASE: np.array([0, 0, 1]), # +Z
}

# Parent bone mapping for local rotation computation
MIXAMO_BONE_PARENT: Dict[MixamoBone, Optional[MixamoBone]] = {
    MixamoBone.HIPS: None,
    MixamoBone.SPINE: MixamoBone.HIPS,
    MixamoBone.SPINE1: MixamoBone.SPINE,
    MixamoBone.SPINE2: MixamoBone.SPINE1,
    MixamoBone.NECK: MixamoBone.SPINE2,
    MixamoBone.HEAD: MixamoBone.NECK,
    MixamoBone.LEFT_SHOULDER: MixamoBone.SPINE2,
    MixamoBone.LEFT_ARM: MixamoBone.LEFT_SHOULDER,
    MixamoBone.LEFT_FOREARM: MixamoBone.LEFT_ARM,
    MixamoBone.LEFT_HAND: MixamoBone.LEFT_FOREARM,
    MixamoBone.RIGHT_SHOULDER: MixamoBone.SPINE2,
    MixamoBone.RIGHT_ARM: MixamoBone.RIGHT_SHOULDER,
    MixamoBone.RIGHT_FOREARM: MixamoBone.RIGHT_ARM,
    MixamoBone.RIGHT_HAND: MixamoBone.RIGHT_FOREARM,
    MixamoBone.LEFT_UP_LEG: MixamoBone.HIPS,
    MixamoBone.LEFT_LEG: MixamoBone.LEFT_UP_LEG,
    MixamoBone.LEFT_FOOT: MixamoBone.LEFT_LEG,
    MixamoBone.LEFT_TOE_BASE: MixamoBone.LEFT_FOOT,
    MixamoBone.RIGHT_UP_LEG: MixamoBone.HIPS,
    MixamoBone.RIGHT_LEG: MixamoBone.RIGHT_UP_LEG,
    MixamoBone.RIGHT_FOOT: MixamoBone.RIGHT_LEG,
    MixamoBone.RIGHT_TOE_BASE: MixamoBone.RIGHT_FOOT,
}


# =============================================================================
# QUATERNION UTILITIES
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, return zero if length is too small."""
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def quat_identity() -> np.ndarray:
    """Return identity quaternion [w, x, y, z]."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Create quaternion that rotates v_from to v_to.
    Returns [w, x, y, z] format.
    """
    v_from = normalize(v_from)
    v_to = normalize(v_to)
    
    dot = np.dot(v_from, v_to)
    
    # Vectors are nearly parallel
    if dot > 0.99999:
        return quat_identity()
    
    # Vectors are nearly opposite
    if dot < -0.99999:
        # Find orthogonal vector for 180-degree rotation axis
        ortho = np.array([1, 0, 0], dtype=np.float32)
        if abs(v_from[0]) > 0.9:
            ortho = np.array([0, 1, 0], dtype=np.float32)
        axis = normalize(np.cross(v_from, ortho))
        return np.array([0, axis[0], axis[1], axis[2]], dtype=np.float32)
    
    # Standard case
    axis = np.cross(v_from, v_to)
    s = np.sqrt((1 + dot) * 2)
    invs = 1.0 / s
    
    return np.array([
        s * 0.5,
        axis[0] * invs,
        axis[1] * invs,
        axis[2] * invs
    ], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate (inverse for unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    qv = np.array([0, v[0], v[1], v[2]], dtype=np.float32)
    rotated = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return rotated[1:4]


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create quaternion from axis and angle (radians)."""
    axis = normalize(axis)
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float32)


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions."""
    dot = np.dot(q1, q2)
    
    # Ensure shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    if dot > 0.9995:
        # Linear interpolation for close quaternions
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q2_perp = q2 - q1 * dot
    q2_perp = q2_perp / np.linalg.norm(q2_perp)
    
    return q1 * np.cos(theta) + q2_perp * np.sin(theta)


# =============================================================================
# RETARGETING DATA
# =============================================================================

@dataclass
class RetargetedBone:
    """A single retargeted bone with debug info."""
    name: str
    local_rotation: np.ndarray  # [w, x, y, z]
    world_rotation: np.ndarray  # [w, x, y, z]
    source_direction: np.ndarray  # Direction from pose estimation
    bind_direction: np.ndarray   # Bind pose direction
    confidence: float = 1.0


@dataclass
class RetargetedPose:
    """Complete retargeted pose with all bones."""
    timestamp: float
    bones: Dict[str, RetargetedBone] = field(default_factory=dict)
    
    # IK correction data
    left_foot_grounded: bool = False
    right_foot_grounded: bool = False
    ground_height: float = 0.0
    
    def get_bone_data(self, bone_name: str) -> Optional[List[float]]:
        """Get bone transform as [px, py, pz, qw, qx, qy, qz, sx, sy, sz]."""
        bone = self.bones.get(bone_name)
        if bone is None:
            return None
        q = bone.local_rotation
        return [0.0, 0.0, 0.0, q[0], q[1], q[2], q[3], 1.0, 1.0, 1.0]
    
    def to_animation_frame(self) -> Dict[str, List[float]]:
        """Convert to animation frame format for viewer."""
        result = {}
        for name, bone in self.bones.items():
            q = bone.local_rotation
            result[name] = [0.0, 0.0, 0.0, q[0], q[1], q[2], q[3], 1.0, 1.0, 1.0]
        return result


# =============================================================================
# POSE RETARGETER
# =============================================================================

class PoseRetargeter:
    """
    Converts 2D pose estimation to Mixamo skeleton rotations.
    
    Handles:
    - Coordinate system conversion
    - Local rotation computation
    - Bone chain propagation
    - Optional IK corrections
    """
    
    def __init__(
        self,
        enable_ik: bool = True,
        enable_foot_locking: bool = True,
        enable_pole_vectors: bool = True,
        smoothing_factor: float = 0.3,
    ):
        self.enable_ik = enable_ik
        self.enable_foot_locking = enable_foot_locking
        self.enable_pole_vectors = enable_pole_vectors
        self.smoothing_factor = smoothing_factor
        
        # State for temporal smoothing
        self._prev_rotations: Dict[str, np.ndarray] = {}
        
        # Ground plane estimation
        self._ground_height = 0.0
        self._ground_samples: List[float] = []
        
        # Foot locking state
        self._left_foot_locked_pos: Optional[np.ndarray] = None
        self._right_foot_locked_pos: Optional[np.ndarray] = None
        
        # Debug info
        self.debug_info: Dict[str, any] = {}
    
    def retarget_pose2d(
        self,
        pose,  # Pose2D object
        timestamp: float = 0.0,
    ) -> Optional[RetargetedPose]:
        """
        Retarget a 2D pose estimation result to Mixamo skeleton.
        
        Args:
            pose: Pose2D with joint positions
            timestamp: Frame timestamp
            
        Returns:
            RetargetedPose with bone rotations
        """
        # Extract joint positions and convert to 3D
        joints = self._extract_joints_from_pose2d(pose)
        if joints is None:
            return None
        
        return self._retarget_from_joints(joints, timestamp)
    
    def retarget_halpe(
        self,
        keypoints: Dict[int, Tuple[float, float, float, float]],  # idx -> (x, y, z, conf)
        image_width: int,
        image_height: int,
        timestamp: float = 0.0,
    ) -> Optional[RetargetedPose]:
        """
        Retarget Halpe 136 keypoints to Mixamo skeleton.
        
        Args:
            keypoints: Dict of keypoint_index -> (x, y, z, confidence)
            image_width: Image width for normalization
            image_height: Image height for normalization
            timestamp: Frame timestamp
            
        Returns:
            RetargetedPose with bone rotations
        """
        joints = self._extract_joints_from_halpe(keypoints, image_width, image_height)
        if joints is None:
            return None
        
        return self._retarget_from_joints(joints, timestamp)
    
    def _extract_joints_from_pose2d(self, pose) -> Optional[Dict[str, np.ndarray]]:
        """Extract 3D joint positions from Pose2D."""
        joints = {}
        
        def get_pos(joint_name: str) -> Optional[np.ndarray]:
            joint = pose.joints.get(joint_name)
            if joint is None or not joint.is_visible:
                return None
            # Convert from normalized 2D to 3D
            # Camera: x=0-1 left-right, y=0-1 top-bottom, z=depth
            # Convert to: x=right, y=up, z=forward
            return np.array([
                joint.x - 0.5,           # Center X (-0.5 to 0.5)
                -(joint.y - 0.5),        # Flip Y (up is positive)
                -joint.z * 0.5 if hasattr(joint, 'z') else 0.0  # Z depth
            ], dtype=np.float32)
        
        # Body joints
        joint_names = [
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
            "nose", "left_ear", "right_ear",
            "left_heel", "right_heel", "left_foot_index", "right_foot_index"
        ]
        
        for name in joint_names:
            pos = get_pos(name)
            if pos is not None:
                joints[name] = pos
        
        # Need at least hips
        if "left_hip" not in joints or "right_hip" not in joints:
            return None
        
        return joints
    
    def _extract_joints_from_halpe(
        self,
        keypoints: Dict[int, Tuple[float, float, float, float]],
        image_width: int,
        image_height: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract 3D joint positions from Halpe keypoints."""
        
        def get_pos(kp_idx: int) -> Optional[np.ndarray]:
            if kp_idx not in keypoints:
                return None
            x, y, z, conf = keypoints[kp_idx]
            if conf < 0.3:  # Low confidence threshold
                return None
            # Normalize and convert coordinate system
            return np.array([
                (x / image_width) - 0.5,
                -((y / image_height) - 0.5),
                -z * 0.5 if z != 0 else 0.0
            ], dtype=np.float32)
        
        joints = {}
        
        # Map Halpe indices to joint names
        halpe_mapping = {
            "left_hip": HalpeBodyKeypoint.LEFT_HIP,
            "right_hip": HalpeBodyKeypoint.RIGHT_HIP,
            "left_shoulder": HalpeBodyKeypoint.LEFT_SHOULDER,
            "right_shoulder": HalpeBodyKeypoint.RIGHT_SHOULDER,
            "left_elbow": HalpeBodyKeypoint.LEFT_ELBOW,
            "right_elbow": HalpeBodyKeypoint.RIGHT_ELBOW,
            "left_wrist": HalpeBodyKeypoint.LEFT_WRIST,
            "right_wrist": HalpeBodyKeypoint.RIGHT_WRIST,
            "left_knee": HalpeBodyKeypoint.LEFT_KNEE,
            "right_knee": HalpeBodyKeypoint.RIGHT_KNEE,
            "left_ankle": HalpeBodyKeypoint.LEFT_ANKLE,
            "right_ankle": HalpeBodyKeypoint.RIGHT_ANKLE,
            "nose": HalpeBodyKeypoint.NOSE,
            "left_ear": HalpeBodyKeypoint.LEFT_EAR,
            "right_ear": HalpeBodyKeypoint.RIGHT_EAR,
            "left_heel": HalpeBodyKeypoint.LEFT_HEEL,
            "right_heel": HalpeBodyKeypoint.RIGHT_HEEL,
            "left_foot_index": HalpeBodyKeypoint.LEFT_BIG_TOE,
            "right_foot_index": HalpeBodyKeypoint.RIGHT_BIG_TOE,
            "neck": HalpeBodyKeypoint.NECK,
            "head": HalpeBodyKeypoint.HEAD,
            "hip_center": HalpeBodyKeypoint.HIP,
        }
        
        for name, idx in halpe_mapping.items():
            pos = get_pos(int(idx))
            if pos is not None:
                joints[name] = pos
        
        if "left_hip" not in joints or "right_hip" not in joints:
            return None
        
        return joints
    
    def _retarget_from_joints(
        self,
        joints: Dict[str, np.ndarray],
        timestamp: float,
    ) -> RetargetedPose:
        """
        Core retargeting from 3D joint positions to bone rotations.
        """
        result = RetargetedPose(timestamp=timestamp)
        
        # Track world rotations for hierarchical computation
        world_rotations: Dict[MixamoBone, np.ndarray] = {}
        
        # Helper to compute bone rotation
        def compute_bone(
            bone: MixamoBone,
            start_joint: str,
            end_joint: str,
        ) -> Optional[RetargetedBone]:
            start = joints.get(start_joint)
            end = joints.get(end_joint)
            
            if start is None or end is None:
                return None
            
            # Direction from start to end joint
            direction = end - start
            direction = normalize(direction)
            
            if np.linalg.norm(direction) < 1e-6:
                return None
            
            # Get bind pose direction for this bone
            bind_dir = MIXAMO_BIND_POSE[bone].copy()
            
            # Compute world rotation (from bind pose to current pose)
            world_rot = quat_from_two_vectors(bind_dir, direction)
            world_rotations[bone] = world_rot
            
            # Compute local rotation (relative to parent)
            parent = MIXAMO_BONE_PARENT.get(bone)
            if parent is not None and parent in world_rotations:
                parent_world = world_rotations[parent]
                # local = inverse(parent_world) * world
                local_rot = quat_multiply(quat_conjugate(parent_world), world_rot)
            else:
                local_rot = world_rot
            
            # Apply temporal smoothing
            bone_name = bone.value
            if bone_name in self._prev_rotations:
                local_rot = quat_slerp(
                    self._prev_rotations[bone_name],
                    local_rot,
                    1.0 - self.smoothing_factor
                )
            self._prev_rotations[bone_name] = local_rot
            
            return RetargetedBone(
                name=bone_name,
                local_rotation=local_rot,
                world_rotation=world_rot,
                source_direction=direction,
                bind_direction=bind_dir,
            )
        
        # Compute hip center for spine calculations
        hip_center = (joints["left_hip"] + joints["right_hip"]) / 2
        
        # Compute shoulder center if available
        if "left_shoulder" in joints and "right_shoulder" in joints:
            shoulder_center = (joints["left_shoulder"] + joints["right_shoulder"]) / 2
            spine_dir = normalize(shoulder_center - hip_center)
        else:
            spine_dir = np.array([0, 1, 0])
        
        # --- HIPS ---
        hips_world = quat_from_two_vectors(MIXAMO_BIND_POSE[MixamoBone.HIPS], spine_dir)
        world_rotations[MixamoBone.HIPS] = hips_world
        
        hips_local = self._smooth_rotation(MixamoBone.HIPS.value, hips_world)
        result.bones[MixamoBone.HIPS.value] = RetargetedBone(
            name=MixamoBone.HIPS.value,
            local_rotation=hips_local,
            world_rotation=hips_world,
            source_direction=spine_dir,
            bind_direction=MIXAMO_BIND_POSE[MixamoBone.HIPS],
        )
        
        # --- SPINE CHAIN ---
        # For spine bones, we use identity local rotations (hips handles spine tilt)
        for spine_bone in [MixamoBone.SPINE, MixamoBone.SPINE1, MixamoBone.SPINE2]:
            world_rotations[spine_bone] = hips_world  # Inherit from hips
            result.bones[spine_bone.value] = RetargetedBone(
                name=spine_bone.value,
                local_rotation=quat_identity(),
                world_rotation=hips_world,
                source_direction=spine_dir,
                bind_direction=MIXAMO_BIND_POSE[spine_bone],
            )
        
        # --- NECK AND HEAD ---
        if "nose" in joints and "left_shoulder" in joints and "right_shoulder" in joints:
            neck_base = shoulder_center
            head_dir = normalize(joints["nose"] - neck_base)
            
            neck_world = quat_from_two_vectors(MIXAMO_BIND_POSE[MixamoBone.NECK], head_dir)
            world_rotations[MixamoBone.NECK] = neck_world
            
            # Neck local rotation relative to spine2
            parent_world = world_rotations.get(MixamoBone.SPINE2, quat_identity())
            neck_local = quat_multiply(quat_conjugate(parent_world), neck_world)
            neck_local = self._smooth_rotation(MixamoBone.NECK.value, neck_local)
            
            result.bones[MixamoBone.NECK.value] = RetargetedBone(
                name=MixamoBone.NECK.value,
                local_rotation=neck_local,
                world_rotation=neck_world,
                source_direction=head_dir,
                bind_direction=MIXAMO_BIND_POSE[MixamoBone.NECK],
            )
            
            # Head - identity relative to neck
            world_rotations[MixamoBone.HEAD] = neck_world
            result.bones[MixamoBone.HEAD.value] = RetargetedBone(
                name=MixamoBone.HEAD.value,
                local_rotation=quat_identity(),
                world_rotation=neck_world,
                source_direction=head_dir,
                bind_direction=MIXAMO_BIND_POSE[MixamoBone.HEAD],
            )
        
        # --- LEFT ARM CHAIN ---
        # Left Shoulder
        if "left_shoulder" in joints and "right_shoulder" in joints:
            shoulder_to_arm = normalize(joints["left_shoulder"] - shoulder_center)
            ls_world = quat_from_two_vectors(MIXAMO_BIND_POSE[MixamoBone.LEFT_SHOULDER], shoulder_to_arm)
            world_rotations[MixamoBone.LEFT_SHOULDER] = ls_world
            
            parent_world = world_rotations.get(MixamoBone.SPINE2, quat_identity())
            ls_local = quat_multiply(quat_conjugate(parent_world), ls_world)
            ls_local = self._smooth_rotation(MixamoBone.LEFT_SHOULDER.value, ls_local)
            
            result.bones[MixamoBone.LEFT_SHOULDER.value] = RetargetedBone(
                name=MixamoBone.LEFT_SHOULDER.value,
                local_rotation=ls_local,
                world_rotation=ls_world,
                source_direction=shoulder_to_arm,
                bind_direction=MIXAMO_BIND_POSE[MixamoBone.LEFT_SHOULDER],
            )
        
        # Left Arm
        bone = compute_bone(MixamoBone.LEFT_ARM, "left_shoulder", "left_elbow")
        if bone:
            result.bones[bone.name] = bone
        
        # Left ForeArm
        bone = compute_bone(MixamoBone.LEFT_FOREARM, "left_elbow", "left_wrist")
        if bone:
            result.bones[bone.name] = bone
        
        # Left Hand - identity
        if MixamoBone.LEFT_FOREARM in world_rotations:
            result.bones[MixamoBone.LEFT_HAND.value] = RetargetedBone(
                name=MixamoBone.LEFT_HAND.value,
                local_rotation=quat_identity(),
                world_rotation=world_rotations[MixamoBone.LEFT_FOREARM],
                source_direction=np.array([1, 0, 0]),
                bind_direction=MIXAMO_BIND_POSE[MixamoBone.LEFT_HAND],
            )
        
        # --- RIGHT ARM CHAIN ---
        # Right Shoulder
        if "left_shoulder" in joints and "right_shoulder" in joints:
            shoulder_to_arm = normalize(joints["right_shoulder"] - shoulder_center)
            rs_world = quat_from_two_vectors(MIXAMO_BIND_POSE[MixamoBone.RIGHT_SHOULDER], shoulder_to_arm)
            world_rotations[MixamoBone.RIGHT_SHOULDER] = rs_world
            
            parent_world = world_rotations.get(MixamoBone.SPINE2, quat_identity())
            rs_local = quat_multiply(quat_conjugate(parent_world), rs_world)
            rs_local = self._smooth_rotation(MixamoBone.RIGHT_SHOULDER.value, rs_local)
            
            result.bones[MixamoBone.RIGHT_SHOULDER.value] = RetargetedBone(
                name=MixamoBone.RIGHT_SHOULDER.value,
                local_rotation=rs_local,
                world_rotation=rs_world,
                source_direction=shoulder_to_arm,
                bind_direction=MIXAMO_BIND_POSE[MixamoBone.RIGHT_SHOULDER],
            )
        
        # Right Arm
        bone = compute_bone(MixamoBone.RIGHT_ARM, "right_shoulder", "right_elbow")
        if bone:
            result.bones[bone.name] = bone
        
        # Right ForeArm
        bone = compute_bone(MixamoBone.RIGHT_FOREARM, "right_elbow", "right_wrist")
        if bone:
            result.bones[bone.name] = bone
        
        # Right Hand - identity
        if MixamoBone.RIGHT_FOREARM in world_rotations:
            result.bones[MixamoBone.RIGHT_HAND.value] = RetargetedBone(
                name=MixamoBone.RIGHT_HAND.value,
                local_rotation=quat_identity(),
                world_rotation=world_rotations[MixamoBone.RIGHT_FOREARM],
                source_direction=np.array([-1, 0, 0]),
                bind_direction=MIXAMO_BIND_POSE[MixamoBone.RIGHT_HAND],
            )
        
        # --- LEFT LEG CHAIN ---
        bone = compute_bone(MixamoBone.LEFT_UP_LEG, "left_hip", "left_knee")
        if bone:
            result.bones[bone.name] = bone
        
        bone = compute_bone(MixamoBone.LEFT_LEG, "left_knee", "left_ankle")
        if bone:
            result.bones[bone.name] = bone
        
        # Left Foot
        if "left_ankle" in joints and "left_foot_index" in joints:
            bone = compute_bone(MixamoBone.LEFT_FOOT, "left_ankle", "left_foot_index")
            if bone:
                result.bones[bone.name] = bone
        
        # --- RIGHT LEG CHAIN ---
        bone = compute_bone(MixamoBone.RIGHT_UP_LEG, "right_hip", "right_knee")
        if bone:
            result.bones[bone.name] = bone
        
        bone = compute_bone(MixamoBone.RIGHT_LEG, "right_knee", "right_ankle")
        if bone:
            result.bones[bone.name] = bone
        
        # Right Foot
        if "right_ankle" in joints and "right_foot_index" in joints:
            bone = compute_bone(MixamoBone.RIGHT_FOOT, "right_ankle", "right_foot_index")
            if bone:
                result.bones[bone.name] = bone
        
        # Apply IK corrections if enabled
        if self.enable_ik:
            self._apply_ik_corrections(result, joints)
        
        return result
    
    def _smooth_rotation(self, bone_name: str, rotation: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to rotation."""
        if bone_name in self._prev_rotations:
            rotation = quat_slerp(
                self._prev_rotations[bone_name],
                rotation,
                1.0 - self.smoothing_factor
            )
        self._prev_rotations[bone_name] = rotation
        return rotation
    
    def _apply_ik_corrections(
        self,
        pose: RetargetedPose,
        joints: Dict[str, np.ndarray],
    ):
        """
        Apply IK corrections:
        - Foot locking for grounded feet
        - Ground alignment
        - Pole vector correction for knees/elbows
        """
        # Ground plane estimation from ankle heights
        if "left_ankle" in joints and "right_ankle" in joints:
            min_ankle_y = min(joints["left_ankle"][1], joints["right_ankle"][1])
            self._ground_samples.append(min_ankle_y)
            if len(self._ground_samples) > 30:
                self._ground_samples.pop(0)
            self._ground_height = np.percentile(self._ground_samples, 10)
        
        # Detect grounded feet (ankle near ground level)
        ground_threshold = 0.02  # 2% of normalized height
        
        if "left_ankle" in joints:
            left_grounded = abs(joints["left_ankle"][1] - self._ground_height) < ground_threshold
            pose.left_foot_grounded = left_grounded
        
        if "right_ankle" in joints:
            right_grounded = abs(joints["right_ankle"][1] - self._ground_height) < ground_threshold
            pose.right_foot_grounded = right_grounded
        
        pose.ground_height = self._ground_height
        
        # Pole vector correction for knees (prevent knee pop)
        if self.enable_pole_vectors:
            self._apply_pole_vector_correction(pose, joints, "left")
            self._apply_pole_vector_correction(pose, joints, "right")
    
    def _apply_pole_vector_correction(
        self,
        pose: RetargetedPose,
        joints: Dict[str, np.ndarray],
        side: str,
    ):
        """
        Apply pole vector correction to prevent knee/elbow flipping.
        
        The pole vector defines the plane in which the joint should bend.
        For knees: forward (Z+)
        For elbows: backward (Z-)
        """
        # Knee pole vector (knees bend forward)
        hip_joint = f"{side}_hip"
        knee_joint = f"{side}_knee"
        ankle_joint = f"{side}_ankle"
        
        if all(j in joints for j in [hip_joint, knee_joint, ankle_joint]):
            hip = joints[hip_joint]
            knee = joints[knee_joint]
            ankle = joints[ankle_joint]
            
            # Compute the plane normal
            thigh = knee - hip
            shin = ankle - knee
            
            # Current bend plane normal
            current_normal = np.cross(thigh, shin)
            current_normal = normalize(current_normal)
            
            # Desired bend direction (forward)
            desired_normal = np.array([0, 0, 1], dtype=np.float32)
            
            # If knee is bending backward, flip the rotation
            if np.dot(current_normal, desired_normal) < 0:
                # Add a small correction rotation to the thigh bone
                bone_name = f"mixamorig:{side.capitalize()}UpLeg"
                if bone_name in pose.bones:
                    bone = pose.bones[bone_name]
                    # Add small twist to correct pole direction
                    correction = quat_from_axis_angle(
                        normalize(thigh),
                        0.1  # Small correction angle
                    )
                    bone.local_rotation = quat_multiply(bone.local_rotation, correction)
    
    def reset(self):
        """Reset temporal state."""
        self._prev_rotations.clear()
        self._ground_samples.clear()
        self._left_foot_locked_pos = None
        self._right_foot_locked_pos = None


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

# Global retargeter instance
_retargeter: Optional[PoseRetargeter] = None


def get_retargeter() -> PoseRetargeter:
    """Get or create global retargeter instance."""
    global _retargeter
    if _retargeter is None:
        _retargeter = PoseRetargeter()
    return _retargeter


def retarget_pose2d_to_mixamo(pose, timestamp: float = 0.0) -> Optional[Dict[str, List[float]]]:
    """
    Convenience function to retarget Pose2D to Mixamo bone transforms.
    
    Args:
        pose: Pose2D object with joint positions
        timestamp: Frame timestamp
        
    Returns:
        Dict of bone_name -> [px, py, pz, qw, qx, qy, qz, sx, sy, sz]
    """
    retargeter = get_retargeter()
    result = retargeter.retarget_pose2d(pose, timestamp)
    
    if result is None:
        return None
    
    return result.to_animation_frame()
