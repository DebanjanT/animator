"""Inverse Kinematics Solver - Two-bone IK, foot locking, spine smoothing"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

from src.core import get_logger, Config
from src.pose.reconstructor_3d import Pose3D, Joint3D
from src.motion.skeleton_solver import (
    SkeletonPose, BoneTransform, UE5Bone,
    quaternion_from_two_vectors, quaternion_multiply,
    quaternion_inverse, normalize
)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


@dataclass
class IKTarget:
    """Target position for IK solving."""
    position: np.ndarray
    weight: float = 1.0
    is_locked: bool = False


@dataclass
class IKChain:
    """Definition of an IK chain."""
    name: str
    root_bone: UE5Bone
    mid_bone: UE5Bone
    end_bone: UE5Bone
    pole_vector: np.ndarray  # Preferred bend direction
    
    root_length: float = 0.0  # Will be computed
    mid_length: float = 0.0


@dataclass
class FootLockState:
    """State for foot locking."""
    position: np.ndarray
    rotation: np.ndarray
    frame_locked: int
    is_locked: bool = True


class TwoBoneIKSolver:
    """
    Analytical two-bone IK solver.
    
    Solves for joint angles given:
    - Root position (e.g., hip)
    - Target position (e.g., ankle)
    - Pole vector (knee direction hint)
    """
    
    @staticmethod
    def solve(
        root_pos: np.ndarray,
        target_pos: np.ndarray,
        bone1_length: float,
        bone2_length: float,
        pole_vector: np.ndarray,
        epsilon: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve two-bone IK.
        
        Args:
            root_pos: Position of root joint (hip/shoulder)
            target_pos: Target position for end effector
            bone1_length: Length of first bone (thigh/upper arm)
            bone2_length: Length of second bone (calf/forearm)
            pole_vector: Direction hint for middle joint bend
        
        Returns:
            Tuple of (mid_position, bone1_rotation, bone2_rotation)
        """
        to_target = target_pos - root_pos
        target_dist = np.linalg.norm(to_target)
        
        if target_dist < epsilon:
            mid_pos = root_pos + pole_vector * bone1_length
            return mid_pos, np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])
        
        max_reach = bone1_length + bone2_length
        min_reach = abs(bone1_length - bone2_length)
        
        if target_dist >= max_reach:
            direction = normalize(to_target)
            mid_pos = root_pos + direction * bone1_length
            
            rest_dir = np.array([0, -1, 0], dtype=np.float32)
            bone1_rot = quaternion_from_two_vectors(rest_dir, direction)
            bone2_rot = np.array([1, 0, 0, 0], dtype=np.float32)
            
            return mid_pos, bone1_rot, bone2_rot
        
        if target_dist <= min_reach:
            target_dist = min_reach + epsilon
        
        cos_angle1 = (bone1_length**2 + target_dist**2 - bone2_length**2) / (2 * bone1_length * target_dist)
        cos_angle1 = clamp(cos_angle1, -1.0, 1.0)
        angle1 = np.arccos(cos_angle1)
        
        target_dir = normalize(to_target)
        
        pole_projected = pole_vector - np.dot(pole_vector, target_dir) * target_dir
        pole_norm = np.linalg.norm(pole_projected)
        
        if pole_norm < epsilon:
            up = np.array([0, 1, 0], dtype=np.float32)
            if abs(np.dot(target_dir, up)) > 0.9:
                up = np.array([1, 0, 0], dtype=np.float32)
            pole_projected = np.cross(target_dir, up)
        
        pole_projected = normalize(pole_projected)
        
        mid_dir = (
            target_dir * np.cos(angle1) +
            pole_projected * np.sin(angle1)
        )
        mid_pos = root_pos + mid_dir * bone1_length
        
        rest_dir = np.array([0, -1, 0], dtype=np.float32)
        bone1_rot = quaternion_from_two_vectors(rest_dir, mid_dir)
        
        end_dir = normalize(target_pos - mid_pos)
        bone2_rot = quaternion_from_two_vectors(rest_dir, end_dir)
        
        bone2_local = quaternion_multiply(quaternion_inverse(bone1_rot), bone2_rot)
        
        return mid_pos, bone1_rot, bone2_local


class IKSolver:
    """
    Complete IK solver with foot locking and constraints.
    
    Features:
    - Two-bone IK for legs and arms
    - Foot locking when grounded
    - Spine smoothing
    - Constraint enforcement
    """
    
    LEG_CHAINS = {
        "left": IKChain(
            name="left_leg",
            root_bone=UE5Bone.THIGH_L,
            mid_bone=UE5Bone.CALF_L,
            end_bone=UE5Bone.FOOT_L,
            pole_vector=np.array([0, 0, 1], dtype=np.float32)
        ),
        "right": IKChain(
            name="right_leg",
            root_bone=UE5Bone.THIGH_R,
            mid_bone=UE5Bone.CALF_R,
            end_bone=UE5Bone.FOOT_R,
            pole_vector=np.array([0, 0, 1], dtype=np.float32)
        )
    }
    
    ARM_CHAINS = {
        "left": IKChain(
            name="left_arm",
            root_bone=UE5Bone.UPPERARM_L,
            mid_bone=UE5Bone.LOWERARM_L,
            end_bone=UE5Bone.HAND_L,
            pole_vector=np.array([0, 0, -1], dtype=np.float32)
        ),
        "right": IKChain(
            name="right_arm",
            root_bone=UE5Bone.UPPERARM_R,
            mid_bone=UE5Bone.LOWERARM_R,
            end_bone=UE5Bone.HAND_R,
            pole_vector=np.array([0, 0, -1], dtype=np.float32)
        )
    }
    
    DEFAULT_BONE_LENGTHS = {
        "thigh": 0.42,
        "calf": 0.40,
        "upperarm": 0.28,
        "lowerarm": 0.25,
    }
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("ik.solver")
        self.config = config or Config()
        
        ik_config = self.config.ik
        
        self._enabled = ik_config.get("enabled", True)
        self._leg_ik = ik_config.get("leg_ik", True)
        self._foot_locking = ik_config.get("foot_locking", True)
        self._spine_smoothing = ik_config.get("spine_smoothing", True)
        self._iterations = ik_config.get("iterations", 10)
        
        self._left_foot_lock: Optional[FootLockState] = None
        self._right_foot_lock: Optional[FootLockState] = None
        
        self._foot_lock_threshold = 0.02
        self._foot_unlock_threshold = 0.05
        
        self._spine_history: deque = deque(maxlen=5)
        
        self._frame_count = 0
        
        self.logger.info(
            f"Initialized IK solver (leg_ik={self._leg_ik}, "
            f"foot_locking={self._foot_locking})"
        )
    
    def process(
        self,
        skeleton_pose: SkeletonPose,
        pose_3d: Pose3D,
        left_grounded: bool = False,
        right_grounded: bool = False
    ) -> SkeletonPose:
        """
        Apply IK and constraints to skeleton pose.
        
        Args:
            skeleton_pose: Input skeleton pose
            pose_3d: Original 3D pose for reference
            left_grounded: Whether left foot is on ground
            right_grounded: Whether right foot is on ground
        
        Returns:
            Modified skeleton pose with IK applied
        """
        if not self._enabled:
            return skeleton_pose
        
        new_transforms = dict(skeleton_pose.bone_transforms)
        
        if self._leg_ik:
            left_leg_transforms = self._solve_leg_ik(
                pose_3d, "left", left_grounded
            )
            new_transforms.update(left_leg_transforms)
            
            right_leg_transforms = self._solve_leg_ik(
                pose_3d, "right", right_grounded
            )
            new_transforms.update(right_leg_transforms)
        
        if self._spine_smoothing:
            spine_transforms = self._smooth_spine(new_transforms)
            new_transforms.update(spine_transforms)
        
        self._frame_count += 1
        
        return SkeletonPose(
            frame_number=skeleton_pose.frame_number,
            timestamp=skeleton_pose.timestamp,
            root_position=skeleton_pose.root_position,
            root_rotation=skeleton_pose.root_rotation,
            bone_transforms=new_transforms
        )
    
    def _solve_leg_ik(
        self,
        pose_3d: Pose3D,
        side: str,
        is_grounded: bool
    ) -> Dict[UE5Bone, BoneTransform]:
        """Solve IK for one leg."""
        result = {}
        
        chain = self.LEG_CHAINS[side]
        
        hip = pose_3d.joints.get(f"{side}_hip")
        knee = pose_3d.joints.get(f"{side}_knee")
        ankle = pose_3d.joints.get(f"{side}_ankle")
        
        if not all([hip, knee, ankle]):
            return result
        
        hip_pos = hip.position
        target_pos = ankle.position.copy()
        
        if self._foot_locking and is_grounded:
            target_pos = self._apply_foot_lock(side, target_pos)
        
        thigh_length = self.DEFAULT_BONE_LENGTHS["thigh"]
        calf_length = self.DEFAULT_BONE_LENGTHS["calf"]
        
        actual_thigh = np.linalg.norm(knee.position - hip.position)
        actual_calf = np.linalg.norm(ankle.position - knee.position)
        if actual_thigh > 0.1 and actual_calf > 0.1:
            thigh_length = actual_thigh
            calf_length = actual_calf
        
        knee_hint = knee.position - hip_pos
        knee_hint[1] = 0
        if np.linalg.norm(knee_hint) < 0.01:
            knee_hint = chain.pole_vector
        else:
            knee_hint = normalize(knee_hint)
        
        mid_pos, thigh_rot, calf_rot = TwoBoneIKSolver.solve(
            hip_pos,
            target_pos,
            thigh_length,
            calf_length,
            knee_hint
        )
        
        result[chain.root_bone] = BoneTransform(
            bone=chain.root_bone,
            rotation=thigh_rot,
            local_rotation=thigh_rot
        )
        
        result[chain.mid_bone] = BoneTransform(
            bone=chain.mid_bone,
            rotation=quaternion_multiply(thigh_rot, calf_rot),
            local_rotation=calf_rot
        )
        
        return result
    
    def _apply_foot_lock(
        self,
        side: str,
        current_pos: np.ndarray
    ) -> np.ndarray:
        """Apply foot locking to prevent sliding."""
        if side == "left":
            lock_state = self._left_foot_lock
        else:
            lock_state = self._right_foot_lock
        
        if lock_state is None or not lock_state.is_locked:
            new_lock = FootLockState(
                position=current_pos.copy(),
                rotation=np.array([1, 0, 0, 0], dtype=np.float32),
                frame_locked=self._frame_count,
                is_locked=True
            )
            if side == "left":
                self._left_foot_lock = new_lock
            else:
                self._right_foot_lock = new_lock
            return current_pos
        
        locked_pos = lock_state.position
        distance = np.linalg.norm(current_pos - locked_pos)
        
        if distance > self._foot_unlock_threshold:
            lock_state.is_locked = False
            return current_pos
        
        blend = 0.8
        return locked_pos * blend + current_pos * (1 - blend)
    
    def unlock_foot(self, side: str) -> None:
        """Manually unlock a foot."""
        if side == "left":
            if self._left_foot_lock:
                self._left_foot_lock.is_locked = False
        else:
            if self._right_foot_lock:
                self._right_foot_lock.is_locked = False
    
    def _smooth_spine(
        self,
        transforms: Dict[UE5Bone, BoneTransform]
    ) -> Dict[UE5Bone, BoneTransform]:
        """Apply temporal smoothing to spine bones."""
        result = {}
        
        spine_bones = [UE5Bone.SPINE_01, UE5Bone.SPINE_02, UE5Bone.SPINE_03]
        
        current_rots = []
        for bone in spine_bones:
            if bone in transforms:
                current_rots.append(transforms[bone].rotation)
        
        if not current_rots:
            return result
        
        avg_rot = np.mean(current_rots, axis=0)
        avg_rot = avg_rot / np.linalg.norm(avg_rot)
        
        self._spine_history.append(avg_rot)
        
        if len(self._spine_history) > 1:
            smoothed = np.mean(list(self._spine_history), axis=0)
            smoothed = smoothed / np.linalg.norm(smoothed)
        else:
            smoothed = avg_rot
        
        for bone in spine_bones:
            if bone in transforms:
                original = transforms[bone]
                blend = 0.7
                blended_rot = original.rotation * (1 - blend) + smoothed * blend
                blended_rot = blended_rot / np.linalg.norm(blended_rot)
                
                result[bone] = BoneTransform(
                    bone=bone,
                    rotation=blended_rot.astype(np.float32),
                    local_rotation=blended_rot.astype(np.float32)
                )
        
        return result
    
    def solve_arm_ik(
        self,
        pose_3d: Pose3D,
        side: str,
        target_pos: Optional[np.ndarray] = None
    ) -> Dict[UE5Bone, BoneTransform]:
        """
        Solve IK for one arm (optional, for hand tracking).
        
        Args:
            pose_3d: 3D pose
            side: "left" or "right"
            target_pos: Optional override target position
        
        Returns:
            Bone transforms for arm
        """
        result = {}
        
        chain = self.ARM_CHAINS[side]
        
        shoulder = pose_3d.joints.get(f"{side}_shoulder")
        elbow = pose_3d.joints.get(f"{side}_elbow")
        wrist = pose_3d.joints.get(f"{side}_wrist")
        
        if not all([shoulder, elbow, wrist]):
            return result
        
        shoulder_pos = shoulder.position
        if target_pos is None:
            target_pos = wrist.position
        
        upper_length = self.DEFAULT_BONE_LENGTHS["upperarm"]
        lower_length = self.DEFAULT_BONE_LENGTHS["lowerarm"]
        
        elbow_hint = elbow.position - shoulder_pos
        if np.linalg.norm(elbow_hint) < 0.01:
            elbow_hint = chain.pole_vector
        else:
            elbow_hint = normalize(elbow_hint)
        
        mid_pos, upper_rot, lower_rot = TwoBoneIKSolver.solve(
            shoulder_pos,
            target_pos,
            upper_length,
            lower_length,
            elbow_hint
        )
        
        result[chain.root_bone] = BoneTransform(
            bone=chain.root_bone,
            rotation=upper_rot,
            local_rotation=upper_rot
        )
        
        result[chain.mid_bone] = BoneTransform(
            bone=chain.mid_bone,
            rotation=quaternion_multiply(upper_rot, lower_rot),
            local_rotation=lower_rot
        )
        
        return result
    
    def reset(self) -> None:
        """Reset IK solver state."""
        self._left_foot_lock = None
        self._right_foot_lock = None
        self._spine_history.clear()
        self._frame_count = 0
    
    @property
    def left_foot_locked(self) -> bool:
        return self._left_foot_lock is not None and self._left_foot_lock.is_locked
    
    @property
    def right_foot_locked(self) -> bool:
        return self._right_foot_lock is not None and self._right_foot_lock.is_locked
