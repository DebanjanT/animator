"""Root Motion Extraction - pelvis projection and foot slide prevention"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
import numpy as np

from src.core import get_logger, Config
from src.pose.reconstructor_3d import Pose3D, Joint3D
from .floor_detector import FloorPlane, FloorDetector


@dataclass
class RootTransform:
    """Root bone transform for a single frame."""
    frame_number: int
    timestamp: float
    position: np.ndarray  # (3,) world position
    velocity: np.ndarray  # (3,) velocity in m/s
    facing_direction: np.ndarray  # (3,) forward vector on XZ plane
    is_grounded: bool
    left_foot_grounded: bool
    right_foot_grounded: bool
    
    @property
    def x(self) -> float:
        return float(self.position[0])
    
    @property
    def y(self) -> float:
        return float(self.position[1])
    
    @property
    def z(self) -> float:
        return float(self.position[2])
    
    @property
    def speed(self) -> float:
        """Horizontal speed (XZ plane)."""
        return float(np.linalg.norm([self.velocity[0], self.velocity[2]]))
    
    @property
    def yaw(self) -> float:
        """Facing direction as yaw angle in radians."""
        return float(np.arctan2(self.facing_direction[0], self.facing_direction[2]))


@dataclass
class FootContact:
    """Foot contact state for foot locking."""
    position: np.ndarray
    frame_start: int
    is_locked: bool = True


class RootMotionExtractor:
    """
    Extract root motion from 3D poses.
    
    Features:
    - Pelvis projection onto floor plane
    - Foot slide prevention via foot locking
    - Velocity estimation
    - Support for walking, running, jumping
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("motion.root")
        self.config = config or Config()
        
        root_config = self.config.root_motion
        
        self._extract_root = root_config.get("extract_root", True)
        self._lock_feet_threshold = root_config.get("lock_feet_threshold", 0.02)
        self._pelvis_height_offset = root_config.get("pelvis_height_offset", 0.0)
        
        self._root_history: List[RootTransform] = []
        self._max_history = 300
        
        self._left_foot_contact: Optional[FootContact] = None
        self._right_foot_contact: Optional[FootContact] = None
        
        self._velocity_window: deque = deque(maxlen=5)
        self._prev_position: Optional[np.ndarray] = None
        self._prev_timestamp: float = 0.0
        
        self._frame_count = 0
        
        self.logger.info(
            f"Initialized root motion extractor (lock_threshold={self._lock_feet_threshold}m)"
        )
    
    def process(
        self,
        pose: Pose3D,
        floor: Optional[FloorPlane] = None
    ) -> RootTransform:
        """
        Extract root motion from 3D pose.
        
        Args:
            pose: 3D pose with joint positions
            floor: Optional floor plane for grounding
        
        Returns:
            RootTransform for this frame
        """
        pelvis_pos = pose.pelvis
        if pelvis_pos is None:
            pelvis_pos = np.array([0, 0, 0], dtype=np.float32)
        
        if floor is not None:
            floor_y = floor.get_floor_y(pelvis_pos[0], pelvis_pos[2]) if hasattr(floor, 'get_floor_y') else 0
            root_position = np.array([
                pelvis_pos[0],
                floor_y + self._pelvis_height_offset,
                pelvis_pos[2]
            ], dtype=np.float32)
        else:
            root_position = pelvis_pos.copy()
        
        velocity = self._compute_velocity(root_position, pose.timestamp)
        
        facing = self._compute_facing_direction(pose)
        
        left_grounded, right_grounded = self._update_foot_contacts(pose, floor)
        is_grounded = left_grounded or right_grounded
        
        if is_grounded:
            root_position = self._apply_foot_locking(
                root_position, pose, left_grounded, right_grounded
            )
        
        root_transform = RootTransform(
            frame_number=self._frame_count,
            timestamp=pose.timestamp,
            position=root_position,
            velocity=velocity,
            facing_direction=facing,
            is_grounded=is_grounded,
            left_foot_grounded=left_grounded,
            right_foot_grounded=right_grounded
        )
        
        self._root_history.append(root_transform)
        if len(self._root_history) > self._max_history:
            self._root_history.pop(0)
        
        self._prev_position = root_position.copy()
        self._prev_timestamp = pose.timestamp
        self._frame_count += 1
        
        return root_transform
    
    def _compute_velocity(
        self,
        position: np.ndarray,
        timestamp: float
    ) -> np.ndarray:
        """Compute velocity from position change."""
        if self._prev_position is None:
            return np.zeros(3, dtype=np.float32)
        
        dt = timestamp - self._prev_timestamp
        if dt <= 0:
            return np.zeros(3, dtype=np.float32)
        
        velocity = (position - self._prev_position) / dt
        
        self._velocity_window.append(velocity)
        smoothed = np.mean(list(self._velocity_window), axis=0)
        
        return smoothed.astype(np.float32)
    
    def _compute_facing_direction(self, pose: Pose3D) -> np.ndarray:
        """Compute forward facing direction from shoulders and hips."""
        left_shoulder = pose.joints.get("left_shoulder")
        right_shoulder = pose.joints.get("right_shoulder")
        left_hip = pose.joints.get("left_hip")
        right_hip = pose.joints.get("right_hip")
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return np.array([0, 0, 1], dtype=np.float32)
        
        shoulder_vec = right_shoulder.position - left_shoulder.position
        hip_vec = right_hip.position - left_hip.position
        
        right_vec = (shoulder_vec + hip_vec) / 2
        right_vec[1] = 0
        right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-6)
        
        up = np.array([0, 1, 0], dtype=np.float32)
        forward = np.cross(right_vec, up)
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        
        return forward.astype(np.float32)
    
    def _update_foot_contacts(
        self,
        pose: Pose3D,
        floor: Optional[FloorPlane]
    ) -> Tuple[bool, bool]:
        """Update foot contact states and return grounded status."""
        left_ankle = pose.joints.get("left_ankle")
        right_ankle = pose.joints.get("right_ankle")
        
        left_grounded = False
        right_grounded = False
        
        if left_ankle:
            left_grounded = self._is_foot_grounded(left_ankle, floor)
            
            if left_grounded:
                if self._left_foot_contact is None or not self._left_foot_contact.is_locked:
                    self._left_foot_contact = FootContact(
                        position=left_ankle.position.copy(),
                        frame_start=self._frame_count,
                        is_locked=True
                    )
            else:
                if self._left_foot_contact is not None:
                    self._left_foot_contact.is_locked = False
        
        if right_ankle:
            right_grounded = self._is_foot_grounded(right_ankle, floor)
            
            if right_grounded:
                if self._right_foot_contact is None or not self._right_foot_contact.is_locked:
                    self._right_foot_contact = FootContact(
                        position=right_ankle.position.copy(),
                        frame_start=self._frame_count,
                        is_locked=True
                    )
            else:
                if self._right_foot_contact is not None:
                    self._right_foot_contact.is_locked = False
        
        return left_grounded, right_grounded
    
    def _is_foot_grounded(
        self,
        ankle: Joint3D,
        floor: Optional[FloorPlane]
    ) -> bool:
        """Check if foot is on the ground."""
        if floor is not None:
            distance = abs(floor.distance_to_point(ankle.position))
            return distance < self._lock_feet_threshold
        else:
            return ankle.y < self._lock_feet_threshold
    
    def _apply_foot_locking(
        self,
        root_position: np.ndarray,
        pose: Pose3D,
        left_grounded: bool,
        right_grounded: bool
    ) -> np.ndarray:
        """
        Apply foot locking to prevent foot sliding.
        
        When a foot is grounded, we adjust root motion to keep the foot stationary.
        """
        correction = np.zeros(3, dtype=np.float32)
        weight = 0.0
        
        if left_grounded and self._left_foot_contact is not None:
            left_ankle = pose.joints.get("left_ankle")
            if left_ankle:
                foot_slide = left_ankle.position - self._left_foot_contact.position
                foot_slide[1] = 0
                correction += foot_slide
                weight += 1.0
        
        if right_grounded and self._right_foot_contact is not None:
            right_ankle = pose.joints.get("right_ankle")
            if right_ankle:
                foot_slide = right_ankle.position - self._right_foot_contact.position
                foot_slide[1] = 0
                correction += foot_slide
                weight += 1.0
        
        if weight > 0:
            correction /= weight
            blend = 0.3
            root_position[0] -= correction[0] * blend
            root_position[2] -= correction[2] * blend
        
        return root_position
    
    def get_root_trajectory(self) -> np.ndarray:
        """
        Get root position trajectory.
        
        Returns:
            Array of shape (N, 3) with positions
        """
        if not self._root_history:
            return np.zeros((0, 3), dtype=np.float32)
        
        return np.array([r.position for r in self._root_history], dtype=np.float32)
    
    def get_velocity_trajectory(self) -> np.ndarray:
        """Get velocity trajectory."""
        if not self._root_history:
            return np.zeros((0, 3), dtype=np.float32)
        
        return np.array([r.velocity for r in self._root_history], dtype=np.float32)
    
    def compute_total_distance(self) -> float:
        """Compute total distance traveled."""
        trajectory = self.get_root_trajectory()
        if len(trajectory) < 2:
            return 0.0
        
        diffs = np.diff(trajectory, axis=0)
        diffs[:, 1] = 0
        distances = np.linalg.norm(diffs, axis=1)
        
        return float(np.sum(distances))
    
    def detect_motion_type(self) -> str:
        """
        Detect type of motion based on velocity.
        
        Returns:
            "idle", "walking", "running", or "jumping"
        """
        if not self._root_history:
            return "idle"
        
        recent = self._root_history[-30:]
        
        speeds = [r.speed for r in recent]
        avg_speed = np.mean(speeds)
        
        grounded_ratio = sum(1 for r in recent if r.is_grounded) / len(recent)
        
        if grounded_ratio < 0.3:
            return "jumping"
        elif avg_speed > 3.0:
            return "running"
        elif avg_speed > 0.5:
            return "walking"
        else:
            return "idle"
    
    def reset(self) -> None:
        """Reset root motion state."""
        self._root_history.clear()
        self._left_foot_contact = None
        self._right_foot_contact = None
        self._velocity_window.clear()
        self._prev_position = None
        self._prev_timestamp = 0.0
        self._frame_count = 0
    
    @property
    def history(self) -> List[RootTransform]:
        return self._root_history
    
    @property
    def current_root(self) -> Optional[RootTransform]:
        return self._root_history[-1] if self._root_history else None
