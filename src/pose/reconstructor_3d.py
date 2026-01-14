"""3D Pose Reconstruction from 2D poses using MediaPipe PoseLandmarker"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.core import get_logger, Config
from .estimator_2d import JointIndex, JOINT_NAMES, Pose2D, Joint2D, download_model_if_needed


@dataclass
class Joint3D:
    """Single 3D joint with position in world coordinates."""
    name: str
    index: int
    x: float  # meters
    y: float  # meters
    z: float  # meters
    confidence: float
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    @property
    def is_visible(self) -> bool:
        return self.confidence > 0.5
    
    def distance_to(self, other: "Joint3D") -> float:
        """Euclidean distance to another joint."""
        return float(np.linalg.norm(self.position - other.position))


@dataclass
class Pose3D:
    """Complete 3D pose for a single frame."""
    frame_number: int
    timestamp: float
    joints: Dict[str, Joint3D] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        required = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
        return all(
            name in self.joints and self.joints[name].is_visible
            for name in required
        )
    
    @property
    def num_visible_joints(self) -> int:
        return sum(1 for j in self.joints.values() if j.is_visible)
    
    def get_joint(self, name: str) -> Optional[Joint3D]:
        return self.joints.get(name)
    
    def get_joint_by_index(self, index: int) -> Optional[Joint3D]:
        name = JOINT_NAMES.get(JointIndex(index))
        return self.joints.get(name) if name else None
    
    @property
    def pelvis(self) -> Optional[np.ndarray]:
        """Get pelvis center (midpoint of hips)."""
        left_hip = self.joints.get("left_hip")
        right_hip = self.joints.get("right_hip")
        if left_hip and right_hip:
            return (left_hip.position + right_hip.position) / 2
        return None
    
    @property
    def chest(self) -> Optional[np.ndarray]:
        """Get chest center (midpoint of shoulders)."""
        left_shoulder = self.joints.get("left_shoulder")
        right_shoulder = self.joints.get("right_shoulder")
        if left_shoulder and right_shoulder:
            return (left_shoulder.position + right_shoulder.position) / 2
        return None
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array (33, 4) - x, y, z, confidence."""
        arr = np.zeros((33, 4), dtype=np.float32)
        for idx in JointIndex:
            name = JOINT_NAMES[idx]
            if name in self.joints:
                joint = self.joints[name]
                arr[idx] = [joint.x, joint.y, joint.z, joint.confidence]
        return arr
    
    def get_bone_length(self, joint1: str, joint2: str) -> Optional[float]:
        """Get distance between two joints."""
        j1 = self.joints.get(joint1)
        j2 = self.joints.get(joint2)
        if j1 and j2:
            return j1.distance_to(j2)
        return None


class OneEuroFilter:
    """
    One Euro Filter for smooth, low-latency signal filtering.
    
    Adapts cutoff frequency based on signal speed.
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None
    
    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)
    
    def _exponential_smoothing(
        self,
        a: float,
        x: np.ndarray,
        x_prev: np.ndarray
    ) -> np.ndarray:
        return a * x + (1 - a) * x_prev
    
    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Filter a value.
        
        Args:
            x: Current value (can be array)
            t: Current timestamp
        
        Returns:
            Filtered value
        """
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x
        
        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev
        
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        if isinstance(cutoff, np.ndarray):
            cutoff = np.mean(cutoff)
        
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self._x_prev)
        
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        
        return x_hat
    
    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None


class MovingAverageFilter:
    """Simple moving average filter."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
    
    def filter(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        self._buffer.append(x)
        return np.mean(list(self._buffer), axis=0)
    
    def reset(self) -> None:
        self._buffer.clear()


class PoseReconstructor3D:
    """
    3D pose reconstruction using MediaPipe PoseLandmarker.
    
    Features:
    - Single-camera depth inference via world landmarks
    - Temporal smoothing (One Euro, Moving Average)
    - Stable 3D joint coordinates in meters
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("pose.3d")
        self.config = config or Config()
        
        pose_config = self.config.pose_estimation
        pose_3d_config = self.config.pose_3d
        
        self._use_world_landmarks = pose_3d_config.get("use_world_landmarks", True)
        self._temporal_smoothing = pose_3d_config.get("temporal_smoothing", True)
        self._smoothing_window = pose_3d_config.get("smoothing_window", 5)
        self._smoothing_method = pose_3d_config.get("smoothing_method", "one_euro")
        
        model_path = download_model_if_needed()
        
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=pose_config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_config.get("min_tracking_confidence", 0.5),
            output_segmentation_masks=False
        )
        
        self._pose = mp_vision.PoseLandmarker.create_from_options(options)
        
        self._joint_filters: Dict[str, OneEuroFilter] = {}
        if self._temporal_smoothing:
            for name in JOINT_NAMES.values():
                if self._smoothing_method == "one_euro":
                    self._joint_filters[name] = OneEuroFilter(
                        min_cutoff=1.0,
                        beta=0.007
                    )
                else:
                    self._joint_filters[name] = MovingAverageFilter(
                        window_size=self._smoothing_window
                    )
        
        self._frame_count = 0
        self._pose_history: List[Pose3D] = []
        self._max_history = 300
        
        self.logger.info(
            f"Initialized 3D reconstructor (world_landmarks={self._use_world_landmarks}, "
            f"smoothing={self._smoothing_method if self._temporal_smoothing else 'none'})"
        )
    
    def process(self, frame: np.ndarray, timestamp: float = 0.0) -> Optional[Pose3D]:
        """
        Process a frame and extract 3D pose.
        
        Args:
            frame: RGB image (H, W, 3)
            timestamp: Frame timestamp in seconds
        
        Returns:
            Pose3D or None if no pose detected
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp_ms = int(timestamp * 1000)
        if timestamp_ms <= 0:
            timestamp_ms = self._frame_count * 33
        
        try:
            results = self._pose.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            self.logger.debug(f"Frame {self._frame_count}: Detection error: {e}")
            self._frame_count += 1
            return None
        
        if self._use_world_landmarks and results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks[0] if results.pose_world_landmarks else None
        elif results.pose_landmarks:
            landmarks = results.pose_landmarks[0] if results.pose_landmarks else None
        else:
            landmarks = None
        
        if landmarks is None or len(landmarks) == 0:
            self.logger.debug(f"Frame {self._frame_count}: No 3D pose detected")
            self._frame_count += 1
            return None
        
        joints = {}
        for idx in JointIndex:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                name = JOINT_NAMES[idx]
                
                x, y, z = landmark.x, landmark.y, landmark.z
                
                if self._temporal_smoothing and name in self._joint_filters:
                    pos = np.array([x, y, z], dtype=np.float32)
                    filtered_pos = self._joint_filters[name].filter(pos, timestamp)
                    x, y, z = filtered_pos
                
                confidence = landmark.visibility if hasattr(landmark, 'visibility') else 0.9
                
                joints[name] = Joint3D(
                    name=name,
                    index=int(idx),
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    confidence=confidence
                )
        
        pose = Pose3D(
            frame_number=self._frame_count,
            timestamp=timestamp,
            joints=joints
        )
        
        self._pose_history.append(pose)
        if len(self._pose_history) > self._max_history:
            self._pose_history.pop(0)
        
        self._frame_count += 1
        
        return pose
    
    def process_from_2d(
        self,
        pose_2d: Pose2D,
        frame: np.ndarray,
        timestamp: float = 0.0
    ) -> Optional[Pose3D]:
        """
        Process 3D pose using existing 2D pose and frame.
        
        This allows reusing the 2D detection while getting 3D world landmarks.
        """
        return self.process(frame, timestamp)
    
    def get_pose_sequence(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[Pose3D]:
        """Get sequence of poses from history."""
        if end_frame is None:
            end_frame = len(self._pose_history)
        
        return [
            p for p in self._pose_history
            if start_frame <= p.frame_number < end_frame
        ]
    
    def get_joint_trajectory(self, joint_name: str) -> np.ndarray:
        """
        Get trajectory of a single joint over time.
        
        Returns:
            Array of shape (N, 3) with x, y, z positions
        """
        positions = []
        for pose in self._pose_history:
            joint = pose.joints.get(joint_name)
            if joint:
                positions.append(joint.position)
        
        return np.array(positions) if positions else np.zeros((0, 3))
    
    def estimate_scale(self) -> float:
        """
        Estimate body scale from bone lengths.
        
        Returns:
            Estimated height in meters
        """
        if not self._pose_history:
            return 1.7  # Default height
        
        bone_lengths = []
        for pose in self._pose_history[-30:]:  # Last 30 frames
            spine_length = pose.get_bone_length("left_hip", "left_shoulder")
            if spine_length:
                bone_lengths.append(spine_length)
        
        if bone_lengths:
            avg_spine = np.mean(bone_lengths)
            estimated_height = avg_spine * 3.0
            return float(np.clip(estimated_height, 1.4, 2.2))
        
        return 1.7
    
    def clear_history(self) -> None:
        """Clear pose history and reset filters."""
        self._pose_history.clear()
        self._frame_count = 0
        
        for filter_obj in self._joint_filters.values():
            filter_obj.reset()
    
    @property
    def history(self) -> List[Pose3D]:
        return self._pose_history
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    def close(self) -> None:
        """Release resources."""
        self._pose.close()
        self.logger.info("3D reconstructor closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
