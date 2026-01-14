"""2D Pose Estimation using MediaPipe PoseLandmarker (Tasks API)"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import urllib.request
import os
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.core import get_logger, Config


class JointIndex(IntEnum):
    """MediaPipe Pose landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


JOINT_NAMES = {
    JointIndex.NOSE: "nose",
    JointIndex.LEFT_EYE_INNER: "left_eye_inner",
    JointIndex.LEFT_EYE: "left_eye",
    JointIndex.LEFT_EYE_OUTER: "left_eye_outer",
    JointIndex.RIGHT_EYE_INNER: "right_eye_inner",
    JointIndex.RIGHT_EYE: "right_eye",
    JointIndex.RIGHT_EYE_OUTER: "right_eye_outer",
    JointIndex.LEFT_EAR: "left_ear",
    JointIndex.RIGHT_EAR: "right_ear",
    JointIndex.MOUTH_LEFT: "mouth_left",
    JointIndex.MOUTH_RIGHT: "mouth_right",
    JointIndex.LEFT_SHOULDER: "left_shoulder",
    JointIndex.RIGHT_SHOULDER: "right_shoulder",
    JointIndex.LEFT_ELBOW: "left_elbow",
    JointIndex.RIGHT_ELBOW: "right_elbow",
    JointIndex.LEFT_WRIST: "left_wrist",
    JointIndex.RIGHT_WRIST: "right_wrist",
    JointIndex.LEFT_PINKY: "left_pinky",
    JointIndex.RIGHT_PINKY: "right_pinky",
    JointIndex.LEFT_INDEX: "left_index",
    JointIndex.RIGHT_INDEX: "right_index",
    JointIndex.LEFT_THUMB: "left_thumb",
    JointIndex.RIGHT_THUMB: "right_thumb",
    JointIndex.LEFT_HIP: "left_hip",
    JointIndex.RIGHT_HIP: "right_hip",
    JointIndex.LEFT_KNEE: "left_knee",
    JointIndex.RIGHT_KNEE: "right_knee",
    JointIndex.LEFT_ANKLE: "left_ankle",
    JointIndex.RIGHT_ANKLE: "right_ankle",
    JointIndex.LEFT_HEEL: "left_heel",
    JointIndex.RIGHT_HEEL: "right_heel",
    JointIndex.LEFT_FOOT_INDEX: "left_foot_index",
    JointIndex.RIGHT_FOOT_INDEX: "right_foot_index",
}

SKELETON_BONES = [
    (JointIndex.LEFT_SHOULDER, JointIndex.RIGHT_SHOULDER),
    (JointIndex.LEFT_SHOULDER, JointIndex.LEFT_ELBOW),
    (JointIndex.LEFT_ELBOW, JointIndex.LEFT_WRIST),
    (JointIndex.RIGHT_SHOULDER, JointIndex.RIGHT_ELBOW),
    (JointIndex.RIGHT_ELBOW, JointIndex.RIGHT_WRIST),
    (JointIndex.LEFT_SHOULDER, JointIndex.LEFT_HIP),
    (JointIndex.RIGHT_SHOULDER, JointIndex.RIGHT_HIP),
    (JointIndex.LEFT_HIP, JointIndex.RIGHT_HIP),
    (JointIndex.LEFT_HIP, JointIndex.LEFT_KNEE),
    (JointIndex.LEFT_KNEE, JointIndex.LEFT_ANKLE),
    (JointIndex.RIGHT_HIP, JointIndex.RIGHT_KNEE),
    (JointIndex.RIGHT_KNEE, JointIndex.RIGHT_ANKLE),
    (JointIndex.LEFT_ANKLE, JointIndex.LEFT_HEEL),
    (JointIndex.LEFT_ANKLE, JointIndex.LEFT_FOOT_INDEX),
    (JointIndex.RIGHT_ANKLE, JointIndex.RIGHT_HEEL),
    (JointIndex.RIGHT_ANKLE, JointIndex.RIGHT_FOOT_INDEX),
    (JointIndex.NOSE, JointIndex.LEFT_EYE),
    (JointIndex.NOSE, JointIndex.RIGHT_EYE),
    (JointIndex.LEFT_EYE, JointIndex.LEFT_EAR),
    (JointIndex.RIGHT_EYE, JointIndex.RIGHT_EAR),
]


@dataclass
class Joint2D:
    """Single 2D joint with position and confidence."""
    name: str
    index: int
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    confidence: float  # 0-1
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @property
    def is_visible(self) -> bool:
        return self.confidence > 0.5
    
    def pixel_coords(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coords to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


@dataclass
class Pose2D:
    """Complete 2D pose for a single frame."""
    frame_number: int
    timestamp: float
    joints: Dict[str, Joint2D] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if pose has minimum required joints."""
        required = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
        return all(
            name in self.joints and self.joints[name].is_visible
            for name in required
        )
    
    @property
    def num_visible_joints(self) -> int:
        return sum(1 for j in self.joints.values() if j.is_visible)
    
    def get_joint(self, name: str) -> Optional[Joint2D]:
        return self.joints.get(name)
    
    def get_joint_by_index(self, index: int) -> Optional[Joint2D]:
        name = JOINT_NAMES.get(JointIndex(index))
        return self.joints.get(name) if name else None
    
    @property
    def pelvis(self) -> Optional[Tuple[float, float]]:
        """Get pelvis center (midpoint of hips)."""
        left_hip = self.joints.get("left_hip")
        right_hip = self.joints.get("right_hip")
        if left_hip and right_hip:
            return (
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2
            )
        return None
    
    @property
    def chest(self) -> Optional[Tuple[float, float]]:
        """Get chest center (midpoint of shoulders)."""
        left_shoulder = self.joints.get("left_shoulder")
        right_shoulder = self.joints.get("right_shoulder")
        if left_shoulder and right_shoulder:
            return (
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            )
        return None
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array (33, 3) - x, y, confidence."""
        arr = np.zeros((33, 3), dtype=np.float32)
        for idx in JointIndex:
            name = JOINT_NAMES[idx]
            if name in self.joints:
                joint = self.joints[name]
                arr[idx] = [joint.x, joint.y, joint.confidence]
        return arr


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pose_landmarker.task"


def download_model_if_needed() -> Path:
    """Download the pose landmarker model if not present."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not MODEL_PATH.exists():
        print(f"Downloading pose model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    
    return MODEL_PATH


class PoseEstimator2D:
    """
    2D pose estimation using MediaPipe PoseLandmarker (Tasks API).
    
    Detects 33 body landmarks per frame with confidence scores.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("pose.2d")
        self.config = config or Config()
        
        pose_config = self.config.pose_estimation
        
        self._min_detection_confidence = pose_config.get("min_detection_confidence", 0.5)
        self._min_tracking_confidence = pose_config.get("min_tracking_confidence", 0.5)
        
        model_path = download_model_if_needed()
        
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_segmentation_masks=False
        )
        
        self._pose = mp_vision.PoseLandmarker.create_from_options(options)
        
        self._frame_count = 0
        self._pose_history: List[Pose2D] = []
        self._max_history = 300  # 10 seconds at 30fps
        
        self.logger.info(
            f"Initialized MediaPipe PoseLandmarker (detection={self._min_detection_confidence}, "
            f"tracking={self._min_tracking_confidence})"
        )
    
    def process(self, frame: np.ndarray, timestamp: float = 0.0) -> Optional[Pose2D]:
        """
        Process a single frame and extract 2D pose.
        
        Args:
            frame: RGB image (H, W, 3)
            timestamp: Frame timestamp in seconds
        
        Returns:
            Pose2D or None if no pose detected
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp_ms = int(timestamp * 1000)
        if timestamp_ms <= 0:
            timestamp_ms = self._frame_count * 33  # ~30fps fallback
        
        try:
            results = self._pose.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            self.logger.debug(f"Frame {self._frame_count}: Detection error: {e}")
            self._frame_count += 1
            return None
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            self.logger.debug(f"Frame {self._frame_count}: No pose detected")
            self._frame_count += 1
            return None
        
        landmarks = results.pose_landmarks[0]
        
        joints = {}
        for idx in JointIndex:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                name = JOINT_NAMES[idx]
                
                joints[name] = Joint2D(
                    name=name,
                    index=int(idx),
                    x=landmark.x,
                    y=landmark.y,
                    confidence=landmark.visibility if hasattr(landmark, 'visibility') else 0.9
                )
        
        pose = Pose2D(
            frame_number=self._frame_count,
            timestamp=timestamp,
            joints=joints
        )
        
        self._pose_history.append(pose)
        if len(self._pose_history) > self._max_history:
            self._pose_history.pop(0)
        
        self._frame_count += 1
        
        return pose
    
    def draw_pose(
        self,
        frame: np.ndarray,
        pose: Optional[Pose2D] = None,
        draw_connections: bool = True,
        joint_color: Tuple[int, int, int] = (0, 255, 0),
        bone_color: Tuple[int, int, int] = (255, 255, 255),
        joint_radius: int = 5,
        bone_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw pose overlay on frame.
        
        Args:
            frame: RGB image to draw on (will be copied)
            pose: Pose2D to draw (uses last detected if None)
            draw_connections: Whether to draw skeleton bones
            joint_color: RGB color for joints
            bone_color: RGB color for bones
            joint_radius: Radius of joint circles
            bone_thickness: Thickness of bone lines
        
        Returns:
            Frame with pose overlay
        """
        output = frame.copy()
        
        if pose is None:
            pose = self._pose_history[-1] if self._pose_history else None
        
        if pose is None:
            return output
        
        height, width = frame.shape[:2]
        
        if draw_connections:
            for start_idx, end_idx in SKELETON_BONES:
                start_name = JOINT_NAMES[start_idx]
                end_name = JOINT_NAMES[end_idx]
                
                start_joint = pose.joints.get(start_name)
                end_joint = pose.joints.get(end_name)
                
                if start_joint and end_joint and start_joint.is_visible and end_joint.is_visible:
                    start_px = start_joint.pixel_coords(width, height)
                    end_px = end_joint.pixel_coords(width, height)
                    
                    import cv2
                    cv2.line(output, start_px, end_px, bone_color, bone_thickness)
        
        for joint in pose.joints.values():
            if joint.is_visible:
                px, py = joint.pixel_coords(width, height)
                
                alpha = min(1.0, joint.confidence + 0.3)
                color = tuple(int(c * alpha) for c in joint_color)
                
                import cv2
                cv2.circle(output, (px, py), joint_radius, color, -1)
        
        return output
    
    def get_pose_sequence(self, start_frame: int = 0, end_frame: Optional[int] = None) -> List[Pose2D]:
        """Get sequence of poses from history."""
        if end_frame is None:
            end_frame = len(self._pose_history)
        
        return [
            p for p in self._pose_history
            if start_frame <= p.frame_number < end_frame
        ]
    
    def clear_history(self) -> None:
        """Clear pose history."""
        self._pose_history.clear()
        self._frame_count = 0
    
    @property
    def history(self) -> List[Pose2D]:
        return self._pose_history
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    def close(self) -> None:
        """Release resources."""
        self._pose.close()
        self.logger.info("Pose estimator closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
