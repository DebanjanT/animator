"""Unified Pose Processor - Single model inference for both 2D and 3D poses"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.core import get_logger, Config
from .estimator_2d import (
    JointIndex, JOINT_NAMES, SKELETON_BONES, 
    Joint2D, Pose2D, POSE_MODELS, MODEL_DIR
)
from .reconstructor_3d import Joint3D, Pose3D, OneEuroFilter, MovingAverageFilter


@dataclass
class UnifiedPoseResult:
    """Combined 2D and 3D pose result from single inference."""
    pose_2d: Optional[Pose2D]
    pose_3d: Optional[Pose3D]
    frame_number: int
    timestamp: float


class UnifiedPoseProcessor:
    """
    Unified pose processor that runs MediaPipe once and extracts both 2D and 3D poses.
    
    This is significantly faster than running separate 2D and 3D estimators.
    """
    
    def __init__(self, config: Optional[Config] = None, model_type: str = None):
        self.logger = get_logger("pose.unified")
        self.config = config or Config()
        
        pose_config = self.config.pose_estimation
        pose_3d_config = self.config.pose_3d
        
        self._min_detection_confidence = pose_config.get("min_detection_confidence", 0.5)
        self._min_tracking_confidence = pose_config.get("min_tracking_confidence", 0.5)
        
        if model_type is None:
            model_type = pose_config.get("model_type", "full")
        self._model_type = model_type
        
        model_path = self._download_model(model_type)
        
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
        
        self._temporal_smoothing = pose_3d_config.get("temporal_smoothing", True)
        self._smoothing_method = pose_3d_config.get("smoothing_method", "one_euro")
        self._smoothing_window = pose_3d_config.get("smoothing_window", 5)
        
        self._joint_filters: Dict[str, OneEuroFilter] = {}
        if self._temporal_smoothing:
            for name in JOINT_NAMES.values():
                if self._smoothing_method == "one_euro":
                    self._joint_filters[name] = OneEuroFilter(min_cutoff=1.0, beta=0.007)
                else:
                    self._joint_filters[name] = MovingAverageFilter(window_size=self._smoothing_window)
        
        self._frame_count = 0
        self._last_result: Optional[UnifiedPoseResult] = None
        
        self.logger.info(
            f"Initialized UnifiedPoseProcessor [{model_type}] "
            f"(detection={self._min_detection_confidence}, tracking={self._min_tracking_confidence})"
        )
    
    def _download_model(self, model_type: str) -> Path:
        """Download model if needed."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        if model_type not in POSE_MODELS:
            model_type = "full"
        
        model_info = POSE_MODELS[model_type]
        model_path = MODEL_DIR / model_info["filename"]
        
        if not model_path.exists():
            import urllib.request
            print(f"Downloading {model_type} pose model to {model_path}...")
            urllib.request.urlretrieve(model_info["url"], model_path)
            print("Download complete.")
        
        return model_path
    
    def process(self, frame: np.ndarray, timestamp: float = 0.0) -> UnifiedPoseResult:
        """
        Process frame and extract both 2D and 3D poses in single inference.
        
        Args:
            frame: RGB image (H, W, 3)
            timestamp: Frame timestamp in seconds
        
        Returns:
            UnifiedPoseResult with both pose_2d and pose_3d
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp_ms = int(timestamp * 1000)
        if timestamp_ms <= 0:
            timestamp_ms = self._frame_count * 33
        
        pose_2d = None
        pose_3d = None
        
        try:
            results = self._pose.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            self.logger.debug(f"Frame {self._frame_count}: Detection error: {e}")
            self._frame_count += 1
            return UnifiedPoseResult(None, None, self._frame_count - 1, timestamp)
        
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks_2d = results.pose_landmarks[0]
            
            joints_2d = {}
            for idx in JointIndex:
                if idx < len(landmarks_2d):
                    lm = landmarks_2d[idx]
                    name = JOINT_NAMES[idx]
                    joints_2d[name] = Joint2D(
                        name=name,
                        index=int(idx),
                        x=lm.x,
                        y=lm.y,
                        confidence=lm.visibility if hasattr(lm, 'visibility') else 0.9
                    )
            
            pose_2d = Pose2D(
                frame_number=self._frame_count,
                timestamp=timestamp,
                joints=joints_2d
            )
        
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            landmarks_3d = results.pose_world_landmarks[0]
            
            joints_3d = {}
            for idx in JointIndex:
                if idx < len(landmarks_3d):
                    lm = landmarks_3d[idx]
                    name = JOINT_NAMES[idx]
                    
                    x, y, z = lm.x, lm.y, lm.z
                    
                    if self._temporal_smoothing and name in self._joint_filters:
                        pos = np.array([x, y, z], dtype=np.float32)
                        filtered = self._joint_filters[name].filter(pos, timestamp)
                        x, y, z = filtered
                    
                    joints_3d[name] = Joint3D(
                        name=name,
                        index=int(idx),
                        x=float(x),
                        y=float(y),
                        z=float(z),
                        confidence=lm.visibility if hasattr(lm, 'visibility') else 0.9
                    )
            
            pose_3d = Pose3D(
                frame_number=self._frame_count,
                timestamp=timestamp,
                joints=joints_3d
            )
        
        result = UnifiedPoseResult(
            pose_2d=pose_2d,
            pose_3d=pose_3d,
            frame_number=self._frame_count,
            timestamp=timestamp
        )
        
        self._last_result = result
        self._frame_count += 1
        
        return result
    
    def draw_pose(
        self,
        frame: np.ndarray,
        pose_2d: Pose2D,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        radius: int = 5
    ) -> np.ndarray:
        """Draw 2D pose skeleton on frame."""
        import cv2
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        for start_name, end_name in SKELETON_BONES:
            start_joint = pose_2d.joints.get(start_name)
            end_joint = pose_2d.joints.get(end_name)
            
            if start_joint and end_joint and start_joint.is_visible and end_joint.is_visible:
                start_pt = start_joint.pixel_coords(w, h)
                end_pt = end_joint.pixel_coords(w, h)
                cv2.line(output, start_pt, end_pt, color, thickness)
        
        for joint in pose_2d.joints.values():
            if joint.is_visible:
                pt = joint.pixel_coords(w, h)
                cv2.circle(output, pt, radius, color, -1)
                cv2.circle(output, pt, radius + 1, (255, 255, 255), 1)
        
        return output
    
    def close(self) -> None:
        """Release resources."""
        if self._pose:
            self._pose.close()
        self.logger.info("Unified pose processor closed")
