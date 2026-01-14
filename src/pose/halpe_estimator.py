"""Halpe Pose Estimator - Real-time Full Body Tracking

This module provides real-time Halpe 136-keypoint pose estimation using:
1. Pre-processed AlphaPose JSON files (offline processing)
2. ONNX runtime inference (if model available)
3. MediaPipe conversion fallback (real-time)

For best results, pre-process videos with AlphaPose and load the JSON.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
import cv2

from src.core import get_logger, Config
from src.pose.halpe_keypoints import (
    HalpeBodyIndex, HalpeHandIndex,
    HALPE_BODY_NAMES, HALPE_HAND_NAMES,
    HALPE_BODY_CONNECTIONS, HALPE_HAND_CONNECTIONS,
    LEFT_HAND_START, RIGHT_HAND_START,
    FACE_KEYPOINT_START,
)
from src.pose.alphapose_estimator import HalpeKeypoint, HalpePose, convert_halpe_to_mixamo


class HalpePoseLoader:
    """Load pre-processed AlphaPose Halpe poses from JSON."""
    
    def __init__(self, json_path: str):
        self.logger = get_logger("pose.halpe_loader")
        self.json_path = json_path
        self.poses_by_frame: Dict[int, List[HalpePose]] = {}
        self._load_json()
    
    def _load_json(self):
        """Load and parse AlphaPose JSON output."""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                # Extract frame number from image_id
                image_id = item.get('image_id', '0')
                if isinstance(image_id, str):
                    frame_num = int(image_id.split('.')[0].split('_')[-1])
                else:
                    frame_num = int(image_id)
                
                # Parse keypoints
                kpts = item.get('keypoints', [])
                keypoints = {}
                num_kpts = len(kpts) // 3
                
                for i in range(num_kpts):
                    x = kpts[i * 3]
                    y = kpts[i * 3 + 1]
                    conf = kpts[i * 3 + 2]
                    
                    if i < 26:
                        name = HALPE_BODY_NAMES.get(HalpeBodyIndex(i), f"body_{i}")
                    elif i < 94:
                        name = f"face_{i - 26}"
                    elif i < 115:
                        hand_idx = i - 94
                        name = f"left_{HALPE_HAND_NAMES.get(HalpeHandIndex(hand_idx), f'h{hand_idx}')}"
                    else:
                        hand_idx = i - 115
                        name = f"right_{HALPE_HAND_NAMES.get(HalpeHandIndex(hand_idx), f'h{hand_idx}')}"
                    
                    keypoints[i] = HalpeKeypoint(
                        index=i, name=name, x=x, y=y, confidence=conf
                    )
                
                bbox = item.get('box', [0, 0, 0, 0])
                
                pose = HalpePose(
                    person_id=item.get('idx', 0),
                    frame_number=frame_num,
                    timestamp=frame_num / 30.0,
                    bbox=tuple(bbox[:4]) if len(bbox) >= 4 else (0, 0, 0, 0),
                    keypoints=keypoints,
                    tracking_id=item.get('track_id')
                )
                
                if frame_num not in self.poses_by_frame:
                    self.poses_by_frame[frame_num] = []
                self.poses_by_frame[frame_num].append(pose)
            
            self.logger.info(f"Loaded {len(self.poses_by_frame)} frames from {self.json_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading JSON: {e}")
    
    def get_poses(self, frame_num: int) -> List[HalpePose]:
        """Get poses for a specific frame."""
        return self.poses_by_frame.get(frame_num, [])
    
    def get_primary_pose(self, frame_num: int) -> Optional[HalpePose]:
        """Get the primary (largest/most confident) pose for a frame."""
        poses = self.get_poses(frame_num)
        if not poses:
            return None
        
        # Return pose with largest bbox or highest confidence
        return max(poses, key=lambda p: p.bbox[2] * p.bbox[3] if p.bbox else 0)


class MediaPipeToHalpeConverter:
    """Convert MediaPipe poses to Halpe format for compatibility."""
    
    # MediaPipe to Halpe body mapping
    MP_TO_HALPE_BODY = {
        0: HalpeBodyIndex.NOSE,
        1: HalpeBodyIndex.LEFT_EYE,  # left_eye_inner -> left_eye
        2: HalpeBodyIndex.LEFT_EYE,
        3: HalpeBodyIndex.LEFT_EAR,  # left_eye_outer -> left_ear approx
        4: HalpeBodyIndex.RIGHT_EYE,  # right_eye_inner
        5: HalpeBodyIndex.RIGHT_EYE,
        6: HalpeBodyIndex.RIGHT_EAR,  # right_eye_outer
        7: HalpeBodyIndex.LEFT_EAR,
        8: HalpeBodyIndex.RIGHT_EAR,
        11: HalpeBodyIndex.LEFT_SHOULDER,
        12: HalpeBodyIndex.RIGHT_SHOULDER,
        13: HalpeBodyIndex.LEFT_ELBOW,
        14: HalpeBodyIndex.RIGHT_ELBOW,
        15: HalpeBodyIndex.LEFT_WRIST,
        16: HalpeBodyIndex.RIGHT_WRIST,
        23: HalpeBodyIndex.LEFT_HIP,
        24: HalpeBodyIndex.RIGHT_HIP,
        25: HalpeBodyIndex.LEFT_KNEE,
        26: HalpeBodyIndex.RIGHT_KNEE,
        27: HalpeBodyIndex.LEFT_ANKLE,
        28: HalpeBodyIndex.RIGHT_ANKLE,
        29: HalpeBodyIndex.LEFT_HEEL,
        30: HalpeBodyIndex.RIGHT_HEEL,
        31: HalpeBodyIndex.LEFT_BIG_TOE,
        32: HalpeBodyIndex.RIGHT_BIG_TOE,
    }
    
    @staticmethod
    def convert(mp_pose, frame_number: int = 0, timestamp: float = 0.0) -> Optional[HalpePose]:
        """Convert MediaPipe Pose2D to HalpePose."""
        keypoints = {}
        
        # Convert body keypoints
        for mp_idx, halpe_idx in MediaPipeToHalpeConverter.MP_TO_HALPE_BODY.items():
            joint = mp_pose.get_joint_by_index(mp_idx)
            if joint:
                keypoints[int(halpe_idx)] = HalpeKeypoint(
                    index=int(halpe_idx),
                    name=HALPE_BODY_NAMES.get(halpe_idx, f"body_{halpe_idx}"),
                    x=joint.x,
                    y=joint.y,
                    z=getattr(joint, 'z', 0.0),
                    confidence=joint.confidence
                )
        
        # Compute derived keypoints
        left_hip = keypoints.get(int(HalpeBodyIndex.LEFT_HIP))
        right_hip = keypoints.get(int(HalpeBodyIndex.RIGHT_HIP))
        left_shoulder = keypoints.get(int(HalpeBodyIndex.LEFT_SHOULDER))
        right_shoulder = keypoints.get(int(HalpeBodyIndex.RIGHT_SHOULDER))
        
        # Hip center
        if left_hip and right_hip:
            keypoints[int(HalpeBodyIndex.HIP)] = HalpeKeypoint(
                index=int(HalpeBodyIndex.HIP),
                name="hip_center",
                x=(left_hip.x + right_hip.x) / 2,
                y=(left_hip.y + right_hip.y) / 2,
                z=(left_hip.z + right_hip.z) / 2,
                confidence=min(left_hip.confidence, right_hip.confidence)
            )
        
        # Neck
        if left_shoulder and right_shoulder:
            keypoints[int(HalpeBodyIndex.NECK)] = HalpeKeypoint(
                index=int(HalpeBodyIndex.NECK),
                name="neck",
                x=(left_shoulder.x + right_shoulder.x) / 2,
                y=(left_shoulder.y + right_shoulder.y) / 2 - 0.05,  # Slightly above shoulders
                z=(left_shoulder.z + right_shoulder.z) / 2,
                confidence=min(left_shoulder.confidence, right_shoulder.confidence)
            )
        
        # Head (above nose)
        nose = keypoints.get(int(HalpeBodyIndex.NOSE))
        if nose:
            keypoints[int(HalpeBodyIndex.HEAD)] = HalpeKeypoint(
                index=int(HalpeBodyIndex.HEAD),
                name="head",
                x=nose.x,
                y=nose.y - 0.08,  # Above nose
                z=nose.z,
                confidence=nose.confidence
            )
        
        if not keypoints:
            return None
        
        return HalpePose(
            person_id=0,
            frame_number=frame_number,
            timestamp=timestamp,
            keypoints=keypoints
        )


class HalpeEstimator:
    """
    Real-time Halpe pose estimator with multiple backends.
    
    Priority order:
    1. Pre-loaded AlphaPose JSON (if available)
    2. ONNX model inference (if model available)
    3. MediaPipe conversion fallback
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        json_path: Optional[str] = None,
        use_mediapipe_fallback: bool = True
    ):
        self.logger = get_logger("pose.halpe")
        self.config = config or Config()
        
        self._json_loader: Optional[HalpePoseLoader] = None
        self._mp_estimator = None
        self._mp_hand_tracker = None
        self._frame_count = 0
        
        # Load JSON if provided
        if json_path and Path(json_path).exists():
            self._json_loader = HalpePoseLoader(json_path)
            self.logger.info(f"Using pre-loaded poses from {json_path}")
        
        # Initialize MediaPipe fallback
        if use_mediapipe_fallback:
            self._init_mediapipe()
        
        self.logger.info(
            f"Initialized Halpe estimator (json={json_path is not None}, "
            f"mediapipe={self._mp_estimator is not None})"
        )
    
    def _init_mediapipe(self):
        """Initialize MediaPipe for fallback."""
        try:
            from src.pose.estimator_2d import PoseEstimator2D
            from src.pose.hand_tracker import HandTracker
            
            self._mp_estimator = PoseEstimator2D(model_type="heavy")
            self._mp_hand_tracker = HandTracker()
            self.logger.info("MediaPipe fallback initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize MediaPipe: {e}")
    
    def process(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0
    ) -> Optional[HalpePose]:
        """
        Process a single frame and return Halpe pose.
        
        Args:
            frame: RGB image (H, W, 3)
            timestamp: Frame timestamp
        
        Returns:
            HalpePose or None
        """
        # Try JSON loader first
        if self._json_loader:
            pose = self._json_loader.get_primary_pose(self._frame_count)
            if pose:
                self._frame_count += 1
                return pose
        
        # Fall back to MediaPipe
        if self._mp_estimator:
            mp_pose = self._mp_estimator.process(frame, timestamp)
            if mp_pose:
                halpe_pose = MediaPipeToHalpeConverter.convert(
                    mp_pose, self._frame_count, timestamp
                )
                
                # Add hand keypoints if tracker available
                if self._mp_hand_tracker and halpe_pose:
                    self._add_hand_keypoints(frame, halpe_pose, timestamp)
                
                self._frame_count += 1
                return halpe_pose
        
        self._frame_count += 1
        return None
    
    def _add_hand_keypoints(self, frame: np.ndarray, pose: HalpePose, timestamp: float):
        """Add hand keypoints from MediaPipe hand tracker."""
        try:
            hands = self._mp_hand_tracker.process(frame, timestamp)
            if not hands:
                return
            
            h, w = frame.shape[:2]
            
            # Left hand
            if hands.left_hand:
                for i, joint in enumerate(hands.left_hand.joints):
                    halpe_idx = LEFT_HAND_START + i
                    pose.keypoints[halpe_idx] = HalpeKeypoint(
                        index=halpe_idx,
                        name=f"left_{HALPE_HAND_NAMES.get(HalpeHandIndex(i), f'h{i}')}",
                        x=joint.x,
                        y=joint.y,
                        z=joint.z if hasattr(joint, 'z') else 0.0,
                        confidence=joint.confidence
                    )
            
            # Right hand
            if hands.right_hand:
                for i, joint in enumerate(hands.right_hand.joints):
                    halpe_idx = RIGHT_HAND_START + i
                    pose.keypoints[halpe_idx] = HalpeKeypoint(
                        index=halpe_idx,
                        name=f"right_{HALPE_HAND_NAMES.get(HalpeHandIndex(i), f'h{i}')}",
                        x=joint.x,
                        y=joint.y,
                        z=joint.z if hasattr(joint, 'z') else 0.0,
                        confidence=joint.confidence
                    )
                    
        except Exception as e:
            self.logger.debug(f"Hand tracking error: {e}")
    
    def draw_pose(
        self,
        frame: np.ndarray,
        pose: HalpePose,
        draw_body: bool = True,
        draw_face: bool = False,
        draw_hands: bool = True,
        draw_labels: bool = True
    ) -> np.ndarray:
        """Draw Halpe pose on frame with all keypoints."""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Color scheme
        BODY_COLOR = (0, 255, 0)      # Green
        FACE_COLOR = (255, 255, 0)    # Cyan
        LEFT_HAND_COLOR = (255, 0, 255)  # Magenta
        RIGHT_HAND_COLOR = (0, 255, 255)  # Yellow
        JOINT_COLOR = (0, 0, 255)     # Red
        
        def to_pixel(kpt: HalpeKeypoint) -> Tuple[int, int]:
            # Handle both normalized (0-1) and pixel coordinates
            if kpt.x <= 1.0 and kpt.y <= 1.0:
                return (int(kpt.x * w), int(kpt.y * h))
            return (int(kpt.x), int(kpt.y))
        
        # Draw body connections
        if draw_body:
            for start_idx, end_idx in HALPE_BODY_CONNECTIONS:
                kpt1 = pose.get_keypoint(int(start_idx))
                kpt2 = pose.get_keypoint(int(end_idx))
                if kpt1 and kpt2 and kpt1.is_visible and kpt2.is_visible:
                    pt1 = to_pixel(kpt1)
                    pt2 = to_pixel(kpt2)
                    cv2.line(result, pt1, pt2, BODY_COLOR, 2)
            
            # Draw body keypoints
            for i in range(26):
                kpt = pose.get_keypoint(i)
                if kpt and kpt.is_visible:
                    pt = to_pixel(kpt)
                    cv2.circle(result, pt, 5, JOINT_COLOR, -1)
                    if draw_labels:
                        cv2.putText(result, str(i), (pt[0]+5, pt[1]-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Draw hands
        if draw_hands:
            # Left hand
            for start_idx, end_idx in HALPE_HAND_CONNECTIONS:
                kpt1 = pose.get_keypoint(LEFT_HAND_START + int(start_idx))
                kpt2 = pose.get_keypoint(LEFT_HAND_START + int(end_idx))
                if kpt1 and kpt2 and kpt1.is_visible and kpt2.is_visible:
                    pt1 = to_pixel(kpt1)
                    pt2 = to_pixel(kpt2)
                    cv2.line(result, pt1, pt2, LEFT_HAND_COLOR, 1)
            
            for i in range(21):
                kpt = pose.get_keypoint(LEFT_HAND_START + i)
                if kpt and kpt.is_visible:
                    pt = to_pixel(kpt)
                    cv2.circle(result, pt, 3, LEFT_HAND_COLOR, -1)
            
            # Right hand
            for start_idx, end_idx in HALPE_HAND_CONNECTIONS:
                kpt1 = pose.get_keypoint(RIGHT_HAND_START + int(start_idx))
                kpt2 = pose.get_keypoint(RIGHT_HAND_START + int(end_idx))
                if kpt1 and kpt2 and kpt1.is_visible and kpt2.is_visible:
                    pt1 = to_pixel(kpt1)
                    pt2 = to_pixel(kpt2)
                    cv2.line(result, pt1, pt2, RIGHT_HAND_COLOR, 1)
            
            for i in range(21):
                kpt = pose.get_keypoint(RIGHT_HAND_START + i)
                if kpt and kpt.is_visible:
                    pt = to_pixel(kpt)
                    cv2.circle(result, pt, 3, RIGHT_HAND_COLOR, -1)
        
        # Draw face keypoints
        if draw_face:
            for i in range(FACE_KEYPOINT_START, 94):
                kpt = pose.get_keypoint(i)
                if kpt and kpt.is_visible:
                    pt = to_pixel(kpt)
                    cv2.circle(result, pt, 2, FACE_COLOR, -1)
        
        # Draw info
        num_visible = sum(1 for kpt in pose.keypoints.values() if kpt.is_visible)
        cv2.putText(result, f"Keypoints: {num_visible}/136", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
