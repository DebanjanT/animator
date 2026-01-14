"""Hand Tracking using MediaPipe HandLandmarker for finger bone detection"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import urllib.request
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.core import get_logger, Config


class HandLandmarkIndex(IntEnum):
    """MediaPipe Hand landmark indices (21 per hand)."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_LANDMARK_NAMES = {
    HandLandmarkIndex.WRIST: "wrist",
    HandLandmarkIndex.THUMB_CMC: "thumb_cmc",
    HandLandmarkIndex.THUMB_MCP: "thumb_mcp",
    HandLandmarkIndex.THUMB_IP: "thumb_ip",
    HandLandmarkIndex.THUMB_TIP: "thumb_tip",
    HandLandmarkIndex.INDEX_MCP: "index_mcp",
    HandLandmarkIndex.INDEX_PIP: "index_pip",
    HandLandmarkIndex.INDEX_DIP: "index_dip",
    HandLandmarkIndex.INDEX_TIP: "index_tip",
    HandLandmarkIndex.MIDDLE_MCP: "middle_mcp",
    HandLandmarkIndex.MIDDLE_PIP: "middle_pip",
    HandLandmarkIndex.MIDDLE_DIP: "middle_dip",
    HandLandmarkIndex.MIDDLE_TIP: "middle_tip",
    HandLandmarkIndex.RING_MCP: "ring_mcp",
    HandLandmarkIndex.RING_PIP: "ring_pip",
    HandLandmarkIndex.RING_DIP: "ring_dip",
    HandLandmarkIndex.RING_TIP: "ring_tip",
    HandLandmarkIndex.PINKY_MCP: "pinky_mcp",
    HandLandmarkIndex.PINKY_PIP: "pinky_pip",
    HandLandmarkIndex.PINKY_DIP: "pinky_dip",
    HandLandmarkIndex.PINKY_TIP: "pinky_tip",
}

HAND_CONNECTIONS = [
    (HandLandmarkIndex.WRIST, HandLandmarkIndex.THUMB_CMC),
    (HandLandmarkIndex.THUMB_CMC, HandLandmarkIndex.THUMB_MCP),
    (HandLandmarkIndex.THUMB_MCP, HandLandmarkIndex.THUMB_IP),
    (HandLandmarkIndex.THUMB_IP, HandLandmarkIndex.THUMB_TIP),
    (HandLandmarkIndex.WRIST, HandLandmarkIndex.INDEX_MCP),
    (HandLandmarkIndex.INDEX_MCP, HandLandmarkIndex.INDEX_PIP),
    (HandLandmarkIndex.INDEX_PIP, HandLandmarkIndex.INDEX_DIP),
    (HandLandmarkIndex.INDEX_DIP, HandLandmarkIndex.INDEX_TIP),
    (HandLandmarkIndex.WRIST, HandLandmarkIndex.MIDDLE_MCP),
    (HandLandmarkIndex.MIDDLE_MCP, HandLandmarkIndex.MIDDLE_PIP),
    (HandLandmarkIndex.MIDDLE_PIP, HandLandmarkIndex.MIDDLE_DIP),
    (HandLandmarkIndex.MIDDLE_DIP, HandLandmarkIndex.MIDDLE_TIP),
    (HandLandmarkIndex.WRIST, HandLandmarkIndex.RING_MCP),
    (HandLandmarkIndex.RING_MCP, HandLandmarkIndex.RING_PIP),
    (HandLandmarkIndex.RING_PIP, HandLandmarkIndex.RING_DIP),
    (HandLandmarkIndex.RING_DIP, HandLandmarkIndex.RING_TIP),
    (HandLandmarkIndex.WRIST, HandLandmarkIndex.PINKY_MCP),
    (HandLandmarkIndex.PINKY_MCP, HandLandmarkIndex.PINKY_PIP),
    (HandLandmarkIndex.PINKY_PIP, HandLandmarkIndex.PINKY_DIP),
    (HandLandmarkIndex.PINKY_DIP, HandLandmarkIndex.PINKY_TIP),
    (HandLandmarkIndex.INDEX_MCP, HandLandmarkIndex.MIDDLE_MCP),
    (HandLandmarkIndex.MIDDLE_MCP, HandLandmarkIndex.RING_MCP),
    (HandLandmarkIndex.RING_MCP, HandLandmarkIndex.PINKY_MCP),
]

UE5_HAND_BONE_MAP = {
    "left": {
        "wrist": "hand_l",
        "thumb_mcp": "thumb_01_l",
        "thumb_ip": "thumb_02_l",
        "thumb_tip": "thumb_03_l",
        "index_mcp": "index_01_l",
        "index_pip": "index_02_l",
        "index_dip": "index_03_l",
        "middle_mcp": "middle_01_l",
        "middle_pip": "middle_02_l",
        "middle_dip": "middle_03_l",
        "ring_mcp": "ring_01_l",
        "ring_pip": "ring_02_l",
        "ring_dip": "ring_03_l",
        "pinky_mcp": "pinky_01_l",
        "pinky_pip": "pinky_02_l",
        "pinky_dip": "pinky_03_l",
    },
    "right": {
        "wrist": "hand_r",
        "thumb_mcp": "thumb_01_r",
        "thumb_ip": "thumb_02_r",
        "thumb_tip": "thumb_03_r",
        "index_mcp": "index_01_r",
        "index_pip": "index_02_r",
        "index_dip": "index_03_r",
        "middle_mcp": "middle_01_r",
        "middle_pip": "middle_02_r",
        "middle_dip": "middle_03_r",
        "ring_mcp": "ring_01_r",
        "ring_pip": "ring_02_r",
        "ring_dip": "ring_03_r",
        "pinky_mcp": "pinky_01_r",
        "pinky_pip": "pinky_02_r",
        "pinky_dip": "pinky_03_r",
    }
}

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "hand_landmarker.task"


def download_hand_model_if_needed() -> Path:
    """Download the hand landmarker model if not present."""
    HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not HAND_MODEL_PATH.exists():
        print(f"Downloading hand model to {HAND_MODEL_PATH}...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("Download complete.")
    
    return HAND_MODEL_PATH


@dataclass
class HandJoint:
    """Single hand joint with 2D and 3D position."""
    name: str
    index: int
    x: float  # normalized 0-1
    y: float  # normalized 0-1
    z: float  # depth (relative)
    confidence: float = 1.0
    
    @property
    def position_2d(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @property
    def position_3d(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    def pixel_coords(self, width: int, height: int) -> Tuple[int, int]:
        return (int(self.x * width), int(self.y * height))


@dataclass
class Hand:
    """Single hand with all 21 landmarks."""
    handedness: str  # "left" or "right"
    joints: Dict[str, HandJoint] = field(default_factory=dict)
    confidence: float = 1.0
    
    @property
    def is_valid(self) -> bool:
        return len(self.joints) >= 21
    
    @property
    def wrist(self) -> Optional[HandJoint]:
        return self.joints.get("wrist")
    
    def get_finger_joints(self, finger: str) -> List[HandJoint]:
        """Get all joints for a finger (thumb, index, middle, ring, pinky)."""
        return [j for name, j in self.joints.items() if name.startswith(finger)]
    
    def get_ue5_bone_name(self, joint_name: str) -> Optional[str]:
        """Map joint name to UE5 Mannequin bone name."""
        bone_map = UE5_HAND_BONE_MAP.get(self.handedness, {})
        return bone_map.get(joint_name)


@dataclass
class HandsData:
    """Hand tracking data for a single frame."""
    frame_number: int
    timestamp: float
    left_hand: Optional[Hand] = None
    right_hand: Optional[Hand] = None
    
    @property
    def has_hands(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None
    
    @property
    def num_hands(self) -> int:
        count = 0
        if self.left_hand:
            count += 1
        if self.right_hand:
            count += 1
        return count


class HandTracker:
    """
    Hand tracking using MediaPipe HandLandmarker.
    
    Detects 21 landmarks per hand with finger bone positions.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("pose.hands")
        self.config = config or Config()
        
        hand_config = self.config.get("hand_tracking", {})
        
        self._min_detection_confidence = hand_config.get("min_detection_confidence", 0.5)
        self._min_tracking_confidence = hand_config.get("min_tracking_confidence", 0.5)
        self._max_hands = hand_config.get("max_hands", 2)
        
        model_path = download_hand_model_if_needed()
        
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        
        self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
        
        self._frame_count = 0
        
        self.logger.info(
            f"Initialized HandTracker (max_hands={self._max_hands}, "
            f"detection={self._min_detection_confidence})"
        )
    
    def process(self, frame: np.ndarray, timestamp: float = 0.0) -> HandsData:
        """
        Process a frame and extract hand landmarks.
        
        Args:
            frame: RGB image (H, W, 3)
            timestamp: Frame timestamp in seconds
        
        Returns:
            HandsData with detected hands
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp_ms = int(timestamp * 1000)
        if timestamp_ms <= 0:
            timestamp_ms = self._frame_count * 33
        
        try:
            results = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            self.logger.debug(f"Frame {self._frame_count}: Hand detection error: {e}")
            self._frame_count += 1
            return HandsData(frame_number=self._frame_count - 1, timestamp=timestamp)
        
        hands_data = HandsData(
            frame_number=self._frame_count,
            timestamp=timestamp
        )
        
        if results.hand_landmarks and results.handedness:
            for i, (landmarks, handedness_list) in enumerate(
                zip(results.hand_landmarks, results.handedness)
            ):
                handedness = handedness_list[0].category_name.lower()
                
                joints = {}
                for idx in HandLandmarkIndex:
                    if idx < len(landmarks):
                        landmark = landmarks[idx]
                        name = HAND_LANDMARK_NAMES[idx]
                        
                        joints[name] = HandJoint(
                            name=name,
                            index=int(idx),
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z,
                            confidence=1.0
                        )
                
                hand = Hand(
                    handedness=handedness,
                    joints=joints,
                    confidence=handedness_list[0].score
                )
                
                if handedness == "left":
                    hands_data.left_hand = hand
                else:
                    hands_data.right_hand = hand
        
        self._frame_count += 1
        return hands_data
    
    def draw_hands(
        self,
        frame: np.ndarray,
        hands_data: HandsData,
        color_left: Tuple[int, int, int] = (255, 128, 0),
        color_right: Tuple[int, int, int] = (0, 128, 255),
        thickness: int = 2,
        radius: int = 4
    ) -> np.ndarray:
        """Draw hand landmarks on frame."""
        import cv2
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        for hand, color in [
            (hands_data.left_hand, color_left),
            (hands_data.right_hand, color_right)
        ]:
            if hand is None:
                continue
            
            for start_idx, end_idx in HAND_CONNECTIONS:
                start_name = HAND_LANDMARK_NAMES[start_idx]
                end_name = HAND_LANDMARK_NAMES[end_idx]
                
                start_joint = hand.joints.get(start_name)
                end_joint = hand.joints.get(end_name)
                
                if start_joint and end_joint:
                    start_pt = start_joint.pixel_coords(w, h)
                    end_pt = end_joint.pixel_coords(w, h)
                    cv2.line(output, start_pt, end_pt, color, thickness)
            
            for joint in hand.joints.values():
                pt = joint.pixel_coords(w, h)
                cv2.circle(output, pt, radius, color, -1)
                cv2.circle(output, pt, radius + 1, (255, 255, 255), 1)
        
        return output
    
    def close(self) -> None:
        """Release resources."""
        if self._hand_landmarker:
            self._hand_landmarker.close()
        self.logger.info("Hand tracker closed")
