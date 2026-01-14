"""Face Mesh Tracker using MediaPipe FaceMesh

Provides 468 face landmarks (mapped to 68 standard landmarks for Halpe compatibility).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.core import get_logger, Config


# MediaPipe FaceMesh provides 468 landmarks
# We map to standard 68 face landmarks used in Halpe
# Standard 68 landmarks:
#   0-16: Jaw contour (17 points)
#   17-21: Left eyebrow (5 points)
#   22-26: Right eyebrow (5 points)
#   27-30: Nose bridge (4 points)
#   31-35: Nose tip (5 points)
#   36-41: Left eye (6 points)
#   42-47: Right eye (6 points)
#   48-59: Outer lip (12 points)
#   60-67: Inner lip (8 points)

# MediaPipe to 68-landmark mapping
MEDIAPIPE_TO_68 = {
    # Jaw contour (0-16)
    0: 234, 1: 93, 2: 132, 3: 58, 4: 172, 5: 136, 6: 150, 7: 176, 8: 152,
    9: 400, 10: 379, 11: 365, 12: 288, 13: 361, 14: 323, 15: 454, 16: 356,
    # Left eyebrow (17-21)
    17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
    # Right eyebrow (22-26)
    22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
    # Nose bridge (27-30)
    27: 168, 28: 6, 29: 197, 30: 195,
    # Nose tip (31-35)
    31: 5, 32: 4, 33: 1, 34: 19, 35: 94,
    # Left eye (36-41)
    36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
    # Right eye (42-47)
    42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
    # Outer lip (48-59)
    48: 61, 49: 40, 50: 37, 51: 0, 52: 267, 53: 270, 54: 291,
    55: 321, 56: 314, 57: 17, 58: 84, 59: 91,
    # Inner lip (60-67)
    60: 78, 61: 81, 62: 13, 63: 311, 64: 308, 65: 402, 66: 14, 67: 178,
}


@dataclass
class FaceLandmark:
    """Single face landmark."""
    index: int
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    z: float = 0.0  # Depth
    confidence: float = 1.0
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def pixel_coords(self, width: int, height: int) -> Tuple[int, int]:
        return (int(self.x * width), int(self.y * height))


@dataclass
class FaceData:
    """Face mesh data for a single face."""
    landmarks: Dict[int, FaceLandmark] = field(default_factory=dict)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    confidence: float = 0.0
    
    @property
    def landmarks_68(self) -> Dict[int, FaceLandmark]:
        """Get standard 68 landmarks."""
        return {k: v for k, v in self.landmarks.items() if k < 68}
    
    @property
    def is_valid(self) -> bool:
        return len(self.landmarks) >= 68 and self.confidence > 0.5
    
    def get_landmark(self, index: int) -> Optional[FaceLandmark]:
        return self.landmarks.get(index)


# Face mesh model URLs
FACE_MODELS = {
    "short": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "filename": "face_landmarker.task"
    }
}


def download_face_model_if_needed(model_type: str = "short") -> str:
    """Download face model if not present."""
    import urllib.request
    import os
    
    model_info = FACE_MODELS.get(model_type, FACE_MODELS["short"])
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / model_info["filename"]
    
    if not model_path.exists():
        print(f"Downloading face model: {model_info['filename']}...")
        urllib.request.urlretrieve(model_info["url"], str(model_path))
        print(f"Downloaded to: {model_path}")
    
    return str(model_path)


class FaceTracker:
    """MediaPipe FaceMesh tracker providing 68 standard face landmarks."""
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("pose.face_tracker")
        self.config = config or Config()
        
        face_config = self.config.get("face_tracking", {})
        self._min_detection_confidence = face_config.get("min_detection_confidence", 0.5)
        self._min_tracking_confidence = face_config.get("min_tracking_confidence", 0.5)
        self._max_faces = face_config.get("max_faces", 1)
        
        model_path = download_face_model_if_needed()
        
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=self._max_faces,
            min_face_detection_confidence=self._min_detection_confidence,
            min_face_presence_confidence=self._min_tracking_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._frame_count = 0
        
        self.logger.info(
            f"Initialized FaceTracker (max_faces={self._max_faces}, "
            f"detection={self._min_detection_confidence})"
        )
    
    def process(self, frame: np.ndarray, timestamp: float = 0.0) -> Optional[FaceData]:
        """
        Process frame and extract face landmarks.
        
        Args:
            frame: RGB image (H, W, 3)
            timestamp: Frame timestamp in seconds
        
        Returns:
            FaceData with 68 landmarks or None
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp_ms = int(timestamp * 1000)
        if timestamp_ms <= 0:
            timestamp_ms = self._frame_count * 33
        
        try:
            results = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            self.logger.debug(f"Face detection error: {e}")
            self._frame_count += 1
            return None
        
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            self._frame_count += 1
            return None
        
        # Get first face
        mp_landmarks = results.face_landmarks[0]
        
        # Convert to 68-landmark format
        landmarks = {}
        for std_idx, mp_idx in MEDIAPIPE_TO_68.items():
            if mp_idx < len(mp_landmarks):
                lm = mp_landmarks[mp_idx]
                landmarks[std_idx] = FaceLandmark(
                    index=std_idx,
                    x=lm.x,
                    y=lm.y,
                    z=lm.z if hasattr(lm, 'z') else 0.0,
                    confidence=1.0
                )
        
        # Calculate bounding box from landmarks
        if landmarks:
            xs = [lm.x for lm in landmarks.values()]
            ys = [lm.y for lm in landmarks.values()]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        else:
            bbox = (0, 0, 0, 0)
        
        face_data = FaceData(
            landmarks=landmarks,
            bbox=bbox,
            confidence=1.0
        )
        
        self._frame_count += 1
        return face_data
    
    def draw_face(
        self,
        frame: np.ndarray,
        face_data: FaceData,
        draw_mesh: bool = False,
        draw_contours: bool = True,
        draw_indices: bool = False
    ) -> np.ndarray:
        """Draw face landmarks on frame."""
        import cv2
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        if not face_data or not face_data.landmarks:
            return result
        
        # Face contour connections
        FACE_CONTOURS = [
            # Jaw
            list(range(0, 17)),
            # Left eyebrow
            list(range(17, 22)),
            # Right eyebrow
            list(range(22, 27)),
            # Nose bridge
            list(range(27, 31)),
            # Nose tip
            [31, 32, 33, 34, 35, 31],
            # Left eye
            [36, 37, 38, 39, 40, 41, 36],
            # Right eye
            [42, 43, 44, 45, 46, 47, 42],
            # Outer lip
            [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48],
            # Inner lip
            [60, 61, 62, 63, 64, 65, 66, 67, 60],
        ]
        
        # Draw contours
        if draw_contours:
            for contour in FACE_CONTOURS:
                for i in range(len(contour) - 1):
                    idx1, idx2 = contour[i], contour[i + 1]
                    lm1 = face_data.get_landmark(idx1)
                    lm2 = face_data.get_landmark(idx2)
                    if lm1 and lm2:
                        pt1 = lm1.pixel_coords(w, h)
                        pt2 = lm2.pixel_coords(w, h)
                        cv2.line(result, pt1, pt2, (0, 255, 255), 1)
        
        # Draw all landmarks
        for idx, lm in face_data.landmarks.items():
            if idx < 68:
                pt = lm.pixel_coords(w, h)
                
                # Color code by region
                if idx <= 16:
                    color = (255, 200, 0)  # Jaw - cyan
                elif idx <= 26:
                    color = (0, 255, 0)  # Eyebrows - green
                elif idx <= 35:
                    color = (255, 0, 255)  # Nose - magenta
                elif idx <= 47:
                    color = (255, 255, 0)  # Eyes - cyan
                else:
                    color = (0, 0, 255)  # Lips - red
                
                cv2.circle(result, pt, 2, color, -1)
                
                if draw_indices:
                    cv2.putText(result, str(idx), (pt[0]+2, pt[1]-2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        
        return result
    
    def close(self):
        """Release resources."""
        if hasattr(self, '_face_landmarker'):
            self._face_landmarker.close()
