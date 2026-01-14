"""Bridge between Python pose estimation and C++ OpenGL viewer.

This module provides a high-level interface to send animation data
from the pose estimation pipeline to the 3D viewer for real-time preview.
"""

import threading
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

from src.core.mixamo_skeleton import (
    MIXAMO_BONE_NAMES,
    BoneTransform,
    SkeletonFrame,
)
from src.motion.skeleton_converter import MediaPipeToMixamoConverter, AnimationBuffer

# Try to import the C++ viewer module
try:
    import mocap_viewer_py
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False
    print("Warning: mocap_viewer_py module not found. Viewer features disabled.")


class ViewerBridge:
    """Bridge to send animation data to the 3D viewer."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        title: str = "MoCap Animation Preview"
    ):
        """
        Args:
            model_path: Path to Mixamo FBX model to load
            width: Viewer window width
            height: Viewer window height
            title: Window title
        """
        if not VIEWER_AVAILABLE:
            raise RuntimeError("Viewer module not available. Build the viewer first.")
            
        self.viewer = mocap_viewer_py.MoCapViewer(width, height, title)
        self.model_path = model_path
        self.converter = MediaPipeToMixamoConverter(scale_factor=100.0)  # cm units
        self.animation_buffer = AnimationBuffer()
        self._running = False
        self._viewer_thread = None
        
    def initialize(self) -> bool:
        """Initialize the viewer and load model."""
        if not self.viewer.initialize():
            return False
            
        if self.model_path:
            self.viewer.load_model(self.model_path)
            
        return True
    
    def start(self, async_mode: bool = True):
        """Start the viewer.
        
        Args:
            async_mode: If True, use frame-by-frame mode (for macOS compatibility)
        """
        self._running = True
        if async_mode:
            # On macOS, GLFW must run on main thread
            # Use run_async to mark as running, then call run_one_frame in loop
            self.viewer.run_async()
        else:
            self.viewer.run()
    
    def update_frame(self) -> bool:
        """Process one viewer frame. Returns False if viewer should close."""
        return self.viewer.run_one_frame()
            
    def stop(self):
        """Stop the viewer."""
        self._running = False
        self.viewer.stop()
        
    def is_running(self) -> bool:
        """Check if viewer is running."""
        return self.viewer.is_running()
    
    def set_reference_pose(self, landmarks: np.ndarray):
        """Set T-pose reference for the converter.
        
        Args:
            landmarks: (33, 3) MediaPipe world landmarks from T-pose
        """
        self.converter.set_reference_pose(landmarks)
        
    def send_pose(self, landmarks: np.ndarray, timestamp: float = None):
        """Send pose landmarks to viewer.
        
        Args:
            landmarks: (33, 3) MediaPipe world landmarks
            timestamp: Optional timestamp in seconds
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Convert to Mixamo skeleton
        frame = self.converter.convert_frame(landmarks, timestamp)
        
        # Store in buffer
        self.animation_buffer.add_frame(frame)
        
        # Send to viewer
        self._send_frame_to_viewer(frame)
        
    def _send_frame_to_viewer(self, frame: SkeletonFrame):
        """Send a skeleton frame to the viewer."""
        bone_data = {}
        
        for bone_name, transform in frame.bone_transforms.items():
            # Create transform array: [px, py, pz, qw, qx, qy, qz, sx, sy, sz]
            data = [
                float(transform.position[0]),
                float(transform.position[1]),
                float(transform.position[2]),
                float(transform.rotation[0]),  # qw
                float(transform.rotation[1]),  # qx
                float(transform.rotation[2]),  # qy
                float(transform.rotation[3]),  # qz
                float(transform.scale[0]),
                float(transform.scale[1]),
                float(transform.scale[2]),
            ]
            bone_data[bone_name] = data
            
        self.viewer.set_animation_frame(bone_data)
        
    def send_bone_transform(
        self,
        bone_name: str,
        position: np.ndarray,
        rotation: np.ndarray,
        scale: np.ndarray = None
    ):
        """Send a single bone transform to viewer.
        
        Args:
            bone_name: Mixamo bone name (e.g., "mixamorig:Hips")
            position: (3,) position
            rotation: (4,) quaternion (w, x, y, z)
            scale: (3,) scale, defaults to (1,1,1)
        """
        if scale is None:
            scale = np.array([1.0, 1.0, 1.0])
            
        self.viewer.set_bone_transform(
            bone_name,
            float(position[0]), float(position[1]), float(position[2]),
            float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]),
            float(scale[0]), float(scale[1]), float(scale[2])
        )


class RealtimePoseVisualizer:
    """High-level interface for real-time pose visualization."""
    
    def __init__(
        self,
        model_path: str,
        camera_source: int = 0,
    ):
        """
        Args:
            model_path: Path to Mixamo FBX model
            camera_source: Camera index for video capture
        """
        self.model_path = model_path
        self.camera_source = camera_source
        self.bridge = None
        self._pose_estimator = None
        
    def setup(self):
        """Initialize viewer and pose estimation."""
        # Initialize viewer bridge
        self.bridge = ViewerBridge(model_path=self.model_path)
        if not self.bridge.initialize():
            raise RuntimeError("Failed to initialize viewer")
            
        # Start viewer in async mode
        self.bridge.start(async_mode=True)
        
    def run_from_video(self, video_path: str):
        """Process video file and stream animation to viewer.
        
        Args:
            video_path: Path to video file
        """
        import cv2
        from src.pose.estimator_2d import PoseEstimator2D
        
        estimator = PoseEstimator2D()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_time = 1.0 / fps
        
        frame_idx = 0
        reference_set = False
        
        try:
            while cap.isOpened() and self.bridge.is_running():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Estimate pose
                result = estimator.process_frame(frame)
                
                if result and result.world_landmarks is not None:
                    landmarks = result.world_landmarks
                    
                    # Set first frame as reference pose
                    if not reference_set:
                        self.bridge.set_reference_pose(landmarks)
                        reference_set = True
                        
                    # Send pose to viewer
                    timestamp = frame_idx * frame_time
                    self.bridge.send_pose(landmarks, timestamp)
                    
                frame_idx += 1
                
                # Match video framerate
                time.sleep(frame_time)
                
        finally:
            cap.release()
            
    def run_from_camera(self):
        """Run real-time pose estimation from camera."""
        import cv2
        from src.pose.estimator_2d import PoseEstimator2D
        
        estimator = PoseEstimator2D()
        cap = cv2.VideoCapture(self.camera_source)
        
        reference_set = False
        last_time = time.time()
        
        try:
            while cap.isOpened() and self.bridge.is_running():
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Estimate pose
                result = estimator.process_frame(frame)
                
                if result and result.world_landmarks is not None:
                    landmarks = result.world_landmarks
                    
                    # Set first valid frame as reference
                    if not reference_set:
                        self.bridge.set_reference_pose(landmarks)
                        reference_set = True
                        
                    # Send pose to viewer
                    current_time = time.time()
                    self.bridge.send_pose(landmarks, current_time - last_time)
                    
                # Display camera feed (optional)
                cv2.imshow('Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    def cleanup(self):
        """Cleanup resources."""
        if self.bridge:
            self.bridge.stop()
