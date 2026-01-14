"""3D Skeleton Viewer - Real-time 3D mannequin visualization"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import cv2

from src.core import get_logger, Config
from src.pose.reconstructor_3d import Pose3D, Joint3D
from src.motion.skeleton_solver import SkeletonPose


UE5_SKELETON_CONNECTIONS = [
    ("pelvis", "spine_01"),
    ("spine_01", "spine_02"),
    ("spine_02", "spine_03"),
    ("spine_03", "neck_01"),
    ("neck_01", "head"),
    
    ("spine_03", "clavicle_l"),
    ("clavicle_l", "upperarm_l"),
    ("upperarm_l", "lowerarm_l"),
    ("lowerarm_l", "hand_l"),
    
    ("spine_03", "clavicle_r"),
    ("clavicle_r", "upperarm_r"),
    ("upperarm_r", "lowerarm_r"),
    ("lowerarm_r", "hand_r"),
    
    ("pelvis", "thigh_l"),
    ("thigh_l", "calf_l"),
    ("calf_l", "foot_l"),
    ("foot_l", "ball_l"),
    
    ("pelvis", "thigh_r"),
    ("thigh_r", "calf_r"),
    ("calf_r", "foot_r"),
    ("foot_r", "ball_r"),
]

MEDIAPIPE_SKELETON_CONNECTIONS = [
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    
    ("left_hip", "left_shoulder"),
    ("right_hip", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_pinky"),
    
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_pinky"),
    
    ("left_shoulder", "nose"),
    ("right_shoulder", "nose"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]

BONE_COLORS = {
    "spine": (255, 255, 0),
    "head": (255, 200, 0),
    "left_arm": (0, 255, 255),
    "right_arm": (255, 0, 255),
    "left_leg": (0, 255, 0),
    "right_leg": (0, 128, 255),
}


class SkeletonViewer3D:
    """
    Real-time 3D skeleton visualization using OpenCV rendering.
    
    Renders a 3D view of the skeleton that can be rotated and viewed
    from different angles.
    """
    
    def __init__(self, config: Optional[Config] = None, width: int = 400, height: int = 400):
        self.logger = get_logger("ui.skeleton3d")
        self.config = config or Config()
        
        self.width = width
        self.height = height
        
        self._rotation_x = 0.0
        self._rotation_y = 0.0
        self._zoom = 1.0
        self._camera_distance = 3.0
        
        self._auto_rotate = True
        self._rotation_speed = 0.5
        
        self.logger.info(f"Initialized 3D skeleton viewer ({width}x{height})")
    
    def render(
        self,
        pose_3d: Optional[Pose3D] = None,
        skeleton_pose: Optional[SkeletonPose] = None,
        background_color: Tuple[int, int, int] = (30, 30, 40)
    ) -> np.ndarray:
        """
        Render 3D skeleton view.
        
        Args:
            pose_3d: 3D pose from MediaPipe
            skeleton_pose: Solved skeleton pose for UE5
            background_color: Background color (RGB)
        
        Returns:
            Rendered image (RGB)
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = background_color
        
        if self._auto_rotate:
            self._rotation_y += self._rotation_speed
            if self._rotation_y > 360:
                self._rotation_y -= 360
        
        if pose_3d and pose_3d.is_valid:
            self._render_mediapipe_skeleton(frame, pose_3d)
        
        self._draw_axes(frame)
        self._draw_ground_grid(frame)
        
        cv2.putText(frame, "3D View", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if self._auto_rotate:
            cv2.putText(frame, f"Rot: {self._rotation_y:.0f}°", (10, self.height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def _render_mediapipe_skeleton(self, frame: np.ndarray, pose_3d: Pose3D) -> None:
        """Render MediaPipe 3D skeleton."""
        points_2d = {}
        
        for name, joint in pose_3d.joints.items():
            x, y, z = joint.x, joint.y, joint.z
            
            px, py = self._project_point(x, y, z)
            points_2d[name] = (px, py, z)
        
        for start_name, end_name in MEDIAPIPE_SKELETON_CONNECTIONS:
            if start_name in points_2d and end_name in points_2d:
                start = points_2d[start_name]
                end = points_2d[end_name]
                
                color = self._get_bone_color(start_name, end_name)
                
                depth = (start[2] + end[2]) / 2
                thickness = max(1, int(3 - depth * 2))
                
                cv2.line(frame, (start[0], start[1]), (end[0], end[1]), color, thickness)
        
        for name, (px, py, z) in points_2d.items():
            radius = max(2, int(5 - z * 3))
            color = (255, 255, 255)
            cv2.circle(frame, (px, py), radius, color, -1)
    
    def _project_point(self, x: float, y: float, z: float) -> Tuple[int, int]:
        """Project 3D point to 2D screen coordinates."""
        y = -y
        
        rot_y_rad = np.radians(self._rotation_y)
        rot_x_rad = np.radians(self._rotation_x)
        
        x_rot = x * np.cos(rot_y_rad) - z * np.sin(rot_y_rad)
        z_rot = x * np.sin(rot_y_rad) + z * np.cos(rot_y_rad)
        
        y_rot = y * np.cos(rot_x_rad) - z_rot * np.sin(rot_x_rad)
        z_final = y * np.sin(rot_x_rad) + z_rot * np.cos(rot_x_rad)
        
        scale = self._zoom * self.width / (self._camera_distance + z_final + 1)
        
        px = int(self.width / 2 + x_rot * scale)
        py = int(self.height / 2 - y_rot * scale)
        
        return (px, py)
    
    def _get_bone_color(self, start_name: str, end_name: str) -> Tuple[int, int, int]:
        """Get color for a bone based on body part."""
        if "left" in start_name.lower() and ("arm" in start_name.lower() or 
            "elbow" in start_name.lower() or "wrist" in start_name.lower() or
            "shoulder" in start_name.lower()):
            return BONE_COLORS["left_arm"]
        elif "right" in start_name.lower() and ("arm" in start_name.lower() or 
            "elbow" in start_name.lower() or "wrist" in start_name.lower() or
            "shoulder" in start_name.lower()):
            return BONE_COLORS["right_arm"]
        elif "left" in start_name.lower() and ("leg" in start_name.lower() or 
            "knee" in start_name.lower() or "ankle" in start_name.lower() or
            "hip" in start_name.lower() or "foot" in start_name.lower()):
            return BONE_COLORS["left_leg"]
        elif "right" in start_name.lower() and ("leg" in start_name.lower() or 
            "knee" in start_name.lower() or "ankle" in start_name.lower() or
            "hip" in start_name.lower() or "foot" in start_name.lower()):
            return BONE_COLORS["right_leg"]
        elif "nose" in start_name.lower() or "eye" in start_name.lower() or "ear" in start_name.lower():
            return BONE_COLORS["head"]
        else:
            return BONE_COLORS["spine"]
    
    def _draw_axes(self, frame: np.ndarray) -> None:
        """Draw coordinate axes."""
        origin = self._project_point(0, 0, 0)
        x_end = self._project_point(0.3, 0, 0)
        y_end = self._project_point(0, 0.3, 0)
        z_end = self._project_point(0, 0, 0.3)
        
        cv2.line(frame, origin, x_end, (255, 0, 0), 2)  # X - Red
        cv2.line(frame, origin, y_end, (0, 255, 0), 2)  # Y - Green
        cv2.line(frame, origin, z_end, (0, 0, 255), 2)  # Z - Blue
        
        cv2.putText(frame, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv2.putText(frame, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(frame, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    def _draw_ground_grid(self, frame: np.ndarray) -> None:
        """Draw ground plane grid."""
        grid_size = 1.0
        grid_steps = 5
        step = grid_size / grid_steps
        
        for i in range(-grid_steps, grid_steps + 1):
            start = self._project_point(i * step, 0, -grid_size)
            end = self._project_point(i * step, 0, grid_size)
            cv2.line(frame, start, end, (60, 60, 60), 1)
            
            start = self._project_point(-grid_size, 0, i * step)
            end = self._project_point(grid_size, 0, i * step)
            cv2.line(frame, start, end, (60, 60, 60), 1)
    
    def set_rotation(self, x: float, y: float) -> None:
        """Set rotation angles."""
        self._rotation_x = x
        self._rotation_y = y
    
    def set_zoom(self, zoom: float) -> None:
        """Set zoom level."""
        self._zoom = max(0.5, min(3.0, zoom))
    
    def toggle_auto_rotate(self) -> bool:
        """Toggle auto-rotation."""
        self._auto_rotate = not self._auto_rotate
        return self._auto_rotate
    
    def rotate(self, dx: float, dy: float) -> None:
        """Rotate view by delta."""
        self._rotation_y += dx
        self._rotation_x += dy
        self._rotation_x = max(-90, min(90, self._rotation_x))
