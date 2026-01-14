"""Floor Plane Detection using RANSAC on foot joint positions"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from src.core import get_logger, Config
from src.pose.reconstructor_3d import Pose3D


@dataclass
class FloorPlane:
    """
    Detected floor plane in 3D space.
    
    Plane equation: ax + by + cz + d = 0
    Normal vector: (a, b, c)
    """
    normal: np.ndarray  # (3,) unit normal vector
    d: float  # Distance from origin
    confidence: float  # 0-1 based on inlier ratio
    inlier_count: int
    
    @property
    def a(self) -> float:
        return float(self.normal[0])
    
    @property
    def b(self) -> float:
        return float(self.normal[1])
    
    @property
    def c(self) -> float:
        return float(self.normal[2])
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Signed distance from point to plane."""
        return float(np.dot(self.normal, point) + self.d)
    
    def project_point(self, point: np.ndarray) -> np.ndarray:
        """Project point onto plane."""
        dist = self.distance_to_point(point)
        return point - dist * self.normal
    
    def is_point_below(self, point: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if point is below (or on) the floor plane."""
        return self.distance_to_point(point) < threshold
    
    def get_transform_to_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get rotation and translation to transform floor to Y=0.
        
        Returns:
            (rotation_matrix, translation_vector)
        """
        world_up = np.array([0, 1, 0], dtype=np.float32)
        
        floor_up = self.normal
        if floor_up[1] < 0:
            floor_up = -floor_up
        
        v = np.cross(floor_up, world_up)
        c = np.dot(floor_up, world_up)
        
        if np.linalg.norm(v) < 1e-6:
            rotation = np.eye(3, dtype=np.float32)
        else:
            s = np.linalg.norm(v)
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], dtype=np.float32)
            rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        
        translation = np.array([0, -self.d, 0], dtype=np.float32)
        
        return rotation.astype(np.float32), translation


class FloorDetector:
    """
    Floor plane detection using RANSAC on foot joint positions.
    
    Collects foot positions over time and fits a plane using RANSAC.
    """
    
    FOOT_JOINTS = [
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("motion.floor")
        self.config = config or Config()
        
        floor_config = self.config.floor_detection
        
        self._enabled = floor_config.get("enabled", True)
        self._ransac_iterations = floor_config.get("ransac_iterations", 100)
        self._ransac_threshold = floor_config.get("ransac_threshold", 0.02)
        self._min_samples = floor_config.get("min_samples", 10)
        self._update_interval = floor_config.get("update_interval", 30)
        
        self._foot_positions: List[np.ndarray] = []
        self._max_positions = 500
        self._current_plane: Optional[FloorPlane] = None
        self._frame_count = 0
        self._last_update_frame = -self._update_interval
        
        self.logger.info(
            f"Initialized floor detector (RANSAC iters={self._ransac_iterations}, "
            f"threshold={self._ransac_threshold}m)"
        )
    
    def update(self, pose: Pose3D) -> Optional[FloorPlane]:
        """
        Update floor detection with new pose.
        
        Args:
            pose: 3D pose with foot joints
        
        Returns:
            Updated FloorPlane or None
        """
        if not self._enabled:
            return self._current_plane
        
        for joint_name in self.FOOT_JOINTS:
            joint = pose.joints.get(joint_name)
            if joint and joint.is_visible:
                self._foot_positions.append(joint.position.copy())
        
        if len(self._foot_positions) > self._max_positions:
            self._foot_positions = self._foot_positions[-self._max_positions:]
        
        self._frame_count += 1
        
        should_update = (
            len(self._foot_positions) >= self._min_samples and
            (self._frame_count - self._last_update_frame) >= self._update_interval
        )
        
        if should_update:
            self._current_plane = self._fit_plane_ransac()
            self._last_update_frame = self._frame_count
            
            if self._current_plane:
                self.logger.debug(
                    f"Floor updated: normal={self._current_plane.normal}, "
                    f"confidence={self._current_plane.confidence:.2f}"
                )
        
        return self._current_plane
    
    def _fit_plane_ransac(self) -> Optional[FloorPlane]:
        """Fit plane to foot positions using RANSAC."""
        points = np.array(self._foot_positions)
        n_points = len(points)
        
        if n_points < 3:
            return None
        
        best_plane = None
        best_inliers = 0
        
        for _ in range(self._ransac_iterations):
            indices = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[indices]
            
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            norm_length = np.linalg.norm(normal)
            if norm_length < 1e-6:
                continue
            
            normal = normal / norm_length
            d = -np.dot(normal, p1)
            
            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.sum(distances < self._ransac_threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal, d)
        
        if best_plane is None:
            return None
        
        normal, d = best_plane
        
        distances = np.abs(np.dot(points, normal) + d)
        inlier_mask = distances < self._ransac_threshold
        inlier_points = points[inlier_mask]
        
        if len(inlier_points) >= 3:
            centroid = np.mean(inlier_points, axis=0)
            centered = inlier_points - centroid
            
            _, _, vh = np.linalg.svd(centered)
            normal = vh[2]
            d = -np.dot(normal, centroid)
        
        if normal[1] < 0:
            normal = -normal
            d = -d
        
        confidence = best_inliers / n_points
        
        return FloorPlane(
            normal=normal.astype(np.float32),
            d=float(d),
            confidence=float(confidence),
            inlier_count=int(best_inliers)
        )
    
    def get_floor_y(self, x: float = 0, z: float = 0) -> float:
        """Get floor Y coordinate at given X, Z position."""
        if self._current_plane is None:
            return 0.0
        
        plane = self._current_plane
        if abs(plane.b) < 1e-6:
            return 0.0
        
        y = -(plane.a * x + plane.c * z + plane.d) / plane.b
        return float(y)
    
    def transform_pose_to_floor(self, pose: Pose3D) -> Pose3D:
        """
        Transform pose so floor is at Y=0.
        
        Args:
            pose: Original 3D pose
        
        Returns:
            Transformed pose with floor at Y=0
        """
        if self._current_plane is None:
            return pose
        
        rotation, translation = self._current_plane.get_transform_to_world()
        
        from src.pose.reconstructor_3d import Joint3D
        
        new_joints = {}
        for name, joint in pose.joints.items():
            pos = joint.position
            new_pos = rotation @ pos + translation
            
            new_joints[name] = Joint3D(
                name=name,
                index=joint.index,
                x=float(new_pos[0]),
                y=float(new_pos[1]),
                z=float(new_pos[2]),
                confidence=joint.confidence
            )
        
        return Pose3D(
            frame_number=pose.frame_number,
            timestamp=pose.timestamp,
            joints=new_joints
        )
    
    def is_foot_grounded(self, pose: Pose3D, foot: str = "left") -> bool:
        """
        Check if foot is on the ground.
        
        Args:
            pose: 3D pose
            foot: "left" or "right"
        
        Returns:
            True if foot is within threshold of floor
        """
        if self._current_plane is None:
            return False
        
        ankle = pose.joints.get(f"{foot}_ankle")
        if ankle is None:
            return False
        
        distance = abs(self._current_plane.distance_to_point(ankle.position))
        return distance < self._ransac_threshold * 2
    
    def force_update(self) -> Optional[FloorPlane]:
        """Force immediate floor plane update."""
        if len(self._foot_positions) >= 3:
            self._current_plane = self._fit_plane_ransac()
            self._last_update_frame = self._frame_count
        return self._current_plane
    
    def reset(self) -> None:
        """Reset floor detection state."""
        self._foot_positions.clear()
        self._current_plane = None
        self._frame_count = 0
        self._last_update_frame = -self._update_interval
    
    @property
    def current_plane(self) -> Optional[FloorPlane]:
        return self._current_plane
    
    @property
    def has_valid_floor(self) -> bool:
        return self._current_plane is not None and self._current_plane.confidence > 0.5
    
    @property
    def sample_count(self) -> int:
        return len(self._foot_positions)
