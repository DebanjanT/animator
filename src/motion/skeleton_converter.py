"""Convert MediaPipe pose landmarks to Mixamo skeleton animation.

This module provides the bridge between pose estimation output and
skeletal animation data that can be sent to the viewer.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.core.mixamo_skeleton import (
    MixamoBone, 
    MIXAMO_BONE_NAMES,
    MIXAMO_BONE_PARENTS,
    MediaPipeLandmark,
    MEDIAPIPE_TO_MIXAMO,
    BoneTransform,
    SkeletonFrame,
    quaternion_from_vectors,
    quaternion_multiply,
)


class MediaPipeToMixamoConverter:
    """Converts MediaPipe pose landmarks to Mixamo skeleton transforms."""
    
    def __init__(self, scale_factor: float = 1.0):
        """
        Args:
            scale_factor: Scale factor to match Mixamo model units (cm vs meters)
        """
        self.scale_factor = scale_factor
        self._reference_pose = None
        self._bone_lengths = {}
        
    def set_reference_pose(self, landmarks: np.ndarray):
        """Set T-pose reference for calculating bone rotations.
        
        Args:
            landmarks: (33, 3) array of MediaPipe landmark positions
        """
        self._reference_pose = landmarks.copy()
        self._calculate_bone_lengths(landmarks)
        
    def _calculate_bone_lengths(self, landmarks: np.ndarray):
        """Calculate reference bone lengths from T-pose."""
        # Arm lengths
        left_upper_arm = np.linalg.norm(
            landmarks[MediaPipeLandmark.LEFT_ELBOW] - 
            landmarks[MediaPipeLandmark.LEFT_SHOULDER]
        )
        left_forearm = np.linalg.norm(
            landmarks[MediaPipeLandmark.LEFT_WRIST] - 
            landmarks[MediaPipeLandmark.LEFT_ELBOW]
        )
        right_upper_arm = np.linalg.norm(
            landmarks[MediaPipeLandmark.RIGHT_ELBOW] - 
            landmarks[MediaPipeLandmark.RIGHT_SHOULDER]
        )
        right_forearm = np.linalg.norm(
            landmarks[MediaPipeLandmark.RIGHT_WRIST] - 
            landmarks[MediaPipeLandmark.RIGHT_ELBOW]
        )
        
        # Leg lengths
        left_thigh = np.linalg.norm(
            landmarks[MediaPipeLandmark.LEFT_KNEE] - 
            landmarks[MediaPipeLandmark.LEFT_HIP]
        )
        left_shin = np.linalg.norm(
            landmarks[MediaPipeLandmark.LEFT_ANKLE] - 
            landmarks[MediaPipeLandmark.LEFT_KNEE]
        )
        right_thigh = np.linalg.norm(
            landmarks[MediaPipeLandmark.RIGHT_KNEE] - 
            landmarks[MediaPipeLandmark.RIGHT_HIP]
        )
        right_shin = np.linalg.norm(
            landmarks[MediaPipeLandmark.RIGHT_ANKLE] - 
            landmarks[MediaPipeLandmark.RIGHT_KNEE]
        )
        
        self._bone_lengths = {
            MixamoBone.LEFT_ARM: left_upper_arm,
            MixamoBone.LEFT_FOREARM: left_forearm,
            MixamoBone.RIGHT_ARM: right_upper_arm,
            MixamoBone.RIGHT_FOREARM: right_forearm,
            MixamoBone.LEFT_UP_LEG: left_thigh,
            MixamoBone.LEFT_LEG: left_shin,
            MixamoBone.RIGHT_UP_LEG: right_thigh,
            MixamoBone.RIGHT_LEG: right_shin,
        }
    
    def convert_frame(self, landmarks: np.ndarray, timestamp: float = 0.0) -> SkeletonFrame:
        """Convert a single frame of MediaPipe landmarks to Mixamo skeleton.
        
        Args:
            landmarks: (33, 3) array of world landmark positions
            timestamp: Frame timestamp in seconds
            
        Returns:
            SkeletonFrame with bone transforms
        """
        bone_transforms = {}
        
        # Calculate hip position (root)
        hip_pos = self._calculate_hip_position(landmarks)
        
        # Process each bone
        for bone in MixamoBone:
            bone_name = MIXAMO_BONE_NAMES[bone]
            transform = self._calculate_bone_transform(bone, landmarks, hip_pos)
            bone_transforms[bone_name] = transform
            
        return SkeletonFrame(timestamp=timestamp, bone_transforms=bone_transforms)
    
    def _calculate_hip_position(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate hip (root) world position."""
        left_hip = landmarks[MediaPipeLandmark.LEFT_HIP]
        right_hip = landmarks[MediaPipeLandmark.RIGHT_HIP]
        return (left_hip + right_hip) / 2 * self.scale_factor
    
    def _calculate_bone_transform(
        self, 
        bone: MixamoBone, 
        landmarks: np.ndarray,
        hip_pos: np.ndarray
    ) -> BoneTransform:
        """Calculate transform for a specific bone."""
        
        mapping = MEDIAPIPE_TO_MIXAMO.get(bone)
        if mapping is None:
            return BoneTransform(
                position=np.zeros(3),
                rotation=np.array([1, 0, 0, 0])
            )
        
        method = mapping["method"]
        landmark_indices = mapping["landmarks"]
        
        # Get landmark positions
        positions = [landmarks[idx] for idx in landmark_indices]
        
        if method == "average":
            position = np.mean(positions, axis=0) * self.scale_factor
            rotation = self._calculate_average_rotation(positions)
            
        elif method == "direct":
            position = positions[0] * self.scale_factor
            rotation = np.array([1, 0, 0, 0])  # Identity
            
        elif method == "joint":
            # For joints, calculate rotation from parent to child direction
            parent_pos = positions[0]
            child_pos = positions[1]
            position = parent_pos * self.scale_factor
            rotation = self._calculate_joint_rotation(bone, parent_pos, child_pos)
            
        elif method == "spine_interpolate":
            weight = mapping.get("weight", 0.5)
            hip_center = (landmarks[MediaPipeLandmark.LEFT_HIP] + 
                         landmarks[MediaPipeLandmark.RIGHT_HIP]) / 2
            shoulder_center = (landmarks[MediaPipeLandmark.LEFT_SHOULDER] + 
                              landmarks[MediaPipeLandmark.RIGHT_SHOULDER]) / 2
            position = (hip_center + (shoulder_center - hip_center) * weight) * self.scale_factor
            rotation = self._calculate_spine_rotation(landmarks, weight)
            
        elif method == "head_center":
            position = positions[0] * self.scale_factor  # Nose
            rotation = self._calculate_head_rotation(landmarks)
            
        elif method == "hand":
            position = positions[0] * self.scale_factor  # Wrist
            rotation = self._calculate_hand_rotation(positions)
            
        else:
            position = np.zeros(3)
            rotation = np.array([1, 0, 0, 0])
        
        # Make position relative to hip for all bones except hips
        if bone != MixamoBone.HIPS:
            position = position - hip_pos
            
        return BoneTransform(position=position, rotation=rotation)
    
    def _calculate_joint_rotation(
        self, 
        bone: MixamoBone,
        parent_pos: np.ndarray, 
        child_pos: np.ndarray
    ) -> np.ndarray:
        """Calculate rotation for a joint bone based on direction to child."""
        direction = child_pos - parent_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Reference direction depends on bone (T-pose orientation)
        if bone in [MixamoBone.LEFT_ARM, MixamoBone.LEFT_FOREARM]:
            ref_dir = np.array([1, 0, 0])  # Left arm points right in T-pose
        elif bone in [MixamoBone.RIGHT_ARM, MixamoBone.RIGHT_FOREARM]:
            ref_dir = np.array([-1, 0, 0])  # Right arm points left in T-pose
        elif bone in [MixamoBone.LEFT_UP_LEG, MixamoBone.LEFT_LEG,
                      MixamoBone.RIGHT_UP_LEG, MixamoBone.RIGHT_LEG]:
            ref_dir = np.array([0, -1, 0])  # Legs point down in T-pose
        elif bone in [MixamoBone.LEFT_FOOT, MixamoBone.RIGHT_FOOT]:
            ref_dir = np.array([0, 0, 1])  # Feet point forward
        else:
            ref_dir = np.array([0, 1, 0])  # Default up
            
        return quaternion_from_vectors(ref_dir, direction)
    
    def _calculate_spine_rotation(self, landmarks: np.ndarray, weight: float) -> np.ndarray:
        """Calculate spine bone rotation based on hip and shoulder orientation."""
        # Calculate forward direction from hip/shoulder plane
        left_hip = landmarks[MediaPipeLandmark.LEFT_HIP]
        right_hip = landmarks[MediaPipeLandmark.RIGHT_HIP]
        left_shoulder = landmarks[MediaPipeLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[MediaPipeLandmark.RIGHT_SHOULDER]
        
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # Up direction
        up = shoulder_center - hip_center
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # Right direction (from left to right)
        hip_right = right_hip - left_hip
        shoulder_right = right_shoulder - left_shoulder
        right = (hip_right * (1 - weight) + shoulder_right * weight)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Forward direction
        forward = np.cross(right, up)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Create rotation from reference (Y-up, Z-forward)
        return quaternion_from_vectors(np.array([0, 1, 0]), up)
    
    def _calculate_head_rotation(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate head rotation from face landmarks."""
        nose = landmarks[MediaPipeLandmark.NOSE]
        left_ear = landmarks[MediaPipeLandmark.LEFT_EAR]
        right_ear = landmarks[MediaPipeLandmark.RIGHT_EAR]
        
        # Forward direction (from ear line to nose)
        ear_center = (left_ear + right_ear) / 2
        forward = nose - ear_center
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Right direction
        right = right_ear - left_ear
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Up direction
        up = np.cross(forward, right)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        return quaternion_from_vectors(np.array([0, 0, 1]), forward)
    
    def _calculate_hand_rotation(self, positions: List[np.ndarray]) -> np.ndarray:
        """Calculate hand rotation from wrist and finger landmarks."""
        wrist = positions[0]
        index = positions[1]
        pinky = positions[2]
        
        # Forward direction (wrist to fingers)
        forward = ((index + pinky) / 2) - wrist
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Right direction (pinky to index)
        right = index - pinky
        right = right / (np.linalg.norm(right) + 1e-8)
        
        return quaternion_from_vectors(np.array([0, 0, 1]), forward)
    
    def _calculate_average_rotation(self, positions: List[np.ndarray]) -> np.ndarray:
        """Calculate average rotation from multiple positions."""
        # For now, return identity rotation
        return np.array([1, 0, 0, 0])


class AnimationBuffer:
    """Buffer for storing animation frames for streaming to viewer."""
    
    def __init__(self, max_frames: int = 300):
        """
        Args:
            max_frames: Maximum frames to keep in buffer (default 10 seconds at 30fps)
        """
        self.max_frames = max_frames
        self.frames: List[SkeletonFrame] = []
        
    def add_frame(self, frame: SkeletonFrame):
        """Add a frame to the buffer."""
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
            
    def get_latest_frame(self) -> Optional[SkeletonFrame]:
        """Get the most recent frame."""
        return self.frames[-1] if self.frames else None
    
    def clear(self):
        """Clear all frames."""
        self.frames.clear()
        
    def to_animation_data(self) -> dict:
        """Convert buffer to animation data format for viewer."""
        return {
            "frame_count": len(self.frames),
            "frames": [frame.to_dict() for frame in self.frames]
        }
