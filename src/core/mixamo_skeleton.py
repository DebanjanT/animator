"""Mixamo skeleton bone structure and MediaPipe mapping.

This module defines the Mixamo bone hierarchy and provides mapping
from MediaPipe pose landmarks to Mixamo bones for skeletal animation.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np


class MixamoBone(IntEnum):
    """Essential Mixamo bone indices for full body animation.
    
    Based on standard Mixamo rig with 65 bones. We use the essential
    bones needed for smooth full-body movement (excluding detailed fingers).
    """
    # Root and Spine
    HIPS = 0
    SPINE = 1
    SPINE1 = 2
    SPINE2 = 3
    
    # Head and Neck
    NECK = 4
    HEAD = 5
    
    # Left Arm
    LEFT_SHOULDER = 6
    LEFT_ARM = 7
    LEFT_FOREARM = 8
    LEFT_HAND = 9
    
    # Right Arm
    RIGHT_SHOULDER = 10
    RIGHT_ARM = 11
    RIGHT_FOREARM = 12
    RIGHT_HAND = 13
    
    # Left Leg
    LEFT_UP_LEG = 14
    LEFT_LEG = 15
    LEFT_FOOT = 16
    LEFT_TOE_BASE = 17
    
    # Right Leg
    RIGHT_UP_LEG = 18
    RIGHT_LEG = 19
    RIGHT_FOOT = 20
    RIGHT_TOE_BASE = 21


# Mixamo bone names as they appear in FBX files
MIXAMO_BONE_NAMES = {
    MixamoBone.HIPS: "mixamorig:Hips",
    MixamoBone.SPINE: "mixamorig:Spine",
    MixamoBone.SPINE1: "mixamorig:Spine1",
    MixamoBone.SPINE2: "mixamorig:Spine2",
    MixamoBone.NECK: "mixamorig:Neck",
    MixamoBone.HEAD: "mixamorig:Head",
    MixamoBone.LEFT_SHOULDER: "mixamorig:LeftShoulder",
    MixamoBone.LEFT_ARM: "mixamorig:LeftArm",
    MixamoBone.LEFT_FOREARM: "mixamorig:LeftForeArm",
    MixamoBone.LEFT_HAND: "mixamorig:LeftHand",
    MixamoBone.RIGHT_SHOULDER: "mixamorig:RightShoulder",
    MixamoBone.RIGHT_ARM: "mixamorig:RightArm",
    MixamoBone.RIGHT_FOREARM: "mixamorig:RightForeArm",
    MixamoBone.RIGHT_HAND: "mixamorig:RightHand",
    MixamoBone.LEFT_UP_LEG: "mixamorig:LeftUpLeg",
    MixamoBone.LEFT_LEG: "mixamorig:LeftLeg",
    MixamoBone.LEFT_FOOT: "mixamorig:LeftFoot",
    MixamoBone.LEFT_TOE_BASE: "mixamorig:LeftToeBase",
    MixamoBone.RIGHT_UP_LEG: "mixamorig:RightUpLeg",
    MixamoBone.RIGHT_LEG: "mixamorig:RightLeg",
    MixamoBone.RIGHT_FOOT: "mixamorig:RightFoot",
    MixamoBone.RIGHT_TOE_BASE: "mixamorig:RightToeBase",
}


# Bone parent relationships (child -> parent)
MIXAMO_BONE_PARENTS = {
    MixamoBone.SPINE: MixamoBone.HIPS,
    MixamoBone.SPINE1: MixamoBone.SPINE,
    MixamoBone.SPINE2: MixamoBone.SPINE1,
    MixamoBone.NECK: MixamoBone.SPINE2,
    MixamoBone.HEAD: MixamoBone.NECK,
    MixamoBone.LEFT_SHOULDER: MixamoBone.SPINE2,
    MixamoBone.LEFT_ARM: MixamoBone.LEFT_SHOULDER,
    MixamoBone.LEFT_FOREARM: MixamoBone.LEFT_ARM,
    MixamoBone.LEFT_HAND: MixamoBone.LEFT_FOREARM,
    MixamoBone.RIGHT_SHOULDER: MixamoBone.SPINE2,
    MixamoBone.RIGHT_ARM: MixamoBone.RIGHT_SHOULDER,
    MixamoBone.RIGHT_FOREARM: MixamoBone.RIGHT_ARM,
    MixamoBone.RIGHT_HAND: MixamoBone.RIGHT_FOREARM,
    MixamoBone.LEFT_UP_LEG: MixamoBone.HIPS,
    MixamoBone.LEFT_LEG: MixamoBone.LEFT_UP_LEG,
    MixamoBone.LEFT_FOOT: MixamoBone.LEFT_LEG,
    MixamoBone.LEFT_TOE_BASE: MixamoBone.LEFT_FOOT,
    MixamoBone.RIGHT_UP_LEG: MixamoBone.HIPS,
    MixamoBone.RIGHT_LEG: MixamoBone.RIGHT_UP_LEG,
    MixamoBone.RIGHT_FOOT: MixamoBone.RIGHT_LEG,
    MixamoBone.RIGHT_TOE_BASE: MixamoBone.RIGHT_FOOT,
}


class MediaPipeLandmark(IntEnum):
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


# Mapping from MediaPipe landmarks to Mixamo bones
# Each entry defines which landmarks contribute to a bone's position/rotation
MEDIAPIPE_TO_MIXAMO = {
    # Hips: Average of left and right hip
    MixamoBone.HIPS: {
        "landmarks": [MediaPipeLandmark.LEFT_HIP, MediaPipeLandmark.RIGHT_HIP],
        "method": "average",
    },
    # Spine chain: Interpolate between hips and shoulders
    MixamoBone.SPINE: {
        "landmarks": [MediaPipeLandmark.LEFT_HIP, MediaPipeLandmark.RIGHT_HIP,
                      MediaPipeLandmark.LEFT_SHOULDER, MediaPipeLandmark.RIGHT_SHOULDER],
        "method": "spine_interpolate",
        "weight": 0.33,
    },
    MixamoBone.SPINE1: {
        "landmarks": [MediaPipeLandmark.LEFT_HIP, MediaPipeLandmark.RIGHT_HIP,
                      MediaPipeLandmark.LEFT_SHOULDER, MediaPipeLandmark.RIGHT_SHOULDER],
        "method": "spine_interpolate",
        "weight": 0.5,
    },
    MixamoBone.SPINE2: {
        "landmarks": [MediaPipeLandmark.LEFT_HIP, MediaPipeLandmark.RIGHT_HIP,
                      MediaPipeLandmark.LEFT_SHOULDER, MediaPipeLandmark.RIGHT_SHOULDER],
        "method": "spine_interpolate",
        "weight": 0.67,
    },
    # Neck and Head
    MixamoBone.NECK: {
        "landmarks": [MediaPipeLandmark.LEFT_SHOULDER, MediaPipeLandmark.RIGHT_SHOULDER],
        "method": "average",
    },
    MixamoBone.HEAD: {
        "landmarks": [MediaPipeLandmark.NOSE, MediaPipeLandmark.LEFT_EAR, 
                      MediaPipeLandmark.RIGHT_EAR],
        "method": "head_center",
    },
    # Left Arm
    MixamoBone.LEFT_SHOULDER: {
        "landmarks": [MediaPipeLandmark.LEFT_SHOULDER],
        "method": "direct",
    },
    MixamoBone.LEFT_ARM: {
        "landmarks": [MediaPipeLandmark.LEFT_SHOULDER, MediaPipeLandmark.LEFT_ELBOW],
        "method": "joint",
    },
    MixamoBone.LEFT_FOREARM: {
        "landmarks": [MediaPipeLandmark.LEFT_ELBOW, MediaPipeLandmark.LEFT_WRIST],
        "method": "joint",
    },
    MixamoBone.LEFT_HAND: {
        "landmarks": [MediaPipeLandmark.LEFT_WRIST, MediaPipeLandmark.LEFT_INDEX,
                      MediaPipeLandmark.LEFT_PINKY],
        "method": "hand",
    },
    # Right Arm
    MixamoBone.RIGHT_SHOULDER: {
        "landmarks": [MediaPipeLandmark.RIGHT_SHOULDER],
        "method": "direct",
    },
    MixamoBone.RIGHT_ARM: {
        "landmarks": [MediaPipeLandmark.RIGHT_SHOULDER, MediaPipeLandmark.RIGHT_ELBOW],
        "method": "joint",
    },
    MixamoBone.RIGHT_FOREARM: {
        "landmarks": [MediaPipeLandmark.RIGHT_ELBOW, MediaPipeLandmark.RIGHT_WRIST],
        "method": "joint",
    },
    MixamoBone.RIGHT_HAND: {
        "landmarks": [MediaPipeLandmark.RIGHT_WRIST, MediaPipeLandmark.RIGHT_INDEX,
                      MediaPipeLandmark.RIGHT_PINKY],
        "method": "hand",
    },
    # Left Leg
    MixamoBone.LEFT_UP_LEG: {
        "landmarks": [MediaPipeLandmark.LEFT_HIP, MediaPipeLandmark.LEFT_KNEE],
        "method": "joint",
    },
    MixamoBone.LEFT_LEG: {
        "landmarks": [MediaPipeLandmark.LEFT_KNEE, MediaPipeLandmark.LEFT_ANKLE],
        "method": "joint",
    },
    MixamoBone.LEFT_FOOT: {
        "landmarks": [MediaPipeLandmark.LEFT_ANKLE, MediaPipeLandmark.LEFT_FOOT_INDEX],
        "method": "joint",
    },
    MixamoBone.LEFT_TOE_BASE: {
        "landmarks": [MediaPipeLandmark.LEFT_FOOT_INDEX],
        "method": "direct",
    },
    # Right Leg
    MixamoBone.RIGHT_UP_LEG: {
        "landmarks": [MediaPipeLandmark.RIGHT_HIP, MediaPipeLandmark.RIGHT_KNEE],
        "method": "joint",
    },
    MixamoBone.RIGHT_LEG: {
        "landmarks": [MediaPipeLandmark.RIGHT_KNEE, MediaPipeLandmark.RIGHT_ANKLE],
        "method": "joint",
    },
    MixamoBone.RIGHT_FOOT: {
        "landmarks": [MediaPipeLandmark.RIGHT_ANKLE, MediaPipeLandmark.RIGHT_FOOT_INDEX],
        "method": "joint",
    },
    MixamoBone.RIGHT_TOE_BASE: {
        "landmarks": [MediaPipeLandmark.RIGHT_FOOT_INDEX],
        "method": "direct",
    },
}


@dataclass
class BoneTransform:
    """Represents a bone's transform in world space."""
    position: np.ndarray  # (3,) world position
    rotation: np.ndarray  # (4,) quaternion (w, x, y, z)
    scale: np.ndarray = None  # (3,) scale, defaults to (1,1,1)
    
    def __post_init__(self):
        if self.scale is None:
            self.scale = np.array([1.0, 1.0, 1.0])


@dataclass
class SkeletonFrame:
    """A single frame of skeleton animation data."""
    timestamp: float
    bone_transforms: Dict[str, BoneTransform]  # bone_name -> transform
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "bones": {
                name: {
                    "position": transform.position.tolist(),
                    "rotation": transform.rotation.tolist(),
                    "scale": transform.scale.tolist(),
                }
                for name, transform in self.bone_transforms.items()
            }
        }


def quaternion_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute quaternion rotation from v1 to v2."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    
    if dot < -0.9999:
        # Vectors are opposite, use perpendicular axis
        perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v1, perp)
        axis = axis / np.linalg.norm(axis)
        return np.array([0, axis[0], axis[1], axis[2]])
    
    w = 1 + dot
    quat = np.array([w, cross[0], cross[1], cross[2]])
    return quat / np.linalg.norm(quat)


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create quaternion from axis and angle (radians)."""
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    half_angle = angle / 2
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
