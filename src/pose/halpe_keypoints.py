"""Halpe Full-Body 136 Keypoints Definition

Halpe provides 136 keypoints total:
- 26 body keypoints (0-25)
- 68 face keypoints (26-93)  
- 21 left hand keypoints (94-114)
- 21 right hand keypoints (115-135)
"""

from enum import IntEnum
from typing import Dict, List, Tuple


class HalpeBodyIndex(IntEnum):
    """Halpe 26 body keypoint indices."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    HEAD = 17
    NECK = 18
    HIP = 19  # Center hip
    LEFT_BIG_TOE = 20
    RIGHT_BIG_TOE = 21
    LEFT_SMALL_TOE = 22
    RIGHT_SMALL_TOE = 23
    LEFT_HEEL = 24
    RIGHT_HEEL = 25


# Body keypoint names
HALPE_BODY_NAMES = {
    HalpeBodyIndex.NOSE: "nose",
    HalpeBodyIndex.LEFT_EYE: "left_eye",
    HalpeBodyIndex.RIGHT_EYE: "right_eye",
    HalpeBodyIndex.LEFT_EAR: "left_ear",
    HalpeBodyIndex.RIGHT_EAR: "right_ear",
    HalpeBodyIndex.LEFT_SHOULDER: "left_shoulder",
    HalpeBodyIndex.RIGHT_SHOULDER: "right_shoulder",
    HalpeBodyIndex.LEFT_ELBOW: "left_elbow",
    HalpeBodyIndex.RIGHT_ELBOW: "right_elbow",
    HalpeBodyIndex.LEFT_WRIST: "left_wrist",
    HalpeBodyIndex.RIGHT_WRIST: "right_wrist",
    HalpeBodyIndex.LEFT_HIP: "left_hip",
    HalpeBodyIndex.RIGHT_HIP: "right_hip",
    HalpeBodyIndex.LEFT_KNEE: "left_knee",
    HalpeBodyIndex.RIGHT_KNEE: "right_knee",
    HalpeBodyIndex.LEFT_ANKLE: "left_ankle",
    HalpeBodyIndex.RIGHT_ANKLE: "right_ankle",
    HalpeBodyIndex.HEAD: "head",
    HalpeBodyIndex.NECK: "neck",
    HalpeBodyIndex.HIP: "hip_center",
    HalpeBodyIndex.LEFT_BIG_TOE: "left_big_toe",
    HalpeBodyIndex.RIGHT_BIG_TOE: "right_big_toe",
    HalpeBodyIndex.LEFT_SMALL_TOE: "left_small_toe",
    HalpeBodyIndex.RIGHT_SMALL_TOE: "right_small_toe",
    HalpeBodyIndex.LEFT_HEEL: "left_heel",
    HalpeBodyIndex.RIGHT_HEEL: "right_heel",
}


# Body skeleton connections for visualization
HALPE_BODY_CONNECTIONS = [
    # Head
    (HalpeBodyIndex.HEAD, HalpeBodyIndex.NOSE),
    (HalpeBodyIndex.NOSE, HalpeBodyIndex.LEFT_EYE),
    (HalpeBodyIndex.NOSE, HalpeBodyIndex.RIGHT_EYE),
    (HalpeBodyIndex.LEFT_EYE, HalpeBodyIndex.LEFT_EAR),
    (HalpeBodyIndex.RIGHT_EYE, HalpeBodyIndex.RIGHT_EAR),
    # Spine
    (HalpeBodyIndex.HEAD, HalpeBodyIndex.NECK),
    (HalpeBodyIndex.NECK, HalpeBodyIndex.HIP),
    (HalpeBodyIndex.HIP, HalpeBodyIndex.LEFT_HIP),
    (HalpeBodyIndex.HIP, HalpeBodyIndex.RIGHT_HIP),
    # Shoulders
    (HalpeBodyIndex.NECK, HalpeBodyIndex.LEFT_SHOULDER),
    (HalpeBodyIndex.NECK, HalpeBodyIndex.RIGHT_SHOULDER),
    # Left arm
    (HalpeBodyIndex.LEFT_SHOULDER, HalpeBodyIndex.LEFT_ELBOW),
    (HalpeBodyIndex.LEFT_ELBOW, HalpeBodyIndex.LEFT_WRIST),
    # Right arm
    (HalpeBodyIndex.RIGHT_SHOULDER, HalpeBodyIndex.RIGHT_ELBOW),
    (HalpeBodyIndex.RIGHT_ELBOW, HalpeBodyIndex.RIGHT_WRIST),
    # Left leg
    (HalpeBodyIndex.LEFT_HIP, HalpeBodyIndex.LEFT_KNEE),
    (HalpeBodyIndex.LEFT_KNEE, HalpeBodyIndex.LEFT_ANKLE),
    (HalpeBodyIndex.LEFT_ANKLE, HalpeBodyIndex.LEFT_HEEL),
    (HalpeBodyIndex.LEFT_ANKLE, HalpeBodyIndex.LEFT_BIG_TOE),
    (HalpeBodyIndex.LEFT_BIG_TOE, HalpeBodyIndex.LEFT_SMALL_TOE),
    # Right leg
    (HalpeBodyIndex.RIGHT_HIP, HalpeBodyIndex.RIGHT_KNEE),
    (HalpeBodyIndex.RIGHT_KNEE, HalpeBodyIndex.RIGHT_ANKLE),
    (HalpeBodyIndex.RIGHT_ANKLE, HalpeBodyIndex.RIGHT_HEEL),
    (HalpeBodyIndex.RIGHT_ANKLE, HalpeBodyIndex.RIGHT_BIG_TOE),
    (HalpeBodyIndex.RIGHT_BIG_TOE, HalpeBodyIndex.RIGHT_SMALL_TOE),
]


class HalpeHandIndex(IntEnum):
    """Hand keypoint indices (relative, 0-20 for each hand)."""
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


# Hand keypoint names
HALPE_HAND_NAMES = {
    HalpeHandIndex.WRIST: "wrist",
    HalpeHandIndex.THUMB_CMC: "thumb_cmc",
    HalpeHandIndex.THUMB_MCP: "thumb_mcp",
    HalpeHandIndex.THUMB_IP: "thumb_ip",
    HalpeHandIndex.THUMB_TIP: "thumb_tip",
    HalpeHandIndex.INDEX_MCP: "index_mcp",
    HalpeHandIndex.INDEX_PIP: "index_pip",
    HalpeHandIndex.INDEX_DIP: "index_dip",
    HalpeHandIndex.INDEX_TIP: "index_tip",
    HalpeHandIndex.MIDDLE_MCP: "middle_mcp",
    HalpeHandIndex.MIDDLE_PIP: "middle_pip",
    HalpeHandIndex.MIDDLE_DIP: "middle_dip",
    HalpeHandIndex.MIDDLE_TIP: "middle_tip",
    HalpeHandIndex.RING_MCP: "ring_mcp",
    HalpeHandIndex.RING_PIP: "ring_pip",
    HalpeHandIndex.RING_DIP: "ring_dip",
    HalpeHandIndex.RING_TIP: "ring_tip",
    HalpeHandIndex.PINKY_MCP: "pinky_mcp",
    HalpeHandIndex.PINKY_PIP: "pinky_pip",
    HalpeHandIndex.PINKY_DIP: "pinky_dip",
    HalpeHandIndex.PINKY_TIP: "pinky_tip",
}


# Hand skeleton connections
HALPE_HAND_CONNECTIONS = [
    # Thumb
    (HalpeHandIndex.WRIST, HalpeHandIndex.THUMB_CMC),
    (HalpeHandIndex.THUMB_CMC, HalpeHandIndex.THUMB_MCP),
    (HalpeHandIndex.THUMB_MCP, HalpeHandIndex.THUMB_IP),
    (HalpeHandIndex.THUMB_IP, HalpeHandIndex.THUMB_TIP),
    # Index
    (HalpeHandIndex.WRIST, HalpeHandIndex.INDEX_MCP),
    (HalpeHandIndex.INDEX_MCP, HalpeHandIndex.INDEX_PIP),
    (HalpeHandIndex.INDEX_PIP, HalpeHandIndex.INDEX_DIP),
    (HalpeHandIndex.INDEX_DIP, HalpeHandIndex.INDEX_TIP),
    # Middle
    (HalpeHandIndex.WRIST, HalpeHandIndex.MIDDLE_MCP),
    (HalpeHandIndex.MIDDLE_MCP, HalpeHandIndex.MIDDLE_PIP),
    (HalpeHandIndex.MIDDLE_PIP, HalpeHandIndex.MIDDLE_DIP),
    (HalpeHandIndex.MIDDLE_DIP, HalpeHandIndex.MIDDLE_TIP),
    # Ring
    (HalpeHandIndex.WRIST, HalpeHandIndex.RING_MCP),
    (HalpeHandIndex.RING_MCP, HalpeHandIndex.RING_PIP),
    (HalpeHandIndex.RING_PIP, HalpeHandIndex.RING_DIP),
    (HalpeHandIndex.RING_DIP, HalpeHandIndex.RING_TIP),
    # Pinky
    (HalpeHandIndex.WRIST, HalpeHandIndex.PINKY_MCP),
    (HalpeHandIndex.PINKY_MCP, HalpeHandIndex.PINKY_PIP),
    (HalpeHandIndex.PINKY_PIP, HalpeHandIndex.PINKY_DIP),
    (HalpeHandIndex.PINKY_DIP, HalpeHandIndex.PINKY_TIP),
    # Palm connections
    (HalpeHandIndex.INDEX_MCP, HalpeHandIndex.MIDDLE_MCP),
    (HalpeHandIndex.MIDDLE_MCP, HalpeHandIndex.RING_MCP),
    (HalpeHandIndex.RING_MCP, HalpeHandIndex.PINKY_MCP),
]


# Face keypoint indices (68 points, indices 26-93 in full Halpe)
# Based on standard 68 face landmark format
FACE_KEYPOINT_START = 26
FACE_KEYPOINT_END = 93

# Face regions
FACE_CONTOUR = list(range(0, 17))  # Jaw line
FACE_LEFT_EYEBROW = list(range(17, 22))
FACE_RIGHT_EYEBROW = list(range(22, 27))
FACE_NOSE_BRIDGE = list(range(27, 31))
FACE_NOSE_TIP = list(range(31, 36))
FACE_LEFT_EYE = list(range(36, 42))
FACE_RIGHT_EYE = list(range(42, 48))
FACE_OUTER_LIP = list(range(48, 60))
FACE_INNER_LIP = list(range(60, 68))

# Hand keypoint offsets in Halpe 136
LEFT_HAND_START = 94
LEFT_HAND_END = 114
RIGHT_HAND_START = 115
RIGHT_HAND_END = 135


def get_halpe_index(keypoint_type: str, local_index: int) -> int:
    """Get global Halpe index from type and local index.
    
    Args:
        keypoint_type: "body", "face", "left_hand", "right_hand"
        local_index: Index within that keypoint group
    
    Returns:
        Global index in Halpe 136 format
    """
    if keypoint_type == "body":
        return local_index
    elif keypoint_type == "face":
        return FACE_KEYPOINT_START + local_index
    elif keypoint_type == "left_hand":
        return LEFT_HAND_START + local_index
    elif keypoint_type == "right_hand":
        return RIGHT_HAND_START + local_index
    else:
        raise ValueError(f"Unknown keypoint type: {keypoint_type}")


# Halpe to Mixamo bone mapping
HALPE_TO_MIXAMO_BODY = {
    # Spine/Torso
    HalpeBodyIndex.HIP: "mixamorig:Hips",
    HalpeBodyIndex.NECK: "mixamorig:Neck",
    HalpeBodyIndex.HEAD: "mixamorig:Head",
    # Left arm
    HalpeBodyIndex.LEFT_SHOULDER: "mixamorig:LeftShoulder",
    HalpeBodyIndex.LEFT_ELBOW: "mixamorig:LeftArm",  # Upper arm ends at elbow
    HalpeBodyIndex.LEFT_WRIST: "mixamorig:LeftForeArm",  # Forearm ends at wrist
    # Right arm
    HalpeBodyIndex.RIGHT_SHOULDER: "mixamorig:RightShoulder",
    HalpeBodyIndex.RIGHT_ELBOW: "mixamorig:RightArm",
    HalpeBodyIndex.RIGHT_WRIST: "mixamorig:RightForeArm",
    # Left leg
    HalpeBodyIndex.LEFT_HIP: "mixamorig:LeftUpLeg",
    HalpeBodyIndex.LEFT_KNEE: "mixamorig:LeftLeg",
    HalpeBodyIndex.LEFT_ANKLE: "mixamorig:LeftFoot",
    # Right leg
    HalpeBodyIndex.RIGHT_HIP: "mixamorig:RightUpLeg",
    HalpeBodyIndex.RIGHT_KNEE: "mixamorig:RightLeg",
    HalpeBodyIndex.RIGHT_ANKLE: "mixamorig:RightFoot",
}


# Left hand Halpe to Mixamo mapping
HALPE_LEFT_HAND_TO_MIXAMO = {
    HalpeHandIndex.WRIST: "mixamorig:LeftHand",
    # Thumb
    HalpeHandIndex.THUMB_CMC: "mixamorig:LeftHandThumb1",
    HalpeHandIndex.THUMB_MCP: "mixamorig:LeftHandThumb2",
    HalpeHandIndex.THUMB_IP: "mixamorig:LeftHandThumb3",
    HalpeHandIndex.THUMB_TIP: "mixamorig:LeftHandThumb4",
    # Index
    HalpeHandIndex.INDEX_MCP: "mixamorig:LeftHandIndex1",
    HalpeHandIndex.INDEX_PIP: "mixamorig:LeftHandIndex2",
    HalpeHandIndex.INDEX_DIP: "mixamorig:LeftHandIndex3",
    HalpeHandIndex.INDEX_TIP: "mixamorig:LeftHandIndex4",
    # Middle
    HalpeHandIndex.MIDDLE_MCP: "mixamorig:LeftHandMiddle1",
    HalpeHandIndex.MIDDLE_PIP: "mixamorig:LeftHandMiddle2",
    HalpeHandIndex.MIDDLE_DIP: "mixamorig:LeftHandMiddle3",
    HalpeHandIndex.MIDDLE_TIP: "mixamorig:LeftHandMiddle4",
    # Ring
    HalpeHandIndex.RING_MCP: "mixamorig:LeftHandRing1",
    HalpeHandIndex.RING_PIP: "mixamorig:LeftHandRing2",
    HalpeHandIndex.RING_DIP: "mixamorig:LeftHandRing3",
    HalpeHandIndex.RING_TIP: "mixamorig:LeftHandRing4",
    # Pinky
    HalpeHandIndex.PINKY_MCP: "mixamorig:LeftHandPinky1",
    HalpeHandIndex.PINKY_PIP: "mixamorig:LeftHandPinky2",
    HalpeHandIndex.PINKY_DIP: "mixamorig:LeftHandPinky3",
    HalpeHandIndex.PINKY_TIP: "mixamorig:LeftHandPinky4",
}


# Right hand Halpe to Mixamo mapping
HALPE_RIGHT_HAND_TO_MIXAMO = {
    HalpeHandIndex.WRIST: "mixamorig:RightHand",
    # Thumb
    HalpeHandIndex.THUMB_CMC: "mixamorig:RightHandThumb1",
    HalpeHandIndex.THUMB_MCP: "mixamorig:RightHandThumb2",
    HalpeHandIndex.THUMB_IP: "mixamorig:RightHandThumb3",
    HalpeHandIndex.THUMB_TIP: "mixamorig:RightHandThumb4",
    # Index
    HalpeHandIndex.INDEX_MCP: "mixamorig:RightHandIndex1",
    HalpeHandIndex.INDEX_PIP: "mixamorig:RightHandIndex2",
    HalpeHandIndex.INDEX_DIP: "mixamorig:RightHandIndex3",
    HalpeHandIndex.INDEX_TIP: "mixamorig:RightHandIndex4",
    # Middle
    HalpeHandIndex.MIDDLE_MCP: "mixamorig:RightHandMiddle1",
    HalpeHandIndex.MIDDLE_PIP: "mixamorig:RightHandMiddle2",
    HalpeHandIndex.MIDDLE_DIP: "mixamorig:RightHandMiddle3",
    HalpeHandIndex.MIDDLE_TIP: "mixamorig:RightHandMiddle4",
    # Ring
    HalpeHandIndex.RING_MCP: "mixamorig:RightHandRing1",
    HalpeHandIndex.RING_PIP: "mixamorig:RightHandRing2",
    HalpeHandIndex.RING_DIP: "mixamorig:RightHandRing3",
    HalpeHandIndex.RING_TIP: "mixamorig:RightHandRing4",
    # Pinky
    HalpeHandIndex.PINKY_MCP: "mixamorig:RightHandPinky1",
    HalpeHandIndex.PINKY_PIP: "mixamorig:RightHandPinky2",
    HalpeHandIndex.PINKY_DIP: "mixamorig:RightHandPinky3",
    HalpeHandIndex.PINKY_TIP: "mixamorig:RightHandPinky4",
}
