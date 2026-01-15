"""
Halpe 136 Keypoint Definitions.

This module provides comprehensive definitions for all 136 Halpe keypoints:
- 26 Body keypoints (indices 0-25)
- 68 Face keypoints (indices 26-93)
- 21 Left Hand keypoints (indices 94-114)
- 21 Right Hand keypoints (indices 115-135)

Reference: https://github.com/Fang-Haoshu/Halpe-FullBody
"""

from enum import IntEnum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# BODY KEYPOINTS (0-25)
# =============================================================================

class HalpeBodyKeypoint(IntEnum):
    """26 Body keypoints in Halpe format."""
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
    HIP = 19  # Center hip / pelvis
    LEFT_BIG_TOE = 20
    RIGHT_BIG_TOE = 21
    LEFT_SMALL_TOE = 22
    RIGHT_SMALL_TOE = 23
    LEFT_HEEL = 24
    RIGHT_HEEL = 25


HALPE_BODY_KEYPOINT_NAMES: Dict[int, str] = {
    0: "Nose",
    1: "LEye",
    2: "REye",
    3: "LEar",
    4: "REar",
    5: "LShoulder",
    6: "RShoulder",
    7: "LElbow",
    8: "RElbow",
    9: "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "RKnee",
    15: "LAnkle",
    16: "RAnkle",
    17: "Head",
    18: "Neck",
    19: "Hip",
    20: "LBigToe",
    21: "RBigToe",
    22: "LSmallToe",
    23: "RSmallToe",
    24: "LHeel",
    25: "RHeel",
}


# Body skeleton connections for visualization
HALPE_BODY_SKELETON: List[Tuple[int, int]] = [
    # Head
    (17, 18),  # Head -> Neck
    (0, 17),   # Nose -> Head
    (0, 1),    # Nose -> LEye
    (0, 2),    # Nose -> REye
    (1, 3),    # LEye -> LEar
    (2, 4),    # REye -> REar
    # Torso
    (18, 19),  # Neck -> Hip
    (18, 5),   # Neck -> LShoulder
    (18, 6),   # Neck -> RShoulder
    (19, 11),  # Hip -> LHip
    (19, 12),  # Hip -> RHip
    # Left arm
    (5, 7),    # LShoulder -> LElbow
    (7, 9),    # LElbow -> LWrist
    # Right arm
    (6, 8),    # RShoulder -> RElbow
    (8, 10),   # RElbow -> RWrist
    # Left leg
    (11, 13),  # LHip -> LKnee
    (13, 15),  # LKnee -> LAnkle
    (15, 20),  # LAnkle -> LBigToe
    (15, 22),  # LAnkle -> LSmallToe
    (15, 24),  # LAnkle -> LHeel
    # Right leg
    (12, 14),  # RHip -> RKnee
    (14, 16),  # RKnee -> RAnkle
    (16, 21),  # RAnkle -> RBigToe
    (16, 23),  # RAnkle -> RSmallToe
    (16, 25),  # RAnkle -> RHeel
]


# =============================================================================
# FACE KEYPOINTS (26-93) - 68 landmarks
# Based on standard 68-point face landmark model
# =============================================================================

FACE_KEYPOINT_START = 26
FACE_KEYPOINT_END = 93
FACE_KEYPOINT_COUNT = 68


class HalpeFaceKeypoint(IntEnum):
    """68 Face keypoints in Halpe format (indices 26-93).
    
    Face landmark regions:
    - Jaw: 0-16 (17 points)
    - Right eyebrow: 17-21 (5 points)
    - Left eyebrow: 22-26 (5 points)
    - Nose bridge: 27-30 (4 points)
    - Nose bottom: 31-35 (5 points)
    - Right eye: 36-41 (6 points)
    - Left eye: 42-47 (6 points)
    - Outer lip: 48-59 (12 points)
    - Inner lip: 60-67 (8 points)
    """
    # Jaw contour (0-16)
    JAW_0 = 26
    JAW_1 = 27
    JAW_2 = 28
    JAW_3 = 29
    JAW_4 = 30
    JAW_5 = 31
    JAW_6 = 32
    JAW_7 = 33
    JAW_8 = 34  # Chin
    JAW_9 = 35
    JAW_10 = 36
    JAW_11 = 37
    JAW_12 = 38
    JAW_13 = 39
    JAW_14 = 40
    JAW_15 = 41
    JAW_16 = 42
    
    # Right eyebrow (17-21)
    RIGHT_EYEBROW_0 = 43
    RIGHT_EYEBROW_1 = 44
    RIGHT_EYEBROW_2 = 45
    RIGHT_EYEBROW_3 = 46
    RIGHT_EYEBROW_4 = 47
    
    # Left eyebrow (22-26)
    LEFT_EYEBROW_0 = 48
    LEFT_EYEBROW_1 = 49
    LEFT_EYEBROW_2 = 50
    LEFT_EYEBROW_3 = 51
    LEFT_EYEBROW_4 = 52
    
    # Nose bridge (27-30)
    NOSE_BRIDGE_0 = 53
    NOSE_BRIDGE_1 = 54
    NOSE_BRIDGE_2 = 55
    NOSE_BRIDGE_3 = 56
    
    # Nose bottom (31-35)
    NOSE_BOTTOM_0 = 57
    NOSE_BOTTOM_1 = 58
    NOSE_TIP = 59
    NOSE_BOTTOM_3 = 60
    NOSE_BOTTOM_4 = 61
    
    # Right eye (36-41)
    RIGHT_EYE_OUTER = 62
    RIGHT_EYE_1 = 63
    RIGHT_EYE_2 = 64
    RIGHT_EYE_INNER = 65
    RIGHT_EYE_4 = 66
    RIGHT_EYE_5 = 67
    
    # Left eye (42-47)
    LEFT_EYE_INNER = 68
    LEFT_EYE_1 = 69
    LEFT_EYE_2 = 70
    LEFT_EYE_OUTER = 71
    LEFT_EYE_4 = 72
    LEFT_EYE_5 = 73
    
    # Outer lip (48-59)
    MOUTH_OUTER_0 = 74
    MOUTH_OUTER_1 = 75
    MOUTH_OUTER_2 = 76
    MOUTH_OUTER_TOP = 77
    MOUTH_OUTER_4 = 78
    MOUTH_OUTER_5 = 79
    MOUTH_OUTER_6 = 80
    MOUTH_OUTER_7 = 81
    MOUTH_OUTER_8 = 82
    MOUTH_OUTER_BOTTOM = 83
    MOUTH_OUTER_10 = 84
    MOUTH_OUTER_11 = 85
    
    # Inner lip (60-67)
    MOUTH_INNER_0 = 86
    MOUTH_INNER_1 = 87
    MOUTH_INNER_2 = 88
    MOUTH_INNER_3 = 89
    MOUTH_INNER_4 = 90
    MOUTH_INNER_5 = 91
    MOUTH_INNER_6 = 92
    MOUTH_INNER_7 = 93


# Face keypoint names (local index 0-67)
HALPE_FACE_KEYPOINT_NAMES: Dict[int, str] = {
    # Jaw (0-16)
    0: "Jaw_0", 1: "Jaw_1", 2: "Jaw_2", 3: "Jaw_3", 4: "Jaw_4",
    5: "Jaw_5", 6: "Jaw_6", 7: "Jaw_7", 8: "Chin", 9: "Jaw_9",
    10: "Jaw_10", 11: "Jaw_11", 12: "Jaw_12", 13: "Jaw_13", 14: "Jaw_14",
    15: "Jaw_15", 16: "Jaw_16",
    # Right eyebrow (17-21)
    17: "REyebrow_0", 18: "REyebrow_1", 19: "REyebrow_2", 20: "REyebrow_3", 21: "REyebrow_4",
    # Left eyebrow (22-26)
    22: "LEyebrow_0", 23: "LEyebrow_1", 24: "LEyebrow_2", 25: "LEyebrow_3", 26: "LEyebrow_4",
    # Nose bridge (27-30)
    27: "NoseBridge_0", 28: "NoseBridge_1", 29: "NoseBridge_2", 30: "NoseBridge_3",
    # Nose bottom (31-35)
    31: "NoseBottom_0", 32: "NoseBottom_1", 33: "NoseTip", 34: "NoseBottom_3", 35: "NoseBottom_4",
    # Right eye (36-41)
    36: "REye_Outer", 37: "REye_1", 38: "REye_2", 39: "REye_Inner", 40: "REye_4", 41: "REye_5",
    # Left eye (42-47)
    42: "LEye_Inner", 43: "LEye_1", 44: "LEye_2", 45: "LEye_Outer", 46: "LEye_4", 47: "LEye_5",
    # Outer lip (48-59)
    48: "MouthOuter_0", 49: "MouthOuter_1", 50: "MouthOuter_2", 51: "MouthOuter_Top",
    52: "MouthOuter_4", 53: "MouthOuter_5", 54: "MouthOuter_6", 55: "MouthOuter_7",
    56: "MouthOuter_8", 57: "MouthOuter_Bottom", 58: "MouthOuter_10", 59: "MouthOuter_11",
    # Inner lip (60-67)
    60: "MouthInner_0", 61: "MouthInner_1", 62: "MouthInner_2", 63: "MouthInner_3",
    64: "MouthInner_4", 65: "MouthInner_5", 66: "MouthInner_6", 67: "MouthInner_7",
}


# Face skeleton connections for visualization
HALPE_FACE_SKELETON: List[Tuple[int, int]] = [
    # Jaw
    *[(i, i+1) for i in range(16)],
    # Right eyebrow
    *[(i, i+1) for i in range(17, 21)],
    # Left eyebrow
    *[(i, i+1) for i in range(22, 26)],
    # Nose bridge
    *[(i, i+1) for i in range(27, 30)],
    # Nose bottom
    (31, 32), (32, 33), (33, 34), (34, 35), (31, 35),
    # Right eye
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    # Left eye
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Outer lip
    *[(i, i+1) for i in range(48, 59)], (59, 48),
    # Inner lip
    *[(i, i+1) for i in range(60, 67)], (67, 60),
]


# =============================================================================
# HAND KEYPOINTS - 21 landmarks per hand
# Based on MediaPipe hand landmark model
# =============================================================================

LEFT_HAND_KEYPOINT_START = 94
LEFT_HAND_KEYPOINT_END = 114
RIGHT_HAND_KEYPOINT_START = 115
RIGHT_HAND_KEYPOINT_END = 135
HAND_KEYPOINT_COUNT = 21


class HalpeHandKeypoint(IntEnum):
    """21 Hand keypoints (local indices 0-20).
    
    Finger order: Wrist, Thumb (4), Index (4), Middle (4), Ring (4), Pinky (4)
    """
    WRIST = 0
    
    # Thumb
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    
    # Index finger
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    
    # Middle finger
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    
    # Ring finger
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    
    # Pinky finger
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Hand keypoint names (local index 0-20)
HALPE_HAND_KEYPOINT_NAMES: Dict[int, str] = {
    0: "Wrist",
    1: "Thumb_CMC",
    2: "Thumb_MCP",
    3: "Thumb_IP",
    4: "Thumb_Tip",
    5: "Index_MCP",
    6: "Index_PIP",
    7: "Index_DIP",
    8: "Index_Tip",
    9: "Middle_MCP",
    10: "Middle_PIP",
    11: "Middle_DIP",
    12: "Middle_Tip",
    13: "Ring_MCP",
    14: "Ring_PIP",
    15: "Ring_DIP",
    16: "Ring_Tip",
    17: "Pinky_MCP",
    18: "Pinky_PIP",
    19: "Pinky_DIP",
    20: "Pinky_Tip",
}


# Hand skeleton connections for visualization
HALPE_HAND_SKELETON: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]


# =============================================================================
# COMPLETE 136 KEYPOINT DEFINITIONS
# =============================================================================

HALPE_KEYPOINT_COUNT = 136
HALPE_BODY_COUNT = 26
HALPE_FACE_COUNT = 68
HALPE_HAND_COUNT = 21


def get_halpe_keypoint_name(index: int) -> str:
    """Get the name of a Halpe keypoint by its global index (0-135)."""
    if 0 <= index <= 25:
        return HALPE_BODY_KEYPOINT_NAMES.get(index, f"Body_{index}")
    elif 26 <= index <= 93:
        local_idx = index - 26
        return f"Face_{HALPE_FACE_KEYPOINT_NAMES.get(local_idx, str(local_idx))}"
    elif 94 <= index <= 114:
        local_idx = index - 94
        return f"LHand_{HALPE_HAND_KEYPOINT_NAMES.get(local_idx, str(local_idx))}"
    elif 115 <= index <= 135:
        local_idx = index - 115
        return f"RHand_{HALPE_HAND_KEYPOINT_NAMES.get(local_idx, str(local_idx))}"
    else:
        return f"Unknown_{index}"


def get_all_halpe_keypoint_names() -> Dict[int, str]:
    """Get a dictionary of all 136 Halpe keypoint names."""
    names = {}
    for i in range(136):
        names[i] = get_halpe_keypoint_name(i)
    return names


# Complete mapping of all 136 keypoints
HALPE_136_KEYPOINT_NAMES: Dict[int, str] = {
    # Body (0-25)
    0: "Nose", 1: "LEye", 2: "REye", 3: "LEar", 4: "REar",
    5: "LShoulder", 6: "RShoulder", 7: "LElbow", 8: "RElbow",
    9: "LWrist", 10: "RWrist", 11: "LHip", 12: "RHip",
    13: "LKnee", 14: "RKnee", 15: "LAnkle", 16: "RAnkle",
    17: "Head", 18: "Neck", 19: "Hip",
    20: "LBigToe", 21: "RBigToe", 22: "LSmallToe", 23: "RSmallToe",
    24: "LHeel", 25: "RHeel",
    
    # Face (26-93)
    26: "Face_Jaw_0", 27: "Face_Jaw_1", 28: "Face_Jaw_2", 29: "Face_Jaw_3",
    30: "Face_Jaw_4", 31: "Face_Jaw_5", 32: "Face_Jaw_6", 33: "Face_Jaw_7",
    34: "Face_Chin", 35: "Face_Jaw_9", 36: "Face_Jaw_10", 37: "Face_Jaw_11",
    38: "Face_Jaw_12", 39: "Face_Jaw_13", 40: "Face_Jaw_14", 41: "Face_Jaw_15",
    42: "Face_Jaw_16",
    43: "Face_REyebrow_0", 44: "Face_REyebrow_1", 45: "Face_REyebrow_2",
    46: "Face_REyebrow_3", 47: "Face_REyebrow_4",
    48: "Face_LEyebrow_0", 49: "Face_LEyebrow_1", 50: "Face_LEyebrow_2",
    51: "Face_LEyebrow_3", 52: "Face_LEyebrow_4",
    53: "Face_NoseBridge_0", 54: "Face_NoseBridge_1", 55: "Face_NoseBridge_2",
    56: "Face_NoseBridge_3",
    57: "Face_NoseBottom_0", 58: "Face_NoseBottom_1", 59: "Face_NoseTip",
    60: "Face_NoseBottom_3", 61: "Face_NoseBottom_4",
    62: "Face_REye_Outer", 63: "Face_REye_1", 64: "Face_REye_2",
    65: "Face_REye_Inner", 66: "Face_REye_4", 67: "Face_REye_5",
    68: "Face_LEye_Inner", 69: "Face_LEye_1", 70: "Face_LEye_2",
    71: "Face_LEye_Outer", 72: "Face_LEye_4", 73: "Face_LEye_5",
    74: "Face_MouthOuter_0", 75: "Face_MouthOuter_1", 76: "Face_MouthOuter_2",
    77: "Face_MouthOuter_Top", 78: "Face_MouthOuter_4", 79: "Face_MouthOuter_5",
    80: "Face_MouthOuter_6", 81: "Face_MouthOuter_7", 82: "Face_MouthOuter_8",
    83: "Face_MouthOuter_Bottom", 84: "Face_MouthOuter_10", 85: "Face_MouthOuter_11",
    86: "Face_MouthInner_0", 87: "Face_MouthInner_1", 88: "Face_MouthInner_2",
    89: "Face_MouthInner_3", 90: "Face_MouthInner_4", 91: "Face_MouthInner_5",
    92: "Face_MouthInner_6", 93: "Face_MouthInner_7",
    
    # Left Hand (94-114)
    94: "LHand_Wrist",
    95: "LHand_Thumb_CMC", 96: "LHand_Thumb_MCP", 97: "LHand_Thumb_IP", 98: "LHand_Thumb_Tip",
    99: "LHand_Index_MCP", 100: "LHand_Index_PIP", 101: "LHand_Index_DIP", 102: "LHand_Index_Tip",
    103: "LHand_Middle_MCP", 104: "LHand_Middle_PIP", 105: "LHand_Middle_DIP", 106: "LHand_Middle_Tip",
    107: "LHand_Ring_MCP", 108: "LHand_Ring_PIP", 109: "LHand_Ring_DIP", 110: "LHand_Ring_Tip",
    111: "LHand_Pinky_MCP", 112: "LHand_Pinky_PIP", 113: "LHand_Pinky_DIP", 114: "LHand_Pinky_Tip",
    
    # Right Hand (115-135)
    115: "RHand_Wrist",
    116: "RHand_Thumb_CMC", 117: "RHand_Thumb_MCP", 118: "RHand_Thumb_IP", 119: "RHand_Thumb_Tip",
    120: "RHand_Index_MCP", 121: "RHand_Index_PIP", 122: "RHand_Index_DIP", 123: "RHand_Index_Tip",
    124: "RHand_Middle_MCP", 125: "RHand_Middle_PIP", 126: "RHand_Middle_DIP", 127: "RHand_Middle_Tip",
    128: "RHand_Ring_MCP", 129: "RHand_Ring_PIP", 130: "RHand_Ring_DIP", 131: "RHand_Ring_Tip",
    132: "RHand_Pinky_MCP", 133: "RHand_Pinky_PIP", 134: "RHand_Pinky_DIP", 135: "RHand_Pinky_Tip",
}


# =============================================================================
# KEYPOINT GROUPINGS
# =============================================================================

class HalpeKeypointGroup(IntEnum):
    """Groups of keypoints for batch operations."""
    BODY = 0
    FACE = 1
    LEFT_HAND = 2
    RIGHT_HAND = 3


def get_keypoint_group(index: int) -> HalpeKeypointGroup:
    """Get the group a keypoint belongs to."""
    if 0 <= index <= 25:
        return HalpeKeypointGroup.BODY
    elif 26 <= index <= 93:
        return HalpeKeypointGroup.FACE
    elif 94 <= index <= 114:
        return HalpeKeypointGroup.LEFT_HAND
    else:
        return HalpeKeypointGroup.RIGHT_HAND


def get_group_indices(group: HalpeKeypointGroup) -> Tuple[int, int]:
    """Get the start and end indices for a keypoint group."""
    if group == HalpeKeypointGroup.BODY:
        return (0, 25)
    elif group == HalpeKeypointGroup.FACE:
        return (26, 93)
    elif group == HalpeKeypointGroup.LEFT_HAND:
        return (94, 114)
    else:
        return (115, 135)


def global_to_local_index(global_index: int) -> Tuple[HalpeKeypointGroup, int]:
    """Convert global index (0-135) to group and local index."""
    group = get_keypoint_group(global_index)
    start, _ = get_group_indices(group)
    return (group, global_index - start)


def local_to_global_index(group: HalpeKeypointGroup, local_index: int) -> int:
    """Convert group and local index to global index (0-135)."""
    start, end = get_group_indices(group)
    max_local = end - start
    if not 0 <= local_index <= max_local:
        raise ValueError(f"Local index {local_index} out of range for {group.name}")
    return start + local_index


# =============================================================================
# RETARGETING MAPPINGS
# =============================================================================

# Mapping from Halpe body keypoints to Mixamo bone names
HALPE_TO_MIXAMO_BODY: Dict[int, str] = {
    HalpeBodyKeypoint.HIP: "mixamorig:Hips",
    HalpeBodyKeypoint.LEFT_HIP: "mixamorig:LeftUpLeg",
    HalpeBodyKeypoint.RIGHT_HIP: "mixamorig:RightUpLeg",
    HalpeBodyKeypoint.LEFT_KNEE: "mixamorig:LeftLeg",
    HalpeBodyKeypoint.RIGHT_KNEE: "mixamorig:RightLeg",
    HalpeBodyKeypoint.LEFT_ANKLE: "mixamorig:LeftFoot",
    HalpeBodyKeypoint.RIGHT_ANKLE: "mixamorig:RightFoot",
    HalpeBodyKeypoint.LEFT_BIG_TOE: "mixamorig:LeftToeBase",
    HalpeBodyKeypoint.RIGHT_BIG_TOE: "mixamorig:RightToeBase",
    HalpeBodyKeypoint.NECK: "mixamorig:Neck",
    HalpeBodyKeypoint.HEAD: "mixamorig:Head",
    HalpeBodyKeypoint.LEFT_SHOULDER: "mixamorig:LeftShoulder",
    HalpeBodyKeypoint.RIGHT_SHOULDER: "mixamorig:RightShoulder",
    HalpeBodyKeypoint.LEFT_ELBOW: "mixamorig:LeftArm",
    HalpeBodyKeypoint.RIGHT_ELBOW: "mixamorig:RightArm",
    HalpeBodyKeypoint.LEFT_WRIST: "mixamorig:LeftForeArm",
    HalpeBodyKeypoint.RIGHT_WRIST: "mixamorig:RightForeArm",
}


# Mapping from Halpe hand keypoints to Mixamo hand bone names
def get_halpe_to_mixamo_hand(is_left: bool) -> Dict[int, str]:
    """Get mapping from Halpe hand keypoints to Mixamo hand bones."""
    side = "Left" if is_left else "Right"
    offset = LEFT_HAND_KEYPOINT_START if is_left else RIGHT_HAND_KEYPOINT_START
    
    return {
        offset + HalpeHandKeypoint.WRIST: f"mixamorig:{side}Hand",
        offset + HalpeHandKeypoint.THUMB_CMC: f"mixamorig:{side}HandThumb1",
        offset + HalpeHandKeypoint.THUMB_MCP: f"mixamorig:{side}HandThumb2",
        offset + HalpeHandKeypoint.THUMB_IP: f"mixamorig:{side}HandThumb3",
        offset + HalpeHandKeypoint.INDEX_MCP: f"mixamorig:{side}HandIndex1",
        offset + HalpeHandKeypoint.INDEX_PIP: f"mixamorig:{side}HandIndex2",
        offset + HalpeHandKeypoint.INDEX_DIP: f"mixamorig:{side}HandIndex3",
        offset + HalpeHandKeypoint.MIDDLE_MCP: f"mixamorig:{side}HandMiddle1",
        offset + HalpeHandKeypoint.MIDDLE_PIP: f"mixamorig:{side}HandMiddle2",
        offset + HalpeHandKeypoint.MIDDLE_DIP: f"mixamorig:{side}HandMiddle3",
        offset + HalpeHandKeypoint.RING_MCP: f"mixamorig:{side}HandRing1",
        offset + HalpeHandKeypoint.RING_PIP: f"mixamorig:{side}HandRing2",
        offset + HalpeHandKeypoint.RING_DIP: f"mixamorig:{side}HandRing3",
        offset + HalpeHandKeypoint.PINKY_MCP: f"mixamorig:{side}HandPinky1",
        offset + HalpeHandKeypoint.PINKY_PIP: f"mixamorig:{side}HandPinky2",
        offset + HalpeHandKeypoint.PINKY_DIP: f"mixamorig:{side}HandPinky3",
    }


# =============================================================================
# DATA STRUCTURES FOR KEYPOINT DATA
# =============================================================================

@dataclass
class HalpeKeypoint:
    """Single Halpe keypoint with position and confidence."""
    index: int
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0
    
    @property
    def name(self) -> str:
        return HALPE_136_KEYPOINT_NAMES.get(self.index, f"Unknown_{self.index}")
    
    @property
    def group(self) -> HalpeKeypointGroup:
        return get_keypoint_group(self.index)
    
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
        }


@dataclass 
class HalpePose:
    """Complete Halpe pose with all 136 keypoints."""
    keypoints: Dict[int, HalpeKeypoint]
    timestamp: float = 0.0
    frame_number: int = 0
    person_id: int = 0
    
    @property
    def body_keypoints(self) -> Dict[int, HalpeKeypoint]:
        return {k: v for k, v in self.keypoints.items() if 0 <= k <= 25}
    
    @property
    def face_keypoints(self) -> Dict[int, HalpeKeypoint]:
        return {k: v for k, v in self.keypoints.items() if 26 <= k <= 93}
    
    @property
    def left_hand_keypoints(self) -> Dict[int, HalpeKeypoint]:
        return {k: v for k, v in self.keypoints.items() if 94 <= k <= 114}
    
    @property
    def right_hand_keypoints(self) -> Dict[int, HalpeKeypoint]:
        return {k: v for k, v in self.keypoints.items() if 115 <= k <= 135}
    
    def get_keypoint(self, index: int) -> Optional[HalpeKeypoint]:
        return self.keypoints.get(index)
    
    def get_keypoint_by_name(self, name: str) -> Optional[HalpeKeypoint]:
        for idx, kp_name in HALPE_136_KEYPOINT_NAMES.items():
            if kp_name == name:
                return self.keypoints.get(idx)
        return None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "person_id": self.person_id,
            "keypoints": {k: v.to_dict() for k, v in self.keypoints.items()},
        }
    
    def to_flat_array(self) -> List[float]:
        """Convert to flat array [x0, y0, z0, c0, x1, y1, z1, c1, ...]."""
        result = []
        for i in range(136):
            kp = self.keypoints.get(i)
            if kp:
                result.extend([kp.x, kp.y, kp.z, kp.confidence])
            else:
                result.extend([0.0, 0.0, 0.0, 0.0])
        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_all_keypoints():
    """Print all 136 keypoint definitions."""
    print("=" * 60)
    print("HALPE 136 KEYPOINT DEFINITIONS")
    print("=" * 60)
    
    print("\n--- BODY (0-25) ---")
    for i in range(26):
        print(f"  {i:3d}: {HALPE_136_KEYPOINT_NAMES[i]}")
    
    print("\n--- FACE (26-93) ---")
    for i in range(26, 94):
        print(f"  {i:3d}: {HALPE_136_KEYPOINT_NAMES[i]}")
    
    print("\n--- LEFT HAND (94-114) ---")
    for i in range(94, 115):
        print(f"  {i:3d}: {HALPE_136_KEYPOINT_NAMES[i]}")
    
    print("\n--- RIGHT HAND (115-135) ---")
    for i in range(115, 136):
        print(f"  {i:3d}: {HALPE_136_KEYPOINT_NAMES[i]}")


if __name__ == "__main__":
    print_all_keypoints()
