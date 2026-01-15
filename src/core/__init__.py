"""Core systems - config, logging, timing, skeleton"""

from .config import Config
from .logging import setup_logging, get_logger
from .timing import FrameTimer, FrameClock
from .mixamo_skeleton import (
    MixamoBone, 
    MIXAMO_BONE_NAMES,
    MIXAMO_BONE_PARENTS,
    MediaPipeLandmark,
    MEDIAPIPE_TO_MIXAMO,
    BoneTransform,
    SkeletonFrame,
)
from .halpe_keypoints import (
    # Constants
    HALPE_KEYPOINT_COUNT,
    HALPE_BODY_COUNT,
    HALPE_FACE_COUNT,
    HALPE_HAND_COUNT,
    FACE_KEYPOINT_START,
    FACE_KEYPOINT_END,
    LEFT_HAND_KEYPOINT_START,
    LEFT_HAND_KEYPOINT_END,
    RIGHT_HAND_KEYPOINT_START,
    RIGHT_HAND_KEYPOINT_END,
    # Enums
    HalpeBodyKeypoint,
    HalpeFaceKeypoint,
    HalpeHandKeypoint,
    HalpeKeypointGroup,
    # Name mappings
    HALPE_136_KEYPOINT_NAMES,
    HALPE_BODY_KEYPOINT_NAMES,
    HALPE_FACE_KEYPOINT_NAMES,
    HALPE_HAND_KEYPOINT_NAMES,
    # Skeleton connections
    HALPE_BODY_SKELETON,
    HALPE_FACE_SKELETON,
    HALPE_HAND_SKELETON,
    # Retargeting
    HALPE_TO_MIXAMO_BODY,
    get_halpe_to_mixamo_hand,
    # Utility functions
    get_halpe_keypoint_name,
    get_all_halpe_keypoint_names,
    get_keypoint_group,
    get_group_indices,
    global_to_local_index,
    local_to_global_index,
    # Data structures
    HalpeKeypoint,
    HalpePose,
)

__all__ = [
    "Config", "setup_logging", "get_logger", "FrameTimer", "FrameClock",
    "MixamoBone", "MIXAMO_BONE_NAMES", "MIXAMO_BONE_PARENTS",
    "MediaPipeLandmark", "MEDIAPIPE_TO_MIXAMO",
    "BoneTransform", "SkeletonFrame",
    # Halpe keypoints
    "HALPE_KEYPOINT_COUNT", "HALPE_BODY_COUNT", "HALPE_FACE_COUNT", "HALPE_HAND_COUNT",
    "FACE_KEYPOINT_START", "FACE_KEYPOINT_END",
    "LEFT_HAND_KEYPOINT_START", "LEFT_HAND_KEYPOINT_END",
    "RIGHT_HAND_KEYPOINT_START", "RIGHT_HAND_KEYPOINT_END",
    "HalpeBodyKeypoint", "HalpeFaceKeypoint", "HalpeHandKeypoint", "HalpeKeypointGroup",
    "HALPE_136_KEYPOINT_NAMES", "HALPE_BODY_KEYPOINT_NAMES",
    "HALPE_FACE_KEYPOINT_NAMES", "HALPE_HAND_KEYPOINT_NAMES",
    "HALPE_BODY_SKELETON", "HALPE_FACE_SKELETON", "HALPE_HAND_SKELETON",
    "HALPE_TO_MIXAMO_BODY", "get_halpe_to_mixamo_hand",
    "get_halpe_keypoint_name", "get_all_halpe_keypoint_names",
    "get_keypoint_group", "get_group_indices",
    "global_to_local_index", "local_to_global_index",
    "HalpeKeypoint", "HalpePose",
]
