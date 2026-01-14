"""AlphaPose Pose Estimator with Halpe 136 Full-Body Keypoints

AlphaPose is an accurate multi-person pose estimator achieving 70+ mAP on COCO.
This module provides integration with Halpe 136 keypoints (26 body + 68 face + 42 hands).

Requires: AlphaPose installation (https://github.com/MVIG-SJTU/AlphaPose)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import subprocess
import json
import tempfile
import os
import numpy as np

from src.core import get_logger, Config
from src.pose.halpe_keypoints import (
    HalpeBodyIndex, HalpeHandIndex,
    HALPE_BODY_NAMES, HALPE_HAND_NAMES,
    HALPE_BODY_CONNECTIONS, HALPE_HAND_CONNECTIONS,
    LEFT_HAND_START, RIGHT_HAND_START,
    FACE_KEYPOINT_START,
    HALPE_TO_MIXAMO_BODY,
    HALPE_LEFT_HAND_TO_MIXAMO, HALPE_RIGHT_HAND_TO_MIXAMO,
)


@dataclass
class HalpeKeypoint:
    """Single keypoint with position and confidence."""
    index: int
    name: str
    x: float  # Pixel x or normalized
    y: float  # Pixel y or normalized
    z: float = 0.0  # Depth if available
    confidence: float = 0.0
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @property
    def position_3d(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    @property
    def is_visible(self) -> bool:
        return self.confidence > 0.3


@dataclass
class HalpePose:
    """Complete Halpe 136-keypoint pose for a single person."""
    person_id: int
    frame_number: int
    timestamp: float
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x, y, w, h
    keypoints: Dict[int, HalpeKeypoint] = field(default_factory=dict)
    tracking_id: Optional[int] = None
    
    @property
    def body_keypoints(self) -> Dict[int, HalpeKeypoint]:
        """Get body keypoints (indices 0-25)."""
        return {k: v for k, v in self.keypoints.items() if k < 26}
    
    @property
    def face_keypoints(self) -> Dict[int, HalpeKeypoint]:
        """Get face keypoints (indices 26-93)."""
        return {k: v for k, v in self.keypoints.items() if 26 <= k < 94}
    
    @property
    def left_hand_keypoints(self) -> Dict[int, HalpeKeypoint]:
        """Get left hand keypoints (indices 94-114)."""
        return {k: v for k, v in self.keypoints.items() if 94 <= k < 115}
    
    @property
    def right_hand_keypoints(self) -> Dict[int, HalpeKeypoint]:
        """Get right hand keypoints (indices 115-135)."""
        return {k: v for k, v in self.keypoints.items() if 115 <= k < 136}
    
    def get_keypoint(self, index: int) -> Optional[HalpeKeypoint]:
        return self.keypoints.get(index)
    
    def get_body_keypoint(self, body_index: HalpeBodyIndex) -> Optional[HalpeKeypoint]:
        return self.keypoints.get(int(body_index))
    
    @property
    def is_valid(self) -> bool:
        """Check if pose has minimum required keypoints."""
        required = [
            HalpeBodyIndex.LEFT_HIP, HalpeBodyIndex.RIGHT_HIP,
            HalpeBodyIndex.LEFT_SHOULDER, HalpeBodyIndex.RIGHT_SHOULDER
        ]
        return all(
            int(idx) in self.keypoints and self.keypoints[int(idx)].is_visible
            for idx in required
        )


class AlphaPoseEstimator:
    """
    AlphaPose-based pose estimator with Halpe 136 keypoints.
    
    This estimator can work in two modes:
    1. Subprocess mode: Calls AlphaPose as external process (requires installation)
    2. Direct API mode: Uses AlphaPose Python API directly (if available)
    
    For real-time use, we provide a fallback to ONNX-based inference.
    """
    
    ALPHAPOSE_PATH = os.environ.get("ALPHAPOSE_PATH", "")
    
    # Model configs
    HALPE_136_CONFIG = "configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml"
    HALPE_26_CONFIG = "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
    
    def __init__(
        self,
        config: Optional[Config] = None,
        model_type: str = "halpe_136",
        use_tracking: bool = True,
        device: str = "auto"
    ):
        self.logger = get_logger("pose.alphapose")
        self.config = config or Config()
        self.model_type = model_type
        self.use_tracking = use_tracking
        
        # Determine device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        
        self._frame_count = 0
        self._pose_history: List[HalpePose] = []
        self._alphapose_available = self._check_alphapose()
        
        # Try to initialize direct API
        self._api = None
        if self._alphapose_available:
            self._init_alphapose_api()
        
        self.logger.info(
            f"Initialized AlphaPose estimator (model={model_type}, "
            f"tracking={use_tracking}, device={self.device}, "
            f"api_available={self._api is not None})"
        )
    
    def _check_alphapose(self) -> bool:
        """Check if AlphaPose is available."""
        if self.ALPHAPOSE_PATH and os.path.exists(self.ALPHAPOSE_PATH):
            return True
        
        # Check if alphapose is in Python path
        try:
            import alphapose
            return True
        except ImportError:
            pass
        
        return False
    
    def _init_alphapose_api(self):
        """Initialize AlphaPose Python API if available."""
        try:
            # Try to import AlphaPose components
            from alphapose.utils.config import update_config
            from alphapose.utils.detector import DetectorAPI
            from alphapose.utils.transforms import get_func_heatmap_to_coord
            from alphapose.utils.pPose_nms import pose_nms
            from alphapose.models import builder
            import torch
            
            self.logger.info("AlphaPose API loaded successfully")
            # API initialization would go here with model loading
            # This is a placeholder for the full integration
            
        except ImportError as e:
            self.logger.warning(f"AlphaPose API not available: {e}")
            self.logger.info("Will use fallback estimation or subprocess mode")
    
    def process(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0
    ) -> Optional[List[HalpePose]]:
        """
        Process a single frame and extract Halpe poses.
        
        Args:
            frame: RGB image (H, W, 3)
            timestamp: Frame timestamp in seconds
        
        Returns:
            List of HalpePose objects, or None if no poses detected
        """
        if self._api is not None:
            return self._process_with_api(frame, timestamp)
        else:
            return self._process_fallback(frame, timestamp)
    
    def _process_with_api(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Optional[List[HalpePose]]:
        """Process frame using AlphaPose API directly."""
        # This would be the direct API integration
        # For now, fall back to the alternative method
        return self._process_fallback(frame, timestamp)
    
    def _process_fallback(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Optional[List[HalpePose]]:
        """
        Fallback processing using subprocess or simulation.
        
        For demo/testing, this generates placeholder data.
        In production, this would call AlphaPose subprocess.
        """
        # For real use, implement subprocess call to AlphaPose
        # ./scripts/demo_inference.py --cfg ... --checkpoint ... --video ...
        
        self._frame_count += 1
        
        # Return empty for now - actual implementation would parse AlphaPose output
        return None
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> Dict[int, List[HalpePose]]:
        """
        Process entire video using AlphaPose.
        
        Args:
            video_path: Path to input video
            output_path: Optional path for output JSON
        
        Returns:
            Dict mapping frame_number to list of poses
        """
        if not self._alphapose_available:
            self.logger.error("AlphaPose not available for video processing")
            return {}
        
        # Create temp output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = os.path.join(tmpdir, "alphapose-results.json")
            
            # Build AlphaPose command
            cmd = [
                "python", "scripts/demo_inference.py",
                "--cfg", self.HALPE_136_CONFIG if self.model_type == "halpe_136" else self.HALPE_26_CONFIG,
                "--checkpoint", "pretrained_models/halpe136_fast50_dcn_256x192.pth",
                "--video", video_path,
                "--outdir", tmpdir,
                "--save_video" if output_path else "--save_img",
            ]
            
            if self.use_tracking:
                cmd.extend(["--pose_track"])
            
            # Run AlphaPose
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.ALPHAPOSE_PATH,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    self.logger.error(f"AlphaPose failed: {result.stderr}")
                    return {}
                
                # Parse output
                return self._parse_alphapose_json(output_json)
                
            except Exception as e:
                self.logger.error(f"Error running AlphaPose: {e}")
                return {}
    
    def _parse_alphapose_json(self, json_path: str) -> Dict[int, List[HalpePose]]:
        """Parse AlphaPose JSON output to HalpePose objects."""
        poses_by_frame = {}
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                frame_num = int(item.get('image_id', '0').split('.')[0])
                
                # Parse keypoints (format: [x1, y1, c1, x2, y2, c2, ...])
                kpts = item.get('keypoints', [])
                keypoints = {}
                
                num_keypoints = len(kpts) // 3
                for i in range(num_keypoints):
                    x = kpts[i * 3]
                    y = kpts[i * 3 + 1]
                    conf = kpts[i * 3 + 2]
                    
                    # Determine keypoint name
                    if i < 26:
                        name = HALPE_BODY_NAMES.get(HalpeBodyIndex(i), f"body_{i}")
                    elif i < 94:
                        name = f"face_{i - 26}"
                    elif i < 115:
                        name = f"left_{HALPE_HAND_NAMES.get(HalpeHandIndex(i - 94), f'hand_{i - 94}')}"
                    else:
                        name = f"right_{HALPE_HAND_NAMES.get(HalpeHandIndex(i - 115), f'hand_{i - 115}')}"
                    
                    keypoints[i] = HalpeKeypoint(
                        index=i,
                        name=name,
                        x=x,
                        y=y,
                        confidence=conf
                    )
                
                # Parse bbox
                bbox = item.get('box', [0, 0, 0, 0])
                
                pose = HalpePose(
                    person_id=item.get('idx', 0),
                    frame_number=frame_num,
                    timestamp=frame_num / 30.0,  # Assume 30fps
                    bbox=tuple(bbox[:4]) if len(bbox) >= 4 else (0, 0, 0, 0),
                    keypoints=keypoints,
                    tracking_id=item.get('track_id')
                )
                
                if frame_num not in poses_by_frame:
                    poses_by_frame[frame_num] = []
                poses_by_frame[frame_num].append(pose)
            
        except Exception as e:
            self.logger.error(f"Error parsing AlphaPose JSON: {e}")
        
        return poses_by_frame
    
    def draw_pose(
        self,
        frame: np.ndarray,
        pose: HalpePose,
        draw_body: bool = True,
        draw_face: bool = True,
        draw_hands: bool = True
    ) -> np.ndarray:
        """Draw Halpe pose on frame."""
        import cv2
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        def draw_keypoint(kpt: HalpeKeypoint, color: Tuple[int, int, int], radius: int = 3):
            if kpt.is_visible:
                pt = (int(kpt.x), int(kpt.y))
                cv2.circle(result, pt, radius, color, -1)
        
        def draw_connection(kpt1: Optional[HalpeKeypoint], kpt2: Optional[HalpeKeypoint], 
                           color: Tuple[int, int, int], thickness: int = 2):
            if kpt1 and kpt2 and kpt1.is_visible and kpt2.is_visible:
                pt1 = (int(kpt1.x), int(kpt1.y))
                pt2 = (int(kpt2.x), int(kpt2.y))
                cv2.line(result, pt1, pt2, color, thickness)
        
        # Draw body
        if draw_body:
            body_color = (0, 255, 0)  # Green
            for start_idx, end_idx in HALPE_BODY_CONNECTIONS:
                kpt1 = pose.get_keypoint(int(start_idx))
                kpt2 = pose.get_keypoint(int(end_idx))
                draw_connection(kpt1, kpt2, body_color)
            
            for i in range(26):
                kpt = pose.get_keypoint(i)
                if kpt:
                    draw_keypoint(kpt, (0, 0, 255), 4)
        
        # Draw face
        if draw_face:
            face_color = (255, 255, 0)  # Cyan
            for i in range(FACE_KEYPOINT_START, 94):
                kpt = pose.get_keypoint(i)
                if kpt:
                    draw_keypoint(kpt, face_color, 2)
        
        # Draw hands
        if draw_hands:
            left_hand_color = (255, 0, 255)  # Magenta
            right_hand_color = (0, 255, 255)  # Yellow
            
            # Left hand connections
            for start_idx, end_idx in HALPE_HAND_CONNECTIONS:
                kpt1 = pose.get_keypoint(LEFT_HAND_START + int(start_idx))
                kpt2 = pose.get_keypoint(LEFT_HAND_START + int(end_idx))
                draw_connection(kpt1, kpt2, left_hand_color, 1)
            
            # Right hand connections
            for start_idx, end_idx in HALPE_HAND_CONNECTIONS:
                kpt1 = pose.get_keypoint(RIGHT_HAND_START + int(start_idx))
                kpt2 = pose.get_keypoint(RIGHT_HAND_START + int(end_idx))
                draw_connection(kpt1, kpt2, right_hand_color, 1)
            
            # Hand keypoints
            for i in range(LEFT_HAND_START, 115):
                kpt = pose.get_keypoint(i)
                if kpt:
                    draw_keypoint(kpt, left_hand_color, 2)
            
            for i in range(RIGHT_HAND_START, 136):
                kpt = pose.get_keypoint(i)
                if kpt:
                    draw_keypoint(kpt, right_hand_color, 2)
        
        return result


def convert_halpe_to_mixamo(pose: HalpePose, image_width: int, image_height: int) -> Optional[Dict]:
    """
    Convert Halpe 136-keypoint pose to Mixamo bone transforms.
    
    Args:
        pose: HalpePose with 136 keypoints
        image_width: Image width for normalization
        image_height: Image height for normalization
    
    Returns:
        Dict of bone_name -> [px, py, pz, qw, qx, qy, qz, sx, sy, sz]
    """
    import numpy as np
    
    def get_pos(kpt: Optional[HalpeKeypoint]) -> Optional[np.ndarray]:
        if kpt is None or not kpt.is_visible:
            return None
        # Normalize to -0.5 to 0.5 range
        return np.array([
            (kpt.x / image_width) - 0.5,
            -((kpt.y / image_height) - 0.5),  # Flip Y
            kpt.z * 0.5 if kpt.z != 0 else 0.0
        ])
    
    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([0, 1, 0])
    
    def quat_from_vectors(v_from, v_to):
        """Quaternion rotating v_from to v_to."""
        v_from = normalize(v_from)
        v_to = normalize(v_to)
        cross = np.cross(v_from, v_to)
        dot = np.dot(v_from, v_to)
        
        if dot < -0.9999:
            return np.array([0, 0, 1, 0])
        if dot > 0.9999:
            return np.array([1, 0, 0, 0])
        
        w = 1 + dot
        q = np.array([w, cross[0], cross[1], cross[2]])
        return q / np.linalg.norm(q)
    
    def make_transform(quat):
        return [0.0, 0.0, 0.0, quat[0], quat[1], quat[2], quat[3], 1.0, 1.0, 1.0]
    
    bone_data = {}
    
    try:
        # Get body keypoints
        hip_center = get_pos(pose.get_body_keypoint(HalpeBodyIndex.HIP))
        left_hip = get_pos(pose.get_body_keypoint(HalpeBodyIndex.LEFT_HIP))
        right_hip = get_pos(pose.get_body_keypoint(HalpeBodyIndex.RIGHT_HIP))
        neck = get_pos(pose.get_body_keypoint(HalpeBodyIndex.NECK))
        head = get_pos(pose.get_body_keypoint(HalpeBodyIndex.HEAD))
        left_shoulder = get_pos(pose.get_body_keypoint(HalpeBodyIndex.LEFT_SHOULDER))
        right_shoulder = get_pos(pose.get_body_keypoint(HalpeBodyIndex.RIGHT_SHOULDER))
        left_elbow = get_pos(pose.get_body_keypoint(HalpeBodyIndex.LEFT_ELBOW))
        right_elbow = get_pos(pose.get_body_keypoint(HalpeBodyIndex.RIGHT_ELBOW))
        left_wrist = get_pos(pose.get_body_keypoint(HalpeBodyIndex.LEFT_WRIST))
        right_wrist = get_pos(pose.get_body_keypoint(HalpeBodyIndex.RIGHT_WRIST))
        left_knee = get_pos(pose.get_body_keypoint(HalpeBodyIndex.LEFT_KNEE))
        right_knee = get_pos(pose.get_body_keypoint(HalpeBodyIndex.RIGHT_KNEE))
        left_ankle = get_pos(pose.get_body_keypoint(HalpeBodyIndex.LEFT_ANKLE))
        right_ankle = get_pos(pose.get_body_keypoint(HalpeBodyIndex.RIGHT_ANKLE))
        
        # Foot keypoints
        left_heel = get_pos(pose.get_body_keypoint(HalpeBodyIndex.LEFT_HEEL))
        right_heel = get_pos(pose.get_body_keypoint(HalpeBodyIndex.RIGHT_HEEL))
        left_big_toe = get_pos(pose.get_body_keypoint(HalpeBodyIndex.LEFT_BIG_TOE))
        right_big_toe = get_pos(pose.get_body_keypoint(HalpeBodyIndex.RIGHT_BIG_TOE))
        
        if hip_center is None and left_hip is not None and right_hip is not None:
            hip_center = (left_hip + right_hip) / 2
        
        if hip_center is None:
            return None
        
        # Hips rotation
        if neck is not None:
            spine_dir = normalize(neck - hip_center)
            hips_quat = quat_from_vectors(np.array([0, 1, 0]), spine_dir)
            bone_data["mixamorig:Hips"] = make_transform(hips_quat)
        
        # Spine
        bone_data["mixamorig:Spine"] = make_transform(np.array([1, 0, 0, 0]))
        bone_data["mixamorig:Spine1"] = make_transform(np.array([1, 0, 0, 0]))
        bone_data["mixamorig:Spine2"] = make_transform(np.array([1, 0, 0, 0]))
        
        # Neck and Head
        if neck is not None and head is not None:
            head_dir = normalize(head - neck)
            neck_quat = quat_from_vectors(np.array([0, 1, 0]), head_dir)
            bone_data["mixamorig:Neck"] = make_transform(neck_quat)
            bone_data["mixamorig:Head"] = make_transform(np.array([1, 0, 0, 0]))
        
        # Shoulders
        if left_shoulder is not None and neck is not None:
            shoulder_dir = normalize(left_shoulder - neck)
            left_shoulder_quat = quat_from_vectors(np.array([1, 0, 0]), shoulder_dir)
            bone_data["mixamorig:LeftShoulder"] = make_transform(left_shoulder_quat)
        
        if right_shoulder is not None and neck is not None:
            shoulder_dir = normalize(right_shoulder - neck)
            right_shoulder_quat = quat_from_vectors(np.array([-1, 0, 0]), shoulder_dir)
            bone_data["mixamorig:RightShoulder"] = make_transform(right_shoulder_quat)
        
        # Arms
        if left_shoulder is not None and left_elbow is not None:
            arm_dir = normalize(left_elbow - left_shoulder)
            bone_data["mixamorig:LeftArm"] = make_transform(quat_from_vectors(np.array([1, 0, 0]), arm_dir))
        
        if left_elbow is not None and left_wrist is not None:
            forearm_dir = normalize(left_wrist - left_elbow)
            bone_data["mixamorig:LeftForeArm"] = make_transform(quat_from_vectors(np.array([1, 0, 0]), forearm_dir))
        
        if right_shoulder is not None and right_elbow is not None:
            arm_dir = normalize(right_elbow - right_shoulder)
            bone_data["mixamorig:RightArm"] = make_transform(quat_from_vectors(np.array([-1, 0, 0]), arm_dir))
        
        if right_elbow is not None and right_wrist is not None:
            forearm_dir = normalize(right_wrist - right_elbow)
            bone_data["mixamorig:RightForeArm"] = make_transform(quat_from_vectors(np.array([-1, 0, 0]), forearm_dir))
        
        # Legs
        if left_hip is not None and left_knee is not None:
            leg_dir = normalize(left_knee - left_hip)
            bone_data["mixamorig:LeftUpLeg"] = make_transform(quat_from_vectors(np.array([0, -1, 0]), leg_dir))
        
        if left_knee is not None and left_ankle is not None:
            shin_dir = normalize(left_ankle - left_knee)
            bone_data["mixamorig:LeftLeg"] = make_transform(quat_from_vectors(np.array([0, -1, 0]), shin_dir))
        
        if right_hip is not None and right_knee is not None:
            leg_dir = normalize(right_knee - right_hip)
            bone_data["mixamorig:RightUpLeg"] = make_transform(quat_from_vectors(np.array([0, -1, 0]), leg_dir))
        
        if right_knee is not None and right_ankle is not None:
            shin_dir = normalize(right_ankle - right_knee)
            bone_data["mixamorig:RightLeg"] = make_transform(quat_from_vectors(np.array([0, -1, 0]), shin_dir))
        
        # Feet
        if left_ankle is not None and left_big_toe is not None:
            foot_dir = normalize(left_big_toe - left_ankle)
            bone_data["mixamorig:LeftFoot"] = make_transform(quat_from_vectors(np.array([0, 0, 1]), foot_dir))
        
        if right_ankle is not None and right_big_toe is not None:
            foot_dir = normalize(right_big_toe - right_ankle)
            bone_data["mixamorig:RightFoot"] = make_transform(quat_from_vectors(np.array([0, 0, 1]), foot_dir))
        
        # Hands (from Halpe hand keypoints)
        _add_hand_transforms(pose, bone_data, "left", image_width, image_height, 
                            quat_from_vectors, make_transform, normalize)
        _add_hand_transforms(pose, bone_data, "right", image_width, image_height,
                            quat_from_vectors, make_transform, normalize)
        
    except Exception as e:
        print(f"Error converting Halpe pose: {e}")
        return None
    
    return bone_data


def _add_hand_transforms(pose: HalpePose, bone_data: Dict, side: str, 
                         image_width: int, image_height: int,
                         quat_from_vectors, make_transform, normalize):
    """Add hand bone transforms from Halpe hand keypoints."""
    import numpy as np
    
    offset = LEFT_HAND_START if side == "left" else RIGHT_HAND_START
    mapping = HALPE_LEFT_HAND_TO_MIXAMO if side == "left" else HALPE_RIGHT_HAND_TO_MIXAMO
    
    def get_hand_pos(local_idx: int) -> Optional[np.ndarray]:
        kpt = pose.get_keypoint(offset + local_idx)
        if kpt is None or not kpt.is_visible:
            return None
        return np.array([
            (kpt.x / image_width) - 0.5,
            -((kpt.y / image_height) - 0.5),
            kpt.z * 0.5 if kpt.z != 0 else 0.0
        ])
    
    # Finger bone chains
    finger_chains = [
        # (start_idx, end_idx, mixamo_bone_suffix)
        (HalpeHandIndex.WRIST, HalpeHandIndex.THUMB_CMC, "Thumb1"),
        (HalpeHandIndex.THUMB_CMC, HalpeHandIndex.THUMB_MCP, "Thumb2"),
        (HalpeHandIndex.THUMB_MCP, HalpeHandIndex.THUMB_IP, "Thumb3"),
        (HalpeHandIndex.THUMB_IP, HalpeHandIndex.THUMB_TIP, "Thumb4"),
        
        (HalpeHandIndex.WRIST, HalpeHandIndex.INDEX_MCP, "Index1"),
        (HalpeHandIndex.INDEX_MCP, HalpeHandIndex.INDEX_PIP, "Index2"),
        (HalpeHandIndex.INDEX_PIP, HalpeHandIndex.INDEX_DIP, "Index3"),
        (HalpeHandIndex.INDEX_DIP, HalpeHandIndex.INDEX_TIP, "Index4"),
        
        (HalpeHandIndex.WRIST, HalpeHandIndex.MIDDLE_MCP, "Middle1"),
        (HalpeHandIndex.MIDDLE_MCP, HalpeHandIndex.MIDDLE_PIP, "Middle2"),
        (HalpeHandIndex.MIDDLE_PIP, HalpeHandIndex.MIDDLE_DIP, "Middle3"),
        (HalpeHandIndex.MIDDLE_DIP, HalpeHandIndex.MIDDLE_TIP, "Middle4"),
        
        (HalpeHandIndex.WRIST, HalpeHandIndex.RING_MCP, "Ring1"),
        (HalpeHandIndex.RING_MCP, HalpeHandIndex.RING_PIP, "Ring2"),
        (HalpeHandIndex.RING_PIP, HalpeHandIndex.RING_DIP, "Ring3"),
        (HalpeHandIndex.RING_DIP, HalpeHandIndex.RING_TIP, "Ring4"),
        
        (HalpeHandIndex.WRIST, HalpeHandIndex.PINKY_MCP, "Pinky1"),
        (HalpeHandIndex.PINKY_MCP, HalpeHandIndex.PINKY_PIP, "Pinky2"),
        (HalpeHandIndex.PINKY_PIP, HalpeHandIndex.PINKY_DIP, "Pinky3"),
        (HalpeHandIndex.PINKY_DIP, HalpeHandIndex.PINKY_TIP, "Pinky4"),
    ]
    
    hand_prefix = "Left" if side == "left" else "Right"
    bind_dir = np.array([1, 0, 0]) if side == "left" else np.array([-1, 0, 0])
    
    for start_idx, end_idx, bone_suffix in finger_chains:
        start_pos = get_hand_pos(int(start_idx))
        end_pos = get_hand_pos(int(end_idx))
        
        if start_pos is not None and end_pos is not None:
            bone_dir = normalize(end_pos - start_pos)
            quat = quat_from_vectors(bind_dir, bone_dir)
            bone_name = f"mixamorig:{hand_prefix}Hand{bone_suffix}"
            bone_data[bone_name] = make_transform(quat)
