#!/usr/bin/env python3
"""Demo script for video-to-skeleton animation with 3D viewer preview.

Usage:
    # From video file:
    python demo_viewer.py --video path/to/video.mp4 --model "downloaded fbx/mixamo-t-Pose.fbx"
    
    # From camera:
    python demo_viewer.py --camera 0 --model "downloaded fbx/mixamo-t-Pose.fbx"
    
    # Viewer only (no pose estimation):
    python demo_viewer.py --model "downloaded fbx/mixamo-t-Pose.fbx"
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add viewer build directory to path for the Python module
viewer_build = project_root / "viewer" / "build"
if viewer_build.exists():
    sys.path.insert(0, str(viewer_build))


def run_viewer_only(model_path: str):
    """Run viewer without pose estimation - just display model."""
    import mocap_viewer_py
    
    viewer = mocap_viewer_py.MoCapViewer(1280, 720, "MoCap Viewer")
    if not viewer.initialize():
        print("Failed to initialize viewer")
        return
        
    viewer.load_model(model_path)
    print("Viewer running. Close window or press ESC to quit.")
    
    # Run on main thread (required for macOS)
    viewer.run()


def run_with_video(model_path: str, video_path: str):
    """Process video and display animation in viewer."""
    import cv2
    import numpy as np
    import mocap_viewer_py
    
    # Use project's pose estimator (Tasks API)
    try:
        from src.pose.estimator_2d import PoseEstimator2D, SKELETON_BONES
        from src.pose.hand_tracker import HandTracker
        POSE_AVAILABLE = True
        HANDS_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Pose estimator not available: {e}")
        POSE_AVAILABLE = False
        HANDS_AVAILABLE = False
    
    # Initialize viewer
    viewer = mocap_viewer_py.MoCapViewer(1280, 720, "MoCap Animation Preview")
    if not viewer.initialize():
        print("Failed to initialize viewer")
        return
        
    viewer.load_model(model_path)
    viewer.run_async()  # Mark as running
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time = 1.0 / fps
    
    print(f"Processing video at {fps:.1f} FPS...")
    print("Close viewer window or press ESC to quit.")
    
    # Initialize pose estimation
    estimator = None
    hand_tracker = None
    if POSE_AVAILABLE:
        print("Initializing pose estimator (heavy model)...")
        estimator = PoseEstimator2D(model_type="heavy")  # Use heavy for best accuracy
    if HANDS_AVAILABLE:
        print("Initializing hand tracker...")
        hand_tracker = HandTracker()
    
    frame_count = 0
    
    try:
        while True:
            # Process viewer frame (handles window events, rendering)
            if not viewer.run_one_frame():
                break
                
            # Read video frame
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Process pose estimation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bone_data = {}
            
            if estimator is not None:
                pose_result = estimator.process(rgb_frame, timestamp)
                
                if pose_result and pose_result.is_valid:
                    # Convert pose to bone transforms
                    pose_bones = convert_pose2d_to_mixamo(pose_result)
                    if pose_bones:
                        bone_data.update(pose_bones)
                    
                    # Draw pose on frame
                    draw_pose_on_frame(frame, pose_result, SKELETON_BONES)
            
            # Process hand tracking
            if hand_tracker is not None:
                hands_result = hand_tracker.process(rgb_frame, timestamp)
                
                if hands_result and hands_result.has_hands:
                    # Convert hands to bone transforms
                    hand_bones = convert_hands_to_mixamo(hands_result)
                    if hand_bones:
                        bone_data.update(hand_bones)
                    
                    # Draw hands on frame
                    draw_hands_on_frame(frame, hands_result)
            
            # Send all bone data to viewer
            if bone_data:
                if frame_count == 1:
                    print(f"Sending {len(bone_data)} bones to viewer:")
                    for name in sorted(bone_data.keys()):
                        print(f"  {name}")
                viewer.set_animation_frame(bone_data)
            
            # Show video frame with pose overlay
            cv2.imshow('Video - Press Q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Timing control
            time.sleep(max(0, frame_time - 0.02))
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        viewer.stop()


def draw_pose_on_frame(frame, pose, connections):
    """Draw ALL 33 pose landmarks and connections on frame."""
    import cv2
    h, w = frame.shape[:2]
    
    # Complete skeleton connections (all 33 landmarks)
    FULL_SKELETON = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye to ear
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye to ear
        (9, 10),  # Mouth
        # Torso
        (11, 12),  # Shoulders
        (11, 23), (12, 24),  # Shoulders to hips
        (23, 24),  # Hips
        # Left arm
        (11, 13), (13, 15),  # Shoulder -> elbow -> wrist
        (15, 17), (15, 19), (15, 21),  # Wrist to fingers
        (17, 19),  # Pinky to index
        # Right arm
        (12, 14), (14, 16),  # Shoulder -> elbow -> wrist
        (16, 18), (16, 20), (16, 22),  # Wrist to fingers
        (18, 20),  # Pinky to index
        # Left leg
        (23, 25), (25, 27),  # Hip -> knee -> ankle
        (27, 29), (27, 31), (29, 31),  # Ankle -> heel/foot
        # Right leg
        (24, 26), (26, 28),  # Hip -> knee -> ankle
        (28, 30), (28, 32), (30, 32),  # Ankle -> heel/foot
    ]
    
    # Draw connections with different colors for body parts
    for (start_idx, end_idx) in FULL_SKELETON:
        start_joint = pose.get_joint_by_index(start_idx)
        end_joint = pose.get_joint_by_index(end_idx)
        if start_joint and end_joint:
            start_pt = start_joint.pixel_coords(w, h)
            end_pt = end_joint.pixel_coords(w, h)
            
            # Color coding: face=cyan, arms=green, torso=yellow, legs=magenta
            if start_idx <= 10 or end_idx <= 10:
                color = (255, 255, 0)  # Cyan for face
            elif start_idx in [11, 13, 15, 17, 19, 21] or end_idx in [11, 13, 15, 17, 19, 21]:
                color = (0, 255, 0)  # Green for left arm
            elif start_idx in [12, 14, 16, 18, 20, 22] or end_idx in [12, 14, 16, 18, 20, 22]:
                color = (0, 255, 0)  # Green for right arm
            elif start_idx in [23, 25, 27, 29, 31]:
                color = (255, 0, 255)  # Magenta for left leg
            elif start_idx in [24, 26, 28, 30, 32]:
                color = (255, 0, 255)  # Magenta for right leg
            else:
                color = (0, 255, 255)  # Yellow for torso
            
            cv2.line(frame, start_pt, end_pt, color, 2)
    
    # Draw ALL joints with index labels
    for joint in pose.joints.values():
        pt = joint.pixel_coords(w, h)
        # Color based on confidence
        if joint.confidence > 0.7:
            color = (0, 0, 255)  # Red - high confidence
        elif joint.confidence > 0.5:
            color = (0, 165, 255)  # Orange - medium confidence
        else:
            color = (128, 128, 128)  # Gray - low confidence
        
        cv2.circle(frame, pt, 5, color, -1)
        # Draw index number
        cv2.putText(frame, str(joint.index), (pt[0]+5, pt[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def draw_hands_on_frame(frame, hands_data):
    """Draw hand landmarks on frame."""
    import cv2
    from src.pose.hand_tracker import HAND_CONNECTIONS
    
    h, w = frame.shape[:2]
    
    for hand in [hands_data.left_hand, hands_data.right_hand]:
        if hand is None:
            continue
        
        # Color: green for left, blue for right
        color = (0, 255, 0) if hand.handedness == "left" else (255, 0, 0)
        
        # Draw connections
        for (start_idx, end_idx) in HAND_CONNECTIONS:
            start_name = list(hand.joints.keys())[start_idx] if start_idx < len(hand.joints) else None
            end_name = list(hand.joints.keys())[end_idx] if end_idx < len(hand.joints) else None
            
            if start_name and end_name:
                start_joint = hand.joints.get(start_name)
                end_joint = hand.joints.get(end_name)
                if start_joint and end_joint:
                    start_pt = (int(start_joint.x * w), int(start_joint.y * h))
                    end_pt = (int(end_joint.x * w), int(end_joint.y * h))
                    cv2.line(frame, start_pt, end_pt, color, 2)
        
        # Draw joints
        for joint in hand.joints.values():
            pt = (int(joint.x * w), int(joint.y * h))
            cv2.circle(frame, pt, 3, color, -1)


def convert_hands_to_mixamo(hands_data):
    """Convert hand tracking data to Mixamo finger bone transforms.
    
    Maps MediaPipe hand landmarks to Mixamo finger bones.
    """
    import numpy as np
    
    # Mixamo hand bone mapping
    MIXAMO_HAND_BONES = {
        "left": {
            "thumb_mcp": "mixamorig:LeftHandThumb1",
            "thumb_ip": "mixamorig:LeftHandThumb2",
            "thumb_tip": "mixamorig:LeftHandThumb3",
            "index_mcp": "mixamorig:LeftHandIndex1",
            "index_pip": "mixamorig:LeftHandIndex2",
            "index_dip": "mixamorig:LeftHandIndex3",
            "middle_mcp": "mixamorig:LeftHandMiddle1",
            "middle_pip": "mixamorig:LeftHandMiddle2",
            "middle_dip": "mixamorig:LeftHandMiddle3",
            "ring_mcp": "mixamorig:LeftHandRing1",
            "ring_pip": "mixamorig:LeftHandRing2",
            "ring_dip": "mixamorig:LeftHandRing3",
            "pinky_mcp": "mixamorig:LeftHandPinky1",
            "pinky_pip": "mixamorig:LeftHandPinky2",
            "pinky_dip": "mixamorig:LeftHandPinky3",
        },
        "right": {
            "thumb_mcp": "mixamorig:RightHandThumb1",
            "thumb_ip": "mixamorig:RightHandThumb2",
            "thumb_tip": "mixamorig:RightHandThumb3",
            "index_mcp": "mixamorig:RightHandIndex1",
            "index_pip": "mixamorig:RightHandIndex2",
            "index_dip": "mixamorig:RightHandIndex3",
            "middle_mcp": "mixamorig:RightHandMiddle1",
            "middle_pip": "mixamorig:RightHandMiddle2",
            "middle_dip": "mixamorig:RightHandMiddle3",
            "ring_mcp": "mixamorig:RightHandRing1",
            "ring_pip": "mixamorig:RightHandRing2",
            "ring_dip": "mixamorig:RightHandRing3",
            "pinky_mcp": "mixamorig:RightHandPinky1",
            "pinky_pip": "mixamorig:RightHandPinky2",
            "pinky_dip": "mixamorig:RightHandPinky3",
        }
    }
    
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
    
    def make_bone_transform(quat):
        return [0.0, 0.0, 0.0, quat[0], quat[1], quat[2], quat[3], 1.0, 1.0, 1.0]
    
    bone_data = {}
    
    for hand in [hands_data.left_hand, hands_data.right_hand]:
        if hand is None:
            continue
        
        bone_map = MIXAMO_HAND_BONES.get(hand.handedness, {})
        
        # For each finger, calculate rotations between joints
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        
        for finger in fingers:
            # Get finger joints
            joints = hand.get_finger_joints(finger)
            if len(joints) < 2:
                continue
            
            # Sort joints by their position in the chain
            joint_order = ["mcp", "pip", "dip", "tip"] if finger != "thumb" else ["cmc", "mcp", "ip", "tip"]
            
            for i, suffix in enumerate(joint_order[:-1]):
                joint_name = f"{finger}_{suffix}"
                next_suffix = joint_order[i + 1]
                next_joint_name = f"{finger}_{next_suffix}"
                
                current = hand.joints.get(joint_name)
                next_joint = hand.joints.get(next_joint_name)
                
                if current and next_joint:
                    mixamo_bone = bone_map.get(joint_name)
                    if mixamo_bone:
                        # Calculate direction to next joint
                        dir_vec = np.array([
                            next_joint.x - current.x,
                            -(next_joint.y - current.y),  # Flip Y
                            0.0
                        ])
                        
                        # Bind pose direction (fingers point outward)
                        if hand.handedness == "left":
                            bind_dir = np.array([1, 0, 0])  # Left hand fingers point +X
                        else:
                            bind_dir = np.array([-1, 0, 0])  # Right hand fingers point -X
                        
                        quat = quat_from_vectors(bind_dir, dir_vec)
                        bone_data[mixamo_bone] = make_bone_transform(quat)
    
    return bone_data


def convert_pose2d_to_mixamo(pose):
    """Convert Pose2D to Mixamo bone transforms.
    
    Uses 2D pose landmarks to compute bone ROTATIONS relative to bind pose.
    For skeletal animation, we only need rotations - positions are handled by hierarchy.
    Returns dict of bone_name -> [px, py, pz, qw, qx, qy, qz, sx, sy, sz]
    """
    import numpy as np
    
    def get_pos(joint_name):
        joint = pose.joints.get(joint_name)
        if joint and joint.is_visible:
            # Convert to 3D using MediaPipe coordinates
            # x: 0-1 normalized (left to right)
            # y: 0-1 normalized (top to bottom)
            # z: depth relative to hips (negative = closer to camera)
            return np.array([
                (joint.x - 0.5),      # Center x (-0.5 to 0.5)
                -(joint.y - 0.5),     # Flip y (up is positive)
                -joint.z * 0.5        # Scale and flip z (MediaPipe z is depth)
            ])
        return None
    
    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([0, 1, 0])
    
    def quat_from_axis_angle(axis, angle):
        """Create quaternion from axis-angle representation."""
        axis = normalize(axis)
        s = np.sin(angle / 2)
        return np.array([np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])
    
    def quat_from_vectors(v_from, v_to):
        """Quaternion rotating v_from to v_to."""
        v_from = normalize(v_from)
        v_to = normalize(v_to)
        cross = np.cross(v_from, v_to)
        dot = np.dot(v_from, v_to)
        
        if dot < -0.9999:
            # 180 degree rotation
            return np.array([0, 0, 1, 0])
        if dot > 0.9999:
            # No rotation needed
            return np.array([1, 0, 0, 0])
        
        w = 1 + dot
        q = np.array([w, cross[0], cross[1], cross[2]])
        return q / np.linalg.norm(q)
    
    def make_bone_transform(quat):
        """Create bone transform with rotation only (position=0, scale=1)."""
        return [0.0, 0.0, 0.0, quat[0], quat[1], quat[2], quat[3], 1.0, 1.0, 1.0]
    
    bone_data = {}
    
    try:
        # Get key positions
        left_hip = get_pos("left_hip")
        right_hip = get_pos("right_hip")
        left_shoulder = get_pos("left_shoulder")
        right_shoulder = get_pos("right_shoulder")
        left_elbow = get_pos("left_elbow")
        right_elbow = get_pos("right_elbow")
        left_wrist = get_pos("left_wrist")
        right_wrist = get_pos("right_wrist")
        left_knee = get_pos("left_knee")
        right_knee = get_pos("right_knee")
        left_ankle = get_pos("left_ankle")
        right_ankle = get_pos("right_ankle")
        
        if left_hip is None or right_hip is None:
            return None
        
        # Calculate body orientation from hips and shoulders
        hip_center = (left_hip + right_hip) / 2
        hip_vector = normalize(left_hip - right_hip)  # Points to the left
        
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = (left_shoulder + right_shoulder) / 2
            spine_dir = normalize(shoulder_center - hip_center)
        else:
            spine_dir = np.array([0, 1, 0])
        
        # Hips rotation - rotate to match spine tilt
        # Bind pose: spine points up (0, 1, 0)
        hips_quat = quat_from_vectors(np.array([0, 1, 0]), spine_dir)
        bone_data["mixamorig:Hips"] = make_bone_transform(hips_quat)
        
        # Spine - identity since hips already rotated
        bone_data["mixamorig:Spine"] = make_bone_transform(np.array([1, 0, 0, 0]))
        
        # Left Arm - bind pose points LEFT (+X in Mixamo)
        if left_shoulder is not None and left_elbow is not None:
            left_arm_dir = normalize(left_elbow - left_shoulder)
            # Bind pose: arm points left (+X)
            left_arm_quat = quat_from_vectors(np.array([1, 0, 0]), left_arm_dir)
            bone_data["mixamorig:LeftArm"] = make_bone_transform(left_arm_quat)
        
        # Left ForeArm
        if left_elbow is not None and left_wrist is not None:
            left_forearm_dir = normalize(left_wrist - left_elbow)
            left_forearm_quat = quat_from_vectors(np.array([1, 0, 0]), left_forearm_dir)
            bone_data["mixamorig:LeftForeArm"] = make_bone_transform(left_forearm_quat)
        
        # Right Arm - bind pose points RIGHT (-X in Mixamo)
        if right_shoulder is not None and right_elbow is not None:
            right_arm_dir = normalize(right_elbow - right_shoulder)
            # Bind pose: arm points right (-X)
            right_arm_quat = quat_from_vectors(np.array([-1, 0, 0]), right_arm_dir)
            bone_data["mixamorig:RightArm"] = make_bone_transform(right_arm_quat)
        
        # Right ForeArm
        if right_elbow is not None and right_wrist is not None:
            right_forearm_dir = normalize(right_wrist - right_elbow)
            right_forearm_quat = quat_from_vectors(np.array([-1, 0, 0]), right_forearm_dir)
            bone_data["mixamorig:RightForeArm"] = make_bone_transform(right_forearm_quat)
        
        # Left Leg - bind pose points DOWN (-Y)
        if left_hip is not None and left_knee is not None:
            left_leg_dir = normalize(left_knee - left_hip)
            left_leg_quat = quat_from_vectors(np.array([0, -1, 0]), left_leg_dir)
            bone_data["mixamorig:LeftUpLeg"] = make_bone_transform(left_leg_quat)
        
        # Left Lower Leg
        if left_knee is not None and left_ankle is not None:
            left_shin_dir = normalize(left_ankle - left_knee)
            left_shin_quat = quat_from_vectors(np.array([0, -1, 0]), left_shin_dir)
            bone_data["mixamorig:LeftLeg"] = make_bone_transform(left_shin_quat)
        
        # Right Leg - bind pose points DOWN (-Y)
        if right_hip is not None and right_knee is not None:
            right_leg_dir = normalize(right_knee - right_hip)
            right_leg_quat = quat_from_vectors(np.array([0, -1, 0]), right_leg_dir)
            bone_data["mixamorig:RightUpLeg"] = make_bone_transform(right_leg_quat)
        
        # Right Lower Leg
        if right_knee is not None and right_ankle is not None:
            right_shin_dir = normalize(right_ankle - right_knee)
            right_shin_quat = quat_from_vectors(np.array([0, -1, 0]), right_shin_dir)
            bone_data["mixamorig:RightLeg"] = make_bone_transform(right_shin_quat)
        
        # Additional bones for better accuracy
        
        # Left Shoulder
        if left_shoulder is not None and right_shoulder is not None:
            # Shoulder rotates arm outward
            shoulder_to_arm = normalize(left_shoulder - (left_shoulder + right_shoulder) / 2)
            left_shoulder_quat = quat_from_vectors(np.array([1, 0, 0]), shoulder_to_arm)
            bone_data["mixamorig:LeftShoulder"] = make_bone_transform(left_shoulder_quat)
        
        # Right Shoulder
        if right_shoulder is not None and left_shoulder is not None:
            shoulder_to_arm = normalize(right_shoulder - (left_shoulder + right_shoulder) / 2)
            right_shoulder_quat = quat_from_vectors(np.array([-1, 0, 0]), shoulder_to_arm)
            bone_data["mixamorig:RightShoulder"] = make_bone_transform(right_shoulder_quat)
        
        # Neck and Head
        nose = get_pos("nose")
        left_ear = get_pos("left_ear")
        right_ear = get_pos("right_ear")
        
        if nose is not None and left_shoulder is not None and right_shoulder is not None:
            neck_base = (left_shoulder + right_shoulder) / 2
            head_dir = normalize(nose - neck_base)
            neck_quat = quat_from_vectors(np.array([0, 1, 0]), head_dir)
            bone_data["mixamorig:Neck"] = make_bone_transform(neck_quat)
            bone_data["mixamorig:Head"] = make_bone_transform(np.array([1, 0, 0, 0]))
        
        # Feet
        left_heel = get_pos("left_heel")
        left_foot_index = get_pos("left_foot_index")
        if left_ankle is not None and left_foot_index is not None:
            foot_dir = normalize(left_foot_index - left_ankle)
            left_foot_quat = quat_from_vectors(np.array([0, 0, 1]), foot_dir)
            bone_data["mixamorig:LeftFoot"] = make_bone_transform(left_foot_quat)
        
        right_heel = get_pos("right_heel")
        right_foot_index = get_pos("right_foot_index")
        if right_ankle is not None and right_foot_index is not None:
            foot_dir = normalize(right_foot_index - right_ankle)
            right_foot_quat = quat_from_vectors(np.array([0, 0, 1]), foot_dir)
            bone_data["mixamorig:RightFoot"] = make_bone_transform(right_foot_quat)
        
    except Exception as e:
        print(f"Error converting pose: {e}")
        return None
    
    return bone_data


def convert_mediapipe_to_mixamo(landmarks):
    """Legacy: Convert MediaPipe pose landmarks to Mixamo bone transforms.
    
    MediaPipe provides 33 body landmarks. We map these to Mixamo skeleton bones.
    Returns dict of bone_name -> [px, py, pz, qw, qx, qy, qz, sx, sy, sz]
    """
    import numpy as np
    
    # MediaPipe landmark indices
    MP_NOSE = 0
    MP_LEFT_SHOULDER = 11
    MP_RIGHT_SHOULDER = 12
    MP_LEFT_ELBOW = 13
    MP_RIGHT_ELBOW = 14
    MP_LEFT_WRIST = 15
    MP_RIGHT_WRIST = 16
    MP_LEFT_HIP = 23
    MP_RIGHT_HIP = 24
    MP_LEFT_KNEE = 25
    MP_RIGHT_KNEE = 26
    MP_LEFT_ANKLE = 27
    MP_RIGHT_ANKLE = 28
    
    # Scale factor (MediaPipe uses meters, Mixamo uses cm-ish units)
    SCALE = 100.0
    
    def get_pos(idx):
        lm = landmarks.landmark[idx]
        # MediaPipe: x=right, y=down, z=forward (towards camera)
        # Convert to: x=right, y=up, z=forward
        return np.array([lm.x * SCALE, -lm.y * SCALE, -lm.z * SCALE])
    
    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([0, 1, 0])
    
    def quat_from_vectors(v1, v2):
        """Quaternion rotating v1 to v2."""
        v1 = normalize(v1)
        v2 = normalize(v2)
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        if dot < -0.9999:
            return np.array([0, 1, 0, 0])  # 180 degree rotation
        w = 1 + dot
        q = np.array([w, cross[0], cross[1], cross[2]])
        return q / np.linalg.norm(q)
    
    def identity_quat():
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    bone_data = {}
    
    try:
        # Get key positions
        left_hip = get_pos(MP_LEFT_HIP)
        right_hip = get_pos(MP_RIGHT_HIP)
        left_shoulder = get_pos(MP_LEFT_SHOULDER)
        right_shoulder = get_pos(MP_RIGHT_SHOULDER)
        left_elbow = get_pos(MP_LEFT_ELBOW)
        right_elbow = get_pos(MP_RIGHT_ELBOW)
        left_wrist = get_pos(MP_LEFT_WRIST)
        right_wrist = get_pos(MP_RIGHT_WRIST)
        left_knee = get_pos(MP_LEFT_KNEE)
        right_knee = get_pos(MP_RIGHT_KNEE)
        left_ankle = get_pos(MP_LEFT_ANKLE)
        right_ankle = get_pos(MP_RIGHT_ANKLE)
        nose = get_pos(MP_NOSE)
        
        # Hip center (root)
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # Spine direction
        spine_dir = normalize(shoulder_center - hip_center)
        spine_quat = quat_from_vectors(np.array([0, 1, 0]), spine_dir)
        
        # Create bone transforms
        # Format: [px, py, pz, qw, qx, qy, qz, sx, sy, sz]
        
        # Hips - root bone
        bone_data["mixamorig:Hips"] = [
            hip_center[0], hip_center[1], hip_center[2],
            spine_quat[0], spine_quat[1], spine_quat[2], spine_quat[3],
            1.0, 1.0, 1.0
        ]
        
        # Spine bones
        spine_pos = hip_center + spine_dir * 10
        bone_data["mixamorig:Spine"] = [
            spine_pos[0], spine_pos[1], spine_pos[2],
            spine_quat[0], spine_quat[1], spine_quat[2], spine_quat[3],
            1.0, 1.0, 1.0
        ]
        
        # Left arm chain
        left_arm_dir = normalize(left_elbow - left_shoulder)
        left_arm_quat = quat_from_vectors(np.array([1, 0, 0]), left_arm_dir)
        bone_data["mixamorig:LeftArm"] = [
            left_shoulder[0], left_shoulder[1], left_shoulder[2],
            left_arm_quat[0], left_arm_quat[1], left_arm_quat[2], left_arm_quat[3],
            1.0, 1.0, 1.0
        ]
        
        left_forearm_dir = normalize(left_wrist - left_elbow)
        left_forearm_quat = quat_from_vectors(np.array([1, 0, 0]), left_forearm_dir)
        bone_data["mixamorig:LeftForeArm"] = [
            left_elbow[0], left_elbow[1], left_elbow[2],
            left_forearm_quat[0], left_forearm_quat[1], left_forearm_quat[2], left_forearm_quat[3],
            1.0, 1.0, 1.0
        ]
        
        bone_data["mixamorig:LeftHand"] = [
            left_wrist[0], left_wrist[1], left_wrist[2],
            left_forearm_quat[0], left_forearm_quat[1], left_forearm_quat[2], left_forearm_quat[3],
            1.0, 1.0, 1.0
        ]
        
        # Right arm chain
        right_arm_dir = normalize(right_elbow - right_shoulder)
        right_arm_quat = quat_from_vectors(np.array([-1, 0, 0]), right_arm_dir)
        bone_data["mixamorig:RightArm"] = [
            right_shoulder[0], right_shoulder[1], right_shoulder[2],
            right_arm_quat[0], right_arm_quat[1], right_arm_quat[2], right_arm_quat[3],
            1.0, 1.0, 1.0
        ]
        
        right_forearm_dir = normalize(right_wrist - right_elbow)
        right_forearm_quat = quat_from_vectors(np.array([-1, 0, 0]), right_forearm_dir)
        bone_data["mixamorig:RightForeArm"] = [
            right_elbow[0], right_elbow[1], right_elbow[2],
            right_forearm_quat[0], right_forearm_quat[1], right_forearm_quat[2], right_forearm_quat[3],
            1.0, 1.0, 1.0
        ]
        
        bone_data["mixamorig:RightHand"] = [
            right_wrist[0], right_wrist[1], right_wrist[2],
            right_forearm_quat[0], right_forearm_quat[1], right_forearm_quat[2], right_forearm_quat[3],
            1.0, 1.0, 1.0
        ]
        
        # Left leg chain
        left_leg_dir = normalize(left_knee - left_hip)
        left_leg_quat = quat_from_vectors(np.array([0, -1, 0]), left_leg_dir)
        bone_data["mixamorig:LeftUpLeg"] = [
            left_hip[0], left_hip[1], left_hip[2],
            left_leg_quat[0], left_leg_quat[1], left_leg_quat[2], left_leg_quat[3],
            1.0, 1.0, 1.0
        ]
        
        left_shin_dir = normalize(left_ankle - left_knee)
        left_shin_quat = quat_from_vectors(np.array([0, -1, 0]), left_shin_dir)
        bone_data["mixamorig:LeftLeg"] = [
            left_knee[0], left_knee[1], left_knee[2],
            left_shin_quat[0], left_shin_quat[1], left_shin_quat[2], left_shin_quat[3],
            1.0, 1.0, 1.0
        ]
        
        bone_data["mixamorig:LeftFoot"] = [
            left_ankle[0], left_ankle[1], left_ankle[2],
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0
        ]
        
        # Right leg chain
        right_leg_dir = normalize(right_knee - right_hip)
        right_leg_quat = quat_from_vectors(np.array([0, -1, 0]), right_leg_dir)
        bone_data["mixamorig:RightUpLeg"] = [
            right_hip[0], right_hip[1], right_hip[2],
            right_leg_quat[0], right_leg_quat[1], right_leg_quat[2], right_leg_quat[3],
            1.0, 1.0, 1.0
        ]
        
        right_shin_dir = normalize(right_ankle - right_knee)
        right_shin_quat = quat_from_vectors(np.array([0, -1, 0]), right_shin_dir)
        bone_data["mixamorig:RightLeg"] = [
            right_knee[0], right_knee[1], right_knee[2],
            right_shin_quat[0], right_shin_quat[1], right_shin_quat[2], right_shin_quat[3],
            1.0, 1.0, 1.0
        ]
        
        bone_data["mixamorig:RightFoot"] = [
            right_ankle[0], right_ankle[1], right_ankle[2],
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0
        ]
        
        # Head
        head_pos = shoulder_center + spine_dir * 15
        bone_data["mixamorig:Head"] = [
            head_pos[0], head_pos[1], head_pos[2],
            spine_quat[0], spine_quat[1], spine_quat[2], spine_quat[3],
            1.0, 1.0, 1.0
        ]
        
    except Exception as e:
        print(f"Error converting pose: {e}")
        return None
    
    return bone_data


def run_with_camera(model_path: str, camera_index: int):
    """Run real-time pose estimation from camera."""
    import cv2
    import mocap_viewer_py
    
    # Initialize viewer
    viewer = mocap_viewer_py.MoCapViewer(1280, 720, "MoCap Live Preview")
    if not viewer.initialize():
        print("Failed to initialize viewer")
        return
        
    viewer.load_model(model_path)
    viewer.run_async()
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return
        
    print("Camera capture started. Close viewer or press 'q' to quit.")
    
    try:
        while True:
            # Process viewer frame
            if not viewer.run_one_frame():
                break
                
            # Read camera frame
            ret, frame = cap.read()
            if not ret:
                continue
                
            # TODO: Add pose estimation here and send to viewer
            
            # Show camera feed
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        viewer.stop()


def main():
    parser = argparse.ArgumentParser(description="Video to Skeleton Animation Demo")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (0 for default)")
    parser.add_argument("--model", type=str, required=True, help="Path to Mixamo FBX model")
    args = parser.parse_args()
    
    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path
        
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
        
    print(f"Loading model: {model_path}")
    
    try:
        if args.video:
            video_path = Path(args.video)
            if not video_path.is_absolute():
                video_path = project_root / video_path
            if not video_path.exists():
                print(f"Error: Video not found: {video_path}")
                sys.exit(1)
            print(f"Processing video: {video_path}")
            run_with_video(str(model_path), str(video_path))
            
        elif args.camera is not None:
            print(f"Starting camera capture (camera {args.camera})...")
            run_with_camera(str(model_path), args.camera)
            
        else:
            # Just run viewer with model
            print("Running viewer only (no video/camera)")
            run_viewer_only(str(model_path))
            
    except ImportError as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have built the viewer:")
        print("  cd viewer && mkdir build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python3) && make")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
            
    print("Done!")


if __name__ == "__main__":
    main()
