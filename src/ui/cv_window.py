"""OpenCV-based UI - cross-platform fallback when GTK is not available"""

import threading
import time
from pathlib import Path
from typing import Optional, List
from enum import Enum, auto

import cv2
import numpy as np

from src.core import Config, get_logger, FrameTimer
from src.video.capture import VideoCapture, CaptureState
from src.pose.estimator_2d import PoseEstimator2D
from src.pose.reconstructor_3d import PoseReconstructor3D
from src.motion.floor_detector import FloorDetector
from src.motion.root_motion import RootMotionExtractor
from src.motion.skeleton_solver import SkeletonSolver, SkeletonPose
from src.ik.solver import IKSolver
from src.export.fbx_exporter import FBXExporter


class AppState(Enum):
    IDLE = auto()
    PLAYING = auto()
    PAUSED = auto()
    RECORDING = auto()


class CVWindow:
    """OpenCV-based application window with full motion capture pipeline."""
    
    WINDOW_NAME = "MoCap to UE5 Animation"
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("ui.cv")
        
        self._video_capture: Optional[VideoCapture] = None
        self._pose_estimator: Optional[PoseEstimator2D] = None
        self._pose_reconstructor: Optional[PoseReconstructor3D] = None
        self._floor_detector: Optional[FloorDetector] = None
        self._root_motion: Optional[RootMotionExtractor] = None
        self._skeleton_solver: Optional[SkeletonSolver] = None
        self._ik_solver: Optional[IKSolver] = None
        self._fbx_exporter: Optional[FBXExporter] = None
        
        self._state = AppState.IDLE
        self._is_recording = False
        self._recorded_poses: List[SkeletonPose] = []
        self._frame_timer = FrameTimer()
        
        self._show_skeleton = True
        self._enable_ik = True
        self._enable_floor = True
        
        self._current_frame: Optional[np.ndarray] = None
        self._fps = 0.0
        self._pose_count = 0
        self._motion_type = "idle"
        
        self._running = False
        
        self._initialize_pipeline()
        self.logger.info("CV Window initialized")
    
    def _initialize_pipeline(self) -> None:
        """Initialize all pipeline components."""
        try:
            self._video_capture = VideoCapture(self.config)
            self._pose_estimator = PoseEstimator2D(self.config)
            self._pose_reconstructor = PoseReconstructor3D(self.config)
            self._floor_detector = FloorDetector(self.config)
            self._root_motion = RootMotionExtractor(self.config)
            self._skeleton_solver = SkeletonSolver(self.config)
            self._ik_solver = IKSolver(self.config)
            self._fbx_exporter = FBXExporter(self.config)
            
            self.logger.info("All pipeline components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def run(self, source: Optional[str] = None) -> None:
        """Run the application main loop."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)
        
        if source is None:
            source = self.config.get("video.source", "webcam")
        
        if not self._video_capture.open(source):
            self.logger.error(f"Failed to open video source: {source}")
            self._show_error_screen("Failed to open video source")
            return
        
        self._video_capture.start()
        self._state = AppState.PLAYING
        self._running = True
        
        self.logger.info("Starting main loop - Press keys for controls:")
        self.logger.info("  SPACE: Pause/Resume")
        self.logger.info("  R: Start/Stop Recording")
        self.logger.info("  E: Export FBX")
        self.logger.info("  S: Toggle Skeleton Overlay")
        self.logger.info("  I: Toggle IK")
        self.logger.info("  F: Toggle Floor Detection")
        self.logger.info("  Q/ESC: Quit")
        
        try:
            self._main_loop()
        finally:
            self._cleanup()
    
    def _main_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            self._frame_timer.start()
            
            if self._state == AppState.PLAYING or self._state == AppState.RECORDING:
                frame_result = self._video_capture.read()
                
                if frame_result is None:
                    if self._video_capture.state == CaptureState.STOPPED:
                        self.logger.info("Video ended")
                        self._state = AppState.IDLE
                    continue
                
                frame = frame_result.frame
                timestamp = frame_result.timestamp
                
                display_frame, skeleton_pose = self._process_frame(frame, timestamp)
                
                if self._is_recording and skeleton_pose is not None:
                    self._recorded_poses.append(skeleton_pose)
                
                self._current_frame = display_frame
            
            if self._current_frame is not None:
                display = self._draw_ui(self._current_frame.copy())
                
                display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                cv2.imshow(self.WINDOW_NAME, display_bgr)
            else:
                self._show_idle_screen()
            
            self._frame_timer.stop()
            self._fps = self._frame_timer.fps
            
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)
    
    def _process_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> tuple:
        """Process a single frame through the pipeline."""
        skeleton_pose = None
        
        pose_2d = self._pose_estimator.process(frame, timestamp)
        
        if pose_2d and pose_2d.is_valid:
            self._pose_count = pose_2d.num_visible_joints
            
            pose_3d = self._pose_reconstructor.process(frame, timestamp)
            
            if pose_3d and pose_3d.is_valid:
                if self._enable_floor:
                    self._floor_detector.update(pose_3d)
                
                floor = self._floor_detector.current_plane
                root_transform = self._root_motion.process(pose_3d, floor)
                
                skeleton_pose = self._skeleton_solver.solve(pose_3d, root_transform)
                
                if self._enable_ik:
                    skeleton_pose = self._ik_solver.process(
                        skeleton_pose,
                        pose_3d,
                        root_transform.left_foot_grounded,
                        root_transform.right_foot_grounded
                    )
                
                self._motion_type = self._root_motion.detect_motion_type()
        else:
            self._pose_count = 0
        
        if self._show_skeleton and pose_2d:
            display_frame = self._pose_estimator.draw_pose(frame, pose_2d)
        else:
            display_frame = frame
        
        return display_frame, skeleton_pose
    
    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        y = 35
        cv2.putText(frame, "MoCap to UE5", (20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y += 30
        state_text = self._state.name
        if self._is_recording:
            state_text = "RECORDING"
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(frame, f"State: {state_text}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        y += 25
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        cv2.putText(frame, f"Pose: {self._pose_count}/33 joints", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        floor_text = "Detected" if self._floor_detector.has_valid_floor else "Detecting..."
        cv2.putText(frame, f"Floor: {floor_text}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        cv2.putText(frame, f"Motion: {self._motion_type}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self._is_recording:
            y += 25
            cv2.putText(frame, f"Frames: {len(self._recorded_poses)}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        controls_y = h - 80
        cv2.rectangle(frame, (10, controls_y - 10), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        controls = [
            "SPACE:Pause", "R:Record", "E:Export", 
            "S:Skeleton", "I:IK", "F:Floor", "Q:Quit"
        ]
        x = 20
        for ctrl in controls:
            cv2.putText(frame, ctrl, (x, controls_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            x += 120
        
        toggles = []
        if self._show_skeleton:
            toggles.append("[S]")
        if self._enable_ik:
            toggles.append("[I]")
        if self._enable_floor:
            toggles.append("[F]")
        
        toggle_text = " ".join(toggles)
        cv2.putText(frame, f"Active: {toggle_text}", (20, controls_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
    
    def _show_idle_screen(self) -> None:
        """Show idle screen when no video is loaded."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        
        cv2.putText(frame, "MoCap to UE5 Animation", (400, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, "No video loaded", (500, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
        cv2.putText(frame, "Press Q to quit", (530, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        cv2.imshow(self.WINDOW_NAME, frame)
    
    def _show_error_screen(self, message: str) -> None:
        """Show error screen."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (40, 40, 60)
        
        cv2.putText(frame, "Error", (580, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(frame, message, (400, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, "Press any key to exit", (500, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        cv2.imshow(self.WINDOW_NAME, frame)
        cv2.waitKey(0)
        self._running = False
    
    def _handle_key(self, key: int) -> None:
        """Handle keyboard input."""
        if key == ord('q') or key == 27:  # Q or ESC
            self._running = False
            
        elif key == ord(' '):  # Space - pause/resume
            if self._state == AppState.PLAYING:
                self._state = AppState.PAUSED
                self._video_capture.pause()
                self.logger.info("Paused")
            elif self._state == AppState.PAUSED:
                self._state = AppState.PLAYING
                self._video_capture.resume()
                self.logger.info("Resumed")
                
        elif key == ord('r'):  # R - toggle recording
            if self._is_recording:
                self._is_recording = False
                self.logger.info(f"Recording stopped: {len(self._recorded_poses)} frames")
            else:
                self._is_recording = True
                self._recorded_poses.clear()
                self.logger.info("Recording started")
                
        elif key == ord('e'):  # E - export
            self._export_animation()
            
        elif key == ord('s'):  # S - toggle skeleton
            self._show_skeleton = not self._show_skeleton
            self.logger.info(f"Skeleton overlay: {self._show_skeleton}")
            
        elif key == ord('i'):  # I - toggle IK
            self._enable_ik = not self._enable_ik
            self.logger.info(f"IK: {self._enable_ik}")
            
        elif key == ord('f'):  # F - toggle floor detection
            self._enable_floor = not self._enable_floor
            self.logger.info(f"Floor detection: {self._enable_floor}")
    
    def _export_animation(self) -> None:
        """Export recorded animation to FBX."""
        if not self._recorded_poses:
            self.logger.warning("No animation to export")
            return
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"animation_{timestamp}"
            
            path = self._fbx_exporter.export(self._recorded_poses, filename)
            self.logger.info(f"Exported FBX to: {path}")
            
            blender_path = self._fbx_exporter.export_for_blender(
                self._recorded_poses, filename
            )
            self.logger.info(f"Exported Blender script to: {blender_path}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._video_capture:
            self._video_capture.close()
        
        if self._pose_estimator:
            self._pose_estimator.close()
        
        if self._pose_reconstructor:
            self._pose_reconstructor.close()
        
        cv2.destroyAllWindows()
        self.logger.info("Cleanup complete")


def run_cv_app(config: Config, source: Optional[str] = None) -> int:
    """Run the OpenCV-based application."""
    try:
        app = CVWindow(config)
        app.run(source)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
