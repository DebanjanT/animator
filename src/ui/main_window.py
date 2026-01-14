"""Main GTK application window with full motion capture pipeline"""

import threading
import time
from pathlib import Path
from typing import Optional

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf

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


class MainWindow(Gtk.Window):
    """Main application window with full motion capture pipeline."""
    
    def __init__(self, config: Config):
        super().__init__(title="MoCap to UE5 Animation")
        self.config = config
        self.logger = get_logger("ui")
        
        self.set_default_size(1400, 800)
        self.set_position(Gtk.WindowPosition.CENTER)
        
        self._video_capture: Optional[VideoCapture] = None
        self._pose_estimator: Optional[PoseEstimator2D] = None
        self._pose_reconstructor: Optional[PoseReconstructor3D] = None
        self._floor_detector: Optional[FloorDetector] = None
        self._root_motion: Optional[RootMotionExtractor] = None
        self._skeleton_solver: Optional[SkeletonSolver] = None
        self._ik_solver: Optional[IKSolver] = None
        self._fbx_exporter: Optional[FBXExporter] = None
        
        self._is_processing = False
        self._is_recording = False
        self._recorded_poses: list = []
        self._frame_timer = FrameTimer()
        self._process_thread: Optional[threading.Thread] = None
        self._stop_processing = threading.Event()
        
        self._setup_ui()
        self._setup_css()
        self._initialize_pipeline()
        
        self.logger.info("Main window initialized")
    
    def _setup_css(self) -> None:
        """Setup custom CSS styling."""
        css = b"""
        .header-label {
            font-size: 24px;
            font-weight: bold;
            margin: 10px;
        }
        .status-label {
            font-size: 14px;
            color: #666;
        }
        .recording {
            color: #ff4444;
            font-weight: bold;
        }
        .control-button {
            min-width: 100px;
            min-height: 40px;
        }
        .stats-label {
            font-family: monospace;
            font-size: 12px;
        }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(main_box)
        
        header_bar = Gtk.HeaderBar()
        header_bar.set_show_close_button(True)
        header_bar.set_title("MoCap to UE5 Animation")
        header_bar.set_subtitle("Video to Skeletal Animation Converter")
        self.set_titlebar(header_bar)
        
        content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        content_box.set_margin_top(10)
        content_box.set_margin_bottom(10)
        content_box.set_margin_start(10)
        content_box.set_margin_end(10)
        main_box.pack_start(content_box, True, True, 0)
        
        left_panel = self._create_video_panel()
        content_box.pack_start(left_panel, True, True, 0)
        
        right_panel = self._create_control_panel()
        content_box.pack_start(right_panel, False, False, 0)
        
        status_bar = self._create_status_bar()
        main_box.pack_end(status_bar, False, False, 0)
    
    def _create_video_panel(self) -> Gtk.Widget:
        """Create the video display panel."""
        frame = Gtk.Frame(label="Video Preview")
        frame.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        box.set_margin_top(5)
        box.set_margin_bottom(5)
        box.set_margin_start(5)
        box.set_margin_end(5)
        frame.add(box)
        
        self._video_image = Gtk.Image()
        self._video_image.set_size_request(960, 540)
        
        placeholder = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, 960, 540)
        placeholder.fill(0x333333ff)
        self._video_image.set_from_pixbuf(placeholder)
        
        box.pack_start(self._video_image, True, True, 0)
        
        playback_box = Gtk.Box(spacing=10)
        playback_box.set_halign(Gtk.Align.CENTER)
        
        self._play_button = Gtk.Button(label="▶ Play")
        self._play_button.connect("clicked", self._on_play_clicked)
        playback_box.pack_start(self._play_button, False, False, 0)
        
        self._pause_button = Gtk.Button(label="⏸ Pause")
        self._pause_button.connect("clicked", self._on_pause_clicked)
        self._pause_button.set_sensitive(False)
        playback_box.pack_start(self._pause_button, False, False, 0)
        
        self._stop_button = Gtk.Button(label="⏹ Stop")
        self._stop_button.connect("clicked", self._on_stop_clicked)
        self._stop_button.set_sensitive(False)
        playback_box.pack_start(self._stop_button, False, False, 0)
        
        box.pack_start(playback_box, False, False, 5)
        
        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_show_text(True)
        self._progress_bar.set_text("No video loaded")
        box.pack_start(self._progress_bar, False, False, 0)
        
        return frame
    
    def _create_control_panel(self) -> Gtk.Widget:
        """Create the control panel."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_size_request(350, -1)
        
        source_frame = Gtk.Frame(label="Video Source")
        source_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        source_box.set_margin_top(10)
        source_box.set_margin_bottom(10)
        source_box.set_margin_start(10)
        source_box.set_margin_end(10)
        source_frame.add(source_box)
        
        self._webcam_radio = Gtk.RadioButton.new_with_label(None, "Webcam")
        source_box.pack_start(self._webcam_radio, False, False, 0)
        
        self._file_radio = Gtk.RadioButton.new_with_label_from_widget(
            self._webcam_radio, "Video File"
        )
        source_box.pack_start(self._file_radio, False, False, 0)
        
        file_box = Gtk.Box(spacing=5)
        self._file_entry = Gtk.Entry()
        self._file_entry.set_placeholder_text("Select video file...")
        file_box.pack_start(self._file_entry, True, True, 0)
        
        browse_button = Gtk.Button(label="Browse")
        browse_button.connect("clicked", self._on_browse_clicked)
        file_box.pack_start(browse_button, False, False, 0)
        
        source_box.pack_start(file_box, False, False, 0)
        
        load_button = Gtk.Button(label="Load Source")
        load_button.connect("clicked", self._on_load_source)
        source_box.pack_start(load_button, False, False, 5)
        
        box.pack_start(source_frame, False, False, 0)
        
        pipeline_frame = Gtk.Frame(label="Pipeline Settings")
        pipeline_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        pipeline_box.set_margin_top(10)
        pipeline_box.set_margin_bottom(10)
        pipeline_box.set_margin_start(10)
        pipeline_box.set_margin_end(10)
        pipeline_frame.add(pipeline_box)
        
        self._show_skeleton_check = Gtk.CheckButton(label="Show Skeleton Overlay")
        self._show_skeleton_check.set_active(True)
        pipeline_box.pack_start(self._show_skeleton_check, False, False, 0)
        
        self._enable_ik_check = Gtk.CheckButton(label="Enable IK")
        self._enable_ik_check.set_active(True)
        pipeline_box.pack_start(self._enable_ik_check, False, False, 0)
        
        self._foot_lock_check = Gtk.CheckButton(label="Foot Locking")
        self._foot_lock_check.set_active(True)
        pipeline_box.pack_start(self._foot_lock_check, False, False, 0)
        
        self._floor_detect_check = Gtk.CheckButton(label="Floor Detection")
        self._floor_detect_check.set_active(True)
        pipeline_box.pack_start(self._floor_detect_check, False, False, 0)
        
        box.pack_start(pipeline_frame, False, False, 0)
        
        record_frame = Gtk.Frame(label="Recording")
        record_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        record_box.set_margin_top(10)
        record_box.set_margin_bottom(10)
        record_box.set_margin_start(10)
        record_box.set_margin_end(10)
        record_frame.add(record_box)
        
        self._record_button = Gtk.Button(label="⏺ Start Recording")
        self._record_button.connect("clicked", self._on_record_clicked)
        self._record_button.set_sensitive(False)
        record_box.pack_start(self._record_button, False, False, 0)
        
        self._record_status = Gtk.Label(label="Not recording")
        record_box.pack_start(self._record_status, False, False, 0)
        
        self._frame_count_label = Gtk.Label(label="Frames: 0")
        record_box.pack_start(self._frame_count_label, False, False, 0)
        
        box.pack_start(record_frame, False, False, 0)
        
        export_frame = Gtk.Frame(label="Export")
        export_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        export_box.set_margin_top(10)
        export_box.set_margin_bottom(10)
        export_box.set_margin_start(10)
        export_box.set_margin_end(10)
        export_frame.add(export_box)
        
        name_box = Gtk.Box(spacing=5)
        name_label = Gtk.Label(label="Name:")
        name_box.pack_start(name_label, False, False, 0)
        self._export_name_entry = Gtk.Entry()
        self._export_name_entry.set_text("animation")
        name_box.pack_start(self._export_name_entry, True, True, 0)
        export_box.pack_start(name_box, False, False, 0)
        
        self._export_button = Gtk.Button(label="Export FBX")
        self._export_button.connect("clicked", self._on_export_clicked)
        self._export_button.set_sensitive(False)
        export_box.pack_start(self._export_button, False, False, 5)
        
        self._export_blender_button = Gtk.Button(label="Export Blender Script")
        self._export_blender_button.connect("clicked", self._on_export_blender_clicked)
        self._export_blender_button.set_sensitive(False)
        export_box.pack_start(self._export_blender_button, False, False, 0)
        
        box.pack_start(export_frame, False, False, 0)
        
        stats_frame = Gtk.Frame(label="Statistics")
        stats_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        stats_box.set_margin_top(10)
        stats_box.set_margin_bottom(10)
        stats_box.set_margin_start(10)
        stats_box.set_margin_end(10)
        stats_frame.add(stats_box)
        
        self._fps_label = Gtk.Label(label="FPS: --")
        self._fps_label.set_halign(Gtk.Align.START)
        self._fps_label.get_style_context().add_class("stats-label")
        stats_box.pack_start(self._fps_label, False, False, 0)
        
        self._pose_label = Gtk.Label(label="Pose: --")
        self._pose_label.set_halign(Gtk.Align.START)
        self._pose_label.get_style_context().add_class("stats-label")
        stats_box.pack_start(self._pose_label, False, False, 0)
        
        self._floor_label = Gtk.Label(label="Floor: --")
        self._floor_label.set_halign(Gtk.Align.START)
        self._floor_label.get_style_context().add_class("stats-label")
        stats_box.pack_start(self._floor_label, False, False, 0)
        
        self._motion_label = Gtk.Label(label="Motion: --")
        self._motion_label.set_halign(Gtk.Align.START)
        self._motion_label.get_style_context().add_class("stats-label")
        stats_box.pack_start(self._motion_label, False, False, 0)
        
        box.pack_start(stats_frame, True, True, 0)
        
        return box
    
    def _create_status_bar(self) -> Gtk.Widget:
        """Create the status bar."""
        box = Gtk.Box(spacing=10)
        box.set_margin_top(5)
        box.set_margin_bottom(5)
        box.set_margin_start(10)
        box.set_margin_end(10)
        
        self._status_label = Gtk.Label(label="Ready")
        self._status_label.set_halign(Gtk.Align.START)
        box.pack_start(self._status_label, True, True, 0)
        
        version_label = Gtk.Label(label="v0.1.0")
        version_label.set_halign(Gtk.Align.END)
        box.pack_end(version_label, False, False, 0)
        
        return box
    
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
            
            self._set_status("Pipeline initialized")
            self.logger.info("All pipeline components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            self._set_status(f"Error: {e}")
    
    def _on_browse_clicked(self, button: Gtk.Button) -> None:
        """Handle browse button click."""
        dialog = Gtk.FileChooserDialog(
            title="Select Video File",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )
        
        filter_video = Gtk.FileFilter()
        filter_video.set_name("Video files")
        filter_video.add_mime_type("video/*")
        filter_video.add_pattern("*.mp4")
        filter_video.add_pattern("*.avi")
        filter_video.add_pattern("*.mov")
        filter_video.add_pattern("*.mkv")
        dialog.add_filter(filter_video)
        
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self._file_entry.set_text(dialog.get_filename())
            self._file_radio.set_active(True)
        
        dialog.destroy()
    
    def _on_load_source(self, button: Gtk.Button) -> None:
        """Handle load source button click."""
        if self._webcam_radio.get_active():
            source = "webcam"
        else:
            source = self._file_entry.get_text()
            if not source:
                self._set_status("Please select a video file")
                return
        
        if self._video_capture.open(source):
            self._set_status(f"Loaded: {source}")
            self._play_button.set_sensitive(True)
            self._record_button.set_sensitive(True)
        else:
            self._set_status(f"Failed to open: {source}")
    
    def _on_play_clicked(self, button: Gtk.Button) -> None:
        """Handle play button click."""
        if self._video_capture is None:
            return
        
        self._stop_processing.clear()
        self._is_processing = True
        
        self._video_capture.start()
        
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        self._play_button.set_sensitive(False)
        self._pause_button.set_sensitive(True)
        self._stop_button.set_sensitive(True)
        
        self._set_status("Processing...")
    
    def _on_pause_clicked(self, button: Gtk.Button) -> None:
        """Handle pause button click."""
        if self._video_capture:
            if self._video_capture.is_paused:
                self._video_capture.resume()
                self._pause_button.set_label("⏸ Pause")
            else:
                self._video_capture.pause()
                self._pause_button.set_label("▶ Resume")
    
    def _on_stop_clicked(self, button: Gtk.Button) -> None:
        """Handle stop button click."""
        self._stop_processing.set()
        self._is_processing = False
        
        if self._video_capture:
            self._video_capture.stop()
        
        self._play_button.set_sensitive(True)
        self._pause_button.set_sensitive(False)
        self._stop_button.set_sensitive(False)
        self._pause_button.set_label("⏸ Pause")
        
        self._set_status("Stopped")
    
    def _on_record_clicked(self, button: Gtk.Button) -> None:
        """Handle record button click."""
        if self._is_recording:
            self._is_recording = False
            self._record_button.set_label("⏺ Start Recording")
            self._record_status.set_text(f"Recorded {len(self._recorded_poses)} frames")
            self._record_status.get_style_context().remove_class("recording")
            
            if self._recorded_poses:
                self._export_button.set_sensitive(True)
                self._export_blender_button.set_sensitive(True)
        else:
            self._is_recording = True
            self._recorded_poses.clear()
            self._record_button.set_label("⏹ Stop Recording")
            self._record_status.set_text("Recording...")
            self._record_status.get_style_context().add_class("recording")
            
            self._export_button.set_sensitive(False)
            self._export_blender_button.set_sensitive(False)
    
    def _on_export_clicked(self, button: Gtk.Button) -> None:
        """Handle export FBX button click."""
        if not self._recorded_poses:
            self._set_status("No animation to export")
            return
        
        name = self._export_name_entry.get_text() or "animation"
        
        try:
            path = self._fbx_exporter.export(self._recorded_poses, name)
            self._set_status(f"Exported to {path}")
            
            dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.INFO,
                buttons=Gtk.ButtonsType.OK,
                text="Export Complete"
            )
            dialog.format_secondary_text(f"Animation exported to:\n{path}")
            dialog.run()
            dialog.destroy()
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            self._set_status(f"Export failed: {e}")
    
    def _on_export_blender_clicked(self, button: Gtk.Button) -> None:
        """Handle export Blender script button click."""
        if not self._recorded_poses:
            self._set_status("No animation to export")
            return
        
        name = self._export_name_entry.get_text() or "animation"
        
        try:
            path = self._fbx_exporter.export_for_blender(self._recorded_poses, name)
            self._set_status(f"Exported Blender script to {path}")
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            self._set_status(f"Export failed: {e}")
    
    def _process_loop(self) -> None:
        """Main processing loop (runs in separate thread)."""
        while not self._stop_processing.is_set():
            if self._video_capture is None or not self._video_capture.is_playing:
                time.sleep(0.01)
                continue
            
            self._frame_timer.start()
            
            frame_result = self._video_capture.read()
            if frame_result is None:
                if self._video_capture.state == CaptureState.STOPPED:
                    GLib.idle_add(self._on_stop_clicked, None)
                    break
                continue
            
            frame = frame_result.frame
            timestamp = frame_result.timestamp
            
            pose_2d = self._pose_estimator.process(frame, timestamp)
            
            pose_3d = None
            skeleton_pose = None
            
            if pose_2d and pose_2d.is_valid:
                pose_3d = self._pose_reconstructor.process(frame, timestamp)
                
                if pose_3d and pose_3d.is_valid:
                    if self._floor_detect_check.get_active():
                        self._floor_detector.update(pose_3d)
                    
                    floor = self._floor_detector.current_plane
                    root_transform = self._root_motion.process(pose_3d, floor)
                    
                    skeleton_pose = self._skeleton_solver.solve(pose_3d, root_transform)
                    
                    if self._enable_ik_check.get_active():
                        skeleton_pose = self._ik_solver.process(
                            skeleton_pose,
                            pose_3d,
                            root_transform.left_foot_grounded,
                            root_transform.right_foot_grounded
                        )
                    
                    if self._is_recording:
                        self._recorded_poses.append(skeleton_pose)
            
            if self._show_skeleton_check.get_active() and pose_2d:
                display_frame = self._pose_estimator.draw_pose(frame, pose_2d)
            else:
                display_frame = frame
            
            self._frame_timer.stop()
            
            GLib.idle_add(
                self._update_display,
                display_frame,
                pose_2d,
                pose_3d,
                skeleton_pose
            )
        
        self.logger.info("Processing loop ended")
    
    def _update_display(
        self,
        frame: np.ndarray,
        pose_2d,
        pose_3d,
        skeleton_pose
    ) -> bool:
        """Update display (called from main thread)."""
        h, w = frame.shape[:2]
        
        display_w, display_h = 960, 540
        scale = min(display_w / w, display_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            frame_resized.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            new_w,
            new_h,
            new_w * 3
        )
        self._video_image.set_from_pixbuf(pixbuf)
        
        fps = self._frame_timer.fps
        self._fps_label.set_text(f"FPS: {fps:.1f}")
        
        if pose_2d:
            self._pose_label.set_text(f"Pose: {pose_2d.num_visible_joints}/33 joints")
        else:
            self._pose_label.set_text("Pose: Not detected")
        
        if self._floor_detector.has_valid_floor:
            conf = self._floor_detector.current_plane.confidence
            self._floor_label.set_text(f"Floor: Detected ({conf:.0%})")
        else:
            self._floor_label.set_text("Floor: Detecting...")
        
        if self._root_motion.current_root:
            motion_type = self._root_motion.detect_motion_type()
            speed = self._root_motion.current_root.speed
            self._motion_label.set_text(f"Motion: {motion_type} ({speed:.1f} m/s)")
        
        if self._is_recording:
            self._frame_count_label.set_text(f"Frames: {len(self._recorded_poses)}")
        
        if self._video_capture and not self._video_capture.is_webcam:
            progress = self._video_capture.progress
            current = self._video_capture.current_time
            total = self._video_capture.duration
            self._progress_bar.set_fraction(progress)
            self._progress_bar.set_text(f"{current:.1f}s / {total:.1f}s")
        else:
            self._progress_bar.set_text("Live")
        
        return False
    
    def _set_status(self, message: str) -> None:
        """Set status bar message."""
        self._status_label.set_text(message)
        self.logger.info(message)
    
    def do_destroy(self) -> None:
        """Clean up on window destroy."""
        self._stop_processing.set()
        
        if self._process_thread:
            self._process_thread.join(timeout=1.0)
        
        if self._video_capture:
            self._video_capture.close()
        
        if self._pose_estimator:
            self._pose_estimator.close()
        
        if self._pose_reconstructor:
            self._pose_reconstructor.close()
        
        self.logger.info("Window destroyed, resources cleaned up")
        
        Gtk.Window.do_destroy(self)
