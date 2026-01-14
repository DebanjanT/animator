"""GTK4 application window with full motion capture pipeline"""

import threading
import time
from pathlib import Path
from typing import Optional, List, Any
from enum import Enum, auto
from dataclasses import dataclass

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf, Gio

import cv2
import numpy as np

from src.core import Config, get_logger, FrameTimer
from src.video.capture import VideoCapture, CaptureState
from src.pose.unified_processor import UnifiedPoseProcessor
from src.pose.reconstructor_3d import Pose3D
from src.pose.hand_tracker import HandTracker, HandsData
from src.motion.floor_detector import FloorDetector
from src.motion.root_motion import RootMotionExtractor
from src.motion.skeleton_solver import SkeletonSolver, SkeletonPose
from src.ik.solver import IKSolver
from src.export.fbx_exporter import FBXExporter
from src.ui.skeleton_viewer import SkeletonViewer3D


class AppState(Enum):
    HOME = auto()
    LIVE = auto()
    PREPROCESSING = auto()
    PLAYBACK = auto()
    PAUSED = auto()


class ProcessingMode(Enum):
    LIVE = "live"
    PREPROCESS = "preprocess"


@dataclass
class RecordedFrame:
    """Single recorded frame with all tracking data."""
    frame_number: int
    timestamp: float
    frame: np.ndarray
    pose_2d: Any
    pose_3d: Any
    hands_data: Optional[HandsData]
    skeleton_pose: Optional[SkeletonPose]


class MoCapApplication(Gtk.Application):
    """GTK4 Application for MoCap to UE5 Animation."""
    
    def __init__(self, config: Config):
        super().__init__(
            application_id="com.mocap.ue5animation",
            flags=Gio.ApplicationFlags.FLAGS_NONE
        )
        self.config = config
        self.logger = get_logger("ui.gtk4")
        self.window = None
    
    def do_activate(self):
        """Called when the application is activated."""
        if not self.window:
            self.window = MainWindow(application=self, config=self.config)
        self.window.present()


class MainWindow(Gtk.ApplicationWindow):
    """Main application window with full motion capture pipeline."""
    
    def __init__(self, application: Gtk.Application, config: Config):
        super().__init__(application=application, title="MoCap to UE5 Animation")
        self.config = config
        self.logger = get_logger("ui.gtk4")
        
        self.set_default_size(1400, 900)
        
        self._video_capture: Optional[VideoCapture] = None
        self._pose_processor: Optional[UnifiedPoseProcessor] = None
        self._hand_tracker: Optional[HandTracker] = None
        self._floor_detector: Optional[FloorDetector] = None
        self._root_motion: Optional[RootMotionExtractor] = None
        self._skeleton_solver: Optional[SkeletonSolver] = None
        self._ik_solver: Optional[IKSolver] = None
        self._fbx_exporter: Optional[FBXExporter] = None
        self._skeleton_viewer: Optional[SkeletonViewer3D] = None
        
        self._state = AppState.HOME
        self._processing_mode = ProcessingMode.LIVE
        self._is_recording = False
        self._recorded_frames: list = []
        self._recorded_poses: list = []
        self._frame_timer = FrameTimer()
        self._process_thread: Optional[threading.Thread] = None
        self._stop_processing = threading.Event()
        
        self._show_skeleton = True
        self._show_hands = True
        self._show_3d_view = True
        self._enable_ik = True
        self._enable_floor = True
        self._enable_hands = config.get("hand_tracking", {}).get("enabled", False)
        
        self._current_frame: Optional[np.ndarray] = None
        self._current_pose_3d = None
        self._fps = 0.0
        self._playback_index = 0
        self._playback_playing = False
        self._preprocess_progress = 0.0
        self._preprocess_total = 0
        self._preprocess_thread: Optional[threading.Thread] = None
        self._frame_count = 0
        
        self._setup_ui()
        self.logger.info("GTK4 Main window initialized")
    
    def _setup_ui(self) -> None:
        """Setup the user interface."""
        header = Gtk.HeaderBar()
        header.set_show_title_buttons(True)
        self.set_titlebar(header)
        
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        title_label = Gtk.Label(label="MoCap to UE5 Animation")
        title_label.add_css_class("title")
        subtitle_label = Gtk.Label(label="Video to Skeletal Animation")
        subtitle_label.add_css_class("subtitle")
        title_box.append(title_label)
        title_box.append(subtitle_label)
        header.set_title_widget(title_box)
        
        home_btn = Gtk.Button(label="Home")
        home_btn.connect("clicked", self._on_home_clicked)
        header.pack_start(home_btn)
        
        settings_btn = Gtk.Button(icon_name="emblem-system-symbolic")
        settings_btn.connect("clicked", self._on_settings_clicked)
        header.pack_end(settings_btn)
        
        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._stack.set_transition_duration(200)
        self.set_child(self._stack)
        
        home_page = self._create_home_page()
        self._stack.add_named(home_page, "home")
        
        processing_page = self._create_processing_page()
        self._stack.add_named(processing_page, "processing")
        
        self._stack.set_visible_child_name("home")
    
    def _create_home_page(self) -> Gtk.Widget:
        """Create the home/welcome page."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)
        box.set_margin_top(50)
        box.set_margin_bottom(50)
        
        welcome_label = Gtk.Label(label="Welcome to MoCap")
        welcome_label.add_css_class("title-1")
        box.append(welcome_label)
        
        desc_label = Gtk.Label(label="Convert video to UE5 skeletal animation")
        desc_label.add_css_class("dim-label")
        box.append(desc_label)
        
        button_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        button_box.set_margin_top(30)
        button_box.set_halign(Gtk.Align.CENTER)
        
        open_file_btn = Gtk.Button(label="Open Video File")
        open_file_btn.add_css_class("suggested-action")
        open_file_btn.add_css_class("pill")
        open_file_btn.set_size_request(250, 50)
        open_file_btn.connect("clicked", self._on_open_file_clicked)
        button_box.append(open_file_btn)
        
        webcam_btn = Gtk.Button(label="Use Webcam")
        webcam_btn.add_css_class("pill")
        webcam_btn.set_size_request(250, 50)
        webcam_btn.connect("clicked", self._on_webcam_clicked)
        button_box.append(webcam_btn)
        
        box.append(button_box)
        
        mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        mode_box.set_halign(Gtk.Align.CENTER)
        mode_box.set_margin_top(30)
        
        mode_label = Gtk.Label(label="Processing Mode:")
        mode_box.append(mode_label)
        
        self._mode_dropdown = Gtk.DropDown.new_from_strings(["Live", "Preprocess"])
        self._mode_dropdown.set_selected(0)
        self._mode_dropdown.connect("notify::selected", self._on_mode_changed)
        mode_box.append(self._mode_dropdown)
        
        box.append(mode_box)
        
        settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        settings_box.set_margin_top(30)
        settings_box.set_halign(Gtk.Align.CENTER)
        
        settings_label = Gtk.Label(label="Quick Settings")
        settings_label.add_css_class("heading")
        settings_box.append(settings_label)
        
        self._skeleton_switch = self._create_switch_row("Show Skeleton", self._show_skeleton)
        settings_box.append(self._skeleton_switch)
        
        self._hands_switch = self._create_switch_row("Hand Tracking", self._enable_hands)
        settings_box.append(self._hands_switch)
        
        self._3d_switch = self._create_switch_row("3D Preview", self._show_3d_view)
        settings_box.append(self._3d_switch)
        
        self._ik_switch = self._create_switch_row("IK Solver", self._enable_ik)
        settings_box.append(self._ik_switch)
        
        box.append(settings_box)
        
        return box
    
    def _create_switch_row(self, label: str, active: bool) -> Gtk.Box:
        """Create a labeled switch row."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row.set_size_request(250, -1)
        
        lbl = Gtk.Label(label=label)
        lbl.set_halign(Gtk.Align.START)
        lbl.set_hexpand(True)
        row.append(lbl)
        
        switch = Gtk.Switch()
        switch.set_active(active)
        switch.set_valign(Gtk.Align.CENTER)
        row.append(switch)
        
        row.switch = switch
        return row
    
    def _create_processing_page(self) -> Gtk.Widget:
        """Create the main processing page."""
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.set_margin_top(10)
        main_box.set_margin_bottom(10)
        main_box.set_margin_start(10)
        main_box.set_margin_end(10)
        
        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        left_box.set_hexpand(True)
        
        video_frame = Gtk.Frame(label="Video Preview")
        video_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        video_box.set_margin_top(5)
        video_box.set_margin_bottom(5)
        video_box.set_margin_start(5)
        video_box.set_margin_end(5)
        
        self._video_picture = Gtk.Picture()
        self._video_picture.set_size_request(960, 540)
        self._video_picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        video_box.append(self._video_picture)
        
        controls_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        controls_box.set_halign(Gtk.Align.CENTER)
        controls_box.set_margin_top(10)
        
        self._play_btn = Gtk.Button(icon_name="media-playback-start-symbolic")
        self._play_btn.connect("clicked", self._on_play_clicked)
        controls_box.append(self._play_btn)
        
        self._pause_btn = Gtk.Button(icon_name="media-playback-pause-symbolic")
        self._pause_btn.connect("clicked", self._on_pause_clicked)
        self._pause_btn.set_sensitive(False)
        controls_box.append(self._pause_btn)
        
        self._stop_btn = Gtk.Button(icon_name="media-playback-stop-symbolic")
        self._stop_btn.connect("clicked", self._on_stop_clicked)
        self._stop_btn.set_sensitive(False)
        controls_box.append(self._stop_btn)
        
        self._record_btn = Gtk.Button(icon_name="media-record-symbolic")
        self._record_btn.connect("clicked", self._on_record_clicked)
        self._record_btn.set_sensitive(False)
        controls_box.append(self._record_btn)
        
        video_box.append(controls_box)
        
        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_show_text(True)
        self._progress_bar.set_text("Ready")
        video_box.append(self._progress_bar)
        
        scrub_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        scrub_box.set_margin_top(5)
        
        self._prev_btn = Gtk.Button(icon_name="go-previous-symbolic")
        self._prev_btn.connect("clicked", self._on_prev_frame)
        self._prev_btn.set_sensitive(False)
        scrub_box.append(self._prev_btn)
        
        self._scrub_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 1)
        self._scrub_scale.set_hexpand(True)
        self._scrub_scale.set_draw_value(False)
        self._scrub_scale.connect("value-changed", self._on_scrub_changed)
        self._scrub_scale.set_sensitive(False)
        scrub_box.append(self._scrub_scale)
        
        self._next_btn = Gtk.Button(icon_name="go-next-symbolic")
        self._next_btn.connect("clicked", self._on_next_frame)
        self._next_btn.set_sensitive(False)
        scrub_box.append(self._next_btn)
        
        video_box.append(scrub_box)
        
        video_frame.set_child(video_box)
        left_box.append(video_frame)
        
        main_box.append(left_box)
        
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        right_box.set_size_request(350, -1)
        
        stats_frame = Gtk.Frame(label="Statistics")
        stats_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        stats_box.set_margin_top(10)
        stats_box.set_margin_bottom(10)
        stats_box.set_margin_start(10)
        stats_box.set_margin_end(10)
        
        self._fps_label = Gtk.Label(label="FPS: --")
        self._fps_label.set_halign(Gtk.Align.START)
        stats_box.append(self._fps_label)
        
        self._state_label = Gtk.Label(label="State: HOME")
        self._state_label.set_halign(Gtk.Align.START)
        stats_box.append(self._state_label)
        
        self._pose_label = Gtk.Label(label="Pose: --")
        self._pose_label.set_halign(Gtk.Align.START)
        stats_box.append(self._pose_label)
        
        self._frames_label = Gtk.Label(label="Recorded: 0 frames")
        self._frames_label.set_halign(Gtk.Align.START)
        stats_box.append(self._frames_label)
        
        stats_frame.set_child(stats_box)
        right_box.append(stats_frame)
        
        export_frame = Gtk.Frame(label="Export")
        export_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        export_box.set_margin_top(10)
        export_box.set_margin_bottom(10)
        export_box.set_margin_start(10)
        export_box.set_margin_end(10)
        
        self._export_entry = Gtk.Entry()
        self._export_entry.set_placeholder_text("Animation name...")
        self._export_entry.set_text("animation")
        export_box.append(self._export_entry)
        
        self._export_btn = Gtk.Button(label="Export FBX")
        self._export_btn.connect("clicked", self._on_export_clicked)
        self._export_btn.set_sensitive(False)
        export_box.append(self._export_btn)
        
        export_frame.set_child(export_box)
        right_box.append(export_frame)
        
        self._3d_frame = Gtk.Frame(label="3D Preview")
        self._3d_picture = Gtk.Picture()
        self._3d_picture.set_size_request(330, 330)
        self._3d_frame.set_child(self._3d_picture)
        right_box.append(self._3d_frame)
        
        main_box.append(right_box)
        
        return main_box
    
    def _initialize_pipeline(self) -> None:
        """Initialize all pipeline components."""
        try:
            self._show_skeleton = self._skeleton_switch.switch.get_active()
            self._enable_hands = self._hands_switch.switch.get_active()
            self._show_3d_view = self._3d_switch.switch.get_active()
            self._enable_ik = self._ik_switch.switch.get_active()
            
            model_type = self.config.get("pose_estimation", {}).get("model_type", "full")
            self._pose_processor = UnifiedPoseProcessor(self.config, model_type=model_type)
            
            if self._enable_hands:
                self._hand_tracker = HandTracker(self.config)
            
            self._floor_detector = FloorDetector(self.config)
            self._root_motion = RootMotionExtractor(self.config)
            self._skeleton_solver = SkeletonSolver(self.config)
            self._ik_solver = IKSolver(self.config)
            self._fbx_exporter = FBXExporter(self.config)
            self._skeleton_viewer = SkeletonViewer3D(self.config, width=330, height=330)
            
            self.logger.info("Pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def _on_home_clicked(self, button: Gtk.Button) -> None:
        """Handle home button click."""
        self._stop_processing.set()
        if self._video_capture:
            self._video_capture.stop()
        self._state = AppState.HOME
        self._stack.set_visible_child_name("home")
    
    def _on_settings_clicked(self, button: Gtk.Button) -> None:
        """Handle settings button click."""
        dialog = Gtk.Dialog(title="Settings", transient_for=self, modal=True)
        dialog.add_button("Close", Gtk.ResponseType.CLOSE)
        
        content = dialog.get_content_area()
        content.set_margin_top(20)
        content.set_margin_bottom(20)
        content.set_margin_start(20)
        content.set_margin_end(20)
        
        label = Gtk.Label(label="Settings dialog - configure options here")
        content.append(label)
        
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()
    
    def _on_open_file_clicked(self, button: Gtk.Button) -> None:
        """Handle open file button click."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Video File")
        
        filter_video = Gtk.FileFilter()
        filter_video.set_name("Video files")
        filter_video.add_mime_type("video/*")
        filter_video.add_pattern("*.mp4")
        filter_video.add_pattern("*.avi")
        filter_video.add_pattern("*.mov")
        filter_video.add_pattern("*.mkv")
        
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(filter_video)
        dialog.set_filters(filters)
        
        dialog.open(self, None, self._on_file_selected)
    
    def _on_file_selected(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle file selection result."""
        try:
            file = dialog.open_finish(result)
            if file:
                path = file.get_path()
                self.logger.info(f"Selected file: {path}")
                self._start_video(path)
        except GLib.Error as e:
            if e.code != Gtk.DialogError.DISMISSED:
                self.logger.error(f"File dialog error: {e}")
    
    def _on_webcam_clicked(self, button: Gtk.Button) -> None:
        """Handle webcam button click."""
        self._start_video("webcam")
    
    def _on_mode_changed(self, dropdown: Gtk.DropDown, param) -> None:
        """Handle processing mode change."""
        selected = dropdown.get_selected()
        self._processing_mode = ProcessingMode.LIVE if selected == 0 else ProcessingMode.PREPROCESS
        self.logger.info(f"Processing mode: {self._processing_mode.value}")
    
    def _start_video(self, source: str) -> None:
        """Start video processing."""
        self._initialize_pipeline()
        
        self._video_capture = VideoCapture(self.config)
        if not self._video_capture.open(source):
            self.logger.error(f"Failed to open: {source}")
            return
        
        self._recorded_frames.clear()
        self._recorded_poses.clear()
        self._frame_count = 0
        
        self._stack.set_visible_child_name("processing")
        
        if self._processing_mode == ProcessingMode.PREPROCESS:
            self._start_preprocessing()
        else:
            self._play_btn.set_sensitive(True)
            self._record_btn.set_sensitive(True)
            self._state_label.set_label(f"State: Ready ({source})")
    
    def _start_preprocessing(self) -> None:
        """Start background preprocessing of video."""
        self._state = AppState.PREPROCESSING
        self._preprocess_progress = 0.0
        self._preprocess_total = int(self._video_capture._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self._play_btn.set_sensitive(False)
        self._pause_btn.set_sensitive(False)
        self._stop_btn.set_sensitive(True)
        self._record_btn.set_sensitive(False)
        self._state_label.set_label("State: PREPROCESSING...")
        
        self._preprocess_thread = threading.Thread(target=self._preprocess_video, daemon=True)
        self._preprocess_thread.start()
        self.logger.info(f"Started preprocessing {self._preprocess_total} frames...")
    
    def _preprocess_video(self) -> None:
        """Background thread for preprocessing video."""
        self._video_capture.start()
        frame_idx = 0
        last_hands_data = None
        
        while self._state == AppState.PREPROCESSING:
            frame_result = self._video_capture.read()
            
            if frame_result is None:
                if self._video_capture.state == CaptureState.STOPPED:
                    break
                continue
            
            frame = frame_result.frame
            timestamp = frame_result.timestamp
            
            result = self._pose_processor.process(frame, timestamp)
            pose_2d = result.pose_2d if result else None
            pose_3d = result.pose_3d if result else None
            
            hands_data = None
            if self._enable_hands and self._hand_tracker:
                if frame_idx % 2 == 0:
                    hands_data = self._hand_tracker.process(frame, timestamp)
                    last_hands_data = hands_data
                else:
                    hands_data = last_hands_data
            
            skeleton_pose = None
            if pose_3d and pose_3d.is_valid:
                self._current_pose_3d = pose_3d
                
                if self._enable_floor:
                    self._floor_detector.update(pose_3d)
                
                floor = self._floor_detector.current_plane
                root_transform = self._root_motion.process(pose_3d, floor)
                skeleton_pose = self._skeleton_solver.solve(pose_3d, root_transform)
                
                if self._enable_ik:
                    skeleton_pose = self._ik_solver.process(
                        skeleton_pose, pose_3d,
                        root_transform.left_foot_grounded,
                        root_transform.right_foot_grounded
                    )
            
            display_frame = frame.copy()
            if self._show_skeleton and pose_2d:
                display_frame = self._pose_processor.draw_pose(display_frame, pose_2d)
            
            if self._show_hands and hands_data and self._hand_tracker:
                display_frame = self._hand_tracker.draw_hands(display_frame, hands_data)
            
            recorded = RecordedFrame(
                frame_number=frame_idx,
                timestamp=timestamp,
                frame=display_frame.copy(),
                pose_2d=pose_2d,
                pose_3d=pose_3d,
                hands_data=hands_data,
                skeleton_pose=skeleton_pose
            )
            self._recorded_frames.append(recorded)
            if skeleton_pose:
                self._recorded_poses.append(skeleton_pose)
            
            frame_idx += 1
            self._preprocess_progress = frame_idx / max(1, self._preprocess_total)
            
            GLib.idle_add(self._update_preprocess_progress)
        
        self.logger.info(f"Preprocessing complete: {len(self._recorded_frames)} frames")
        GLib.idle_add(self._on_preprocessing_complete)
    
    def _update_preprocess_progress(self) -> bool:
        """Update preprocessing progress bar (called from main thread)."""
        self._progress_bar.set_fraction(self._preprocess_progress)
        percent = int(self._preprocess_progress * 100)
        self._progress_bar.set_text(f"Processing: {percent}% ({len(self._recorded_frames)}/{self._preprocess_total})")
        self._state_label.set_label(f"State: PREPROCESSING {percent}%")
        return False
    
    def _on_preprocessing_complete(self) -> bool:
        """Handle preprocessing completion (called from main thread)."""
        self._state = AppState.PLAYBACK
        self._playback_index = 0
        self._playback_playing = True
        
        self._play_btn.set_sensitive(True)
        self._pause_btn.set_sensitive(True)
        self._stop_btn.set_sensitive(True)
        self._record_btn.set_sensitive(False)
        self._export_btn.set_sensitive(True)
        self._state_label.set_label(f"State: PLAYBACK ({len(self._recorded_frames)} frames)")
        
        self._prev_btn.set_sensitive(True)
        self._next_btn.set_sensitive(True)
        self._scrub_scale.set_sensitive(True)
        self._scrub_scale.set_range(0, max(1, len(self._recorded_frames) - 1))
        self._scrub_scale.set_value(0)
        
        self._start_playback_loop()
        return False
    
    def _start_playback_loop(self) -> None:
        """Start the playback timer."""
        GLib.timeout_add(33, self._playback_tick)
    
    def _playback_tick(self) -> bool:
        """Playback timer tick."""
        if self._state != AppState.PLAYBACK:
            return False
        
        if not self._recorded_frames:
            return False
        
        if self._playback_playing:
            self._playback_index = (self._playback_index + 1) % len(self._recorded_frames)
        
        recorded = self._recorded_frames[self._playback_index]
        self._current_pose_3d = recorded.pose_3d
        
        self._update_playback_display(recorded.frame)
        
        return self._state == AppState.PLAYBACK
    
    def _update_playback_display(self, frame: np.ndarray) -> None:
        """Update display during playback."""
        h, w = frame.shape[:2]
        display_w, display_h = 960, 540
        scale = min(display_w / w, display_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            frame_resized.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False, 8,
            new_w, new_h,
            new_w * 3
        )
        texture = Gdk.Texture.new_for_pixbuf(pixbuf)
        self._video_picture.set_paintable(texture)
        
        progress = self._playback_index / max(1, len(self._recorded_frames) - 1)
        self._progress_bar.set_fraction(progress)
        self._progress_bar.set_text(f"Frame {self._playback_index + 1}/{len(self._recorded_frames)}")
        
        status = "Playing" if self._playback_playing else "Paused"
        self._state_label.set_label(f"State: PLAYBACK - {status}")
        self._frames_label.set_label(f"Frame: {self._playback_index + 1}/{len(self._recorded_frames)}")
        
        if self._show_3d_view and self._skeleton_viewer and self._current_pose_3d:
            skeleton_3d = self._skeleton_viewer.render(self._current_pose_3d)
            
            pixbuf_3d = GdkPixbuf.Pixbuf.new_from_data(
                skeleton_3d.tobytes(),
                GdkPixbuf.Colorspace.RGB,
                False, 8,
                skeleton_3d.shape[1], skeleton_3d.shape[0],
                skeleton_3d.shape[1] * 3
            )
            texture_3d = Gdk.Texture.new_for_pixbuf(pixbuf_3d)
            self._3d_picture.set_paintable(texture_3d)
    
    def _on_play_clicked(self, button: Gtk.Button) -> None:
        """Handle play button click."""
        if self._state == AppState.PLAYBACK:
            self._playback_playing = True
            self._state_label.set_label("State: PLAYBACK - Playing")
            return
        
        if self._video_capture is None:
            return
        
        self._stop_processing.clear()
        self._video_capture.start()
        self._state = AppState.LIVE
        
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        self._play_btn.set_sensitive(False)
        self._pause_btn.set_sensitive(True)
        self._stop_btn.set_sensitive(True)
        self._state_label.set_label("State: LIVE")
    
    def _on_pause_clicked(self, button: Gtk.Button) -> None:
        """Handle pause button click."""
        if self._state == AppState.PLAYBACK:
            self._playback_playing = not self._playback_playing
            status = "Playing" if self._playback_playing else "Paused"
            self._state_label.set_label(f"State: PLAYBACK - {status}")
            return
        
        if self._video_capture:
            if self._state == AppState.PAUSED:
                self._video_capture.resume()
                self._state = AppState.LIVE
                self._state_label.set_label("State: LIVE")
            else:
                self._video_capture.pause()
                self._state = AppState.PAUSED
                self._state_label.set_label("State: PAUSED")
    
    def _on_stop_clicked(self, button: Gtk.Button) -> None:
        """Handle stop button click."""
        self._stop_processing.set()
        
        if self._state == AppState.PREPROCESSING:
            self._state = AppState.HOME
        elif self._state == AppState.PLAYBACK:
            self._state = AppState.HOME
        
        if self._video_capture:
            self._video_capture.stop()
        
        self._play_btn.set_sensitive(True)
        self._pause_btn.set_sensitive(False)
        self._stop_btn.set_sensitive(False)
        self._prev_btn.set_sensitive(False)
        self._next_btn.set_sensitive(False)
        self._scrub_scale.set_sensitive(False)
        self._state_label.set_label("State: Stopped")
    
    def _on_prev_frame(self, button: Gtk.Button) -> None:
        """Handle previous frame button click."""
        if self._state == AppState.PLAYBACK and self._recorded_frames:
            self._playback_playing = False
            self._playback_index = max(0, self._playback_index - 1)
            self._scrub_scale.set_value(self._playback_index)
    
    def _on_next_frame(self, button: Gtk.Button) -> None:
        """Handle next frame button click."""
        if self._state == AppState.PLAYBACK and self._recorded_frames:
            self._playback_playing = False
            self._playback_index = min(len(self._recorded_frames) - 1, self._playback_index + 1)
            self._scrub_scale.set_value(self._playback_index)
    
    def _on_scrub_changed(self, scale: Gtk.Scale) -> None:
        """Handle scrub slider change."""
        if self._state == AppState.PLAYBACK and self._recorded_frames:
            self._playback_playing = False
            self._playback_index = int(scale.get_value())
    
    def _on_record_clicked(self, button: Gtk.Button) -> None:
        """Handle record button click."""
        if self._is_recording:
            self._is_recording = False
            self._record_btn.remove_css_class("destructive-action")
            self._frames_label.set_label(f"Recorded: {len(self._recorded_poses)} frames")
            if self._recorded_poses:
                self._export_btn.set_sensitive(True)
        else:
            self._is_recording = True
            self._recorded_poses.clear()
            self._record_btn.add_css_class("destructive-action")
            self._export_btn.set_sensitive(False)
    
    def _on_export_clicked(self, button: Gtk.Button) -> None:
        """Handle export button click."""
        if not self._recorded_poses:
            return
        
        name = self._export_entry.get_text() or "animation"
        
        try:
            path = self._fbx_exporter.export(self._recorded_poses, name)
            self.logger.info(f"Exported to: {path}")
            
            dialog = Gtk.AlertDialog()
            dialog.set_message("Export Complete")
            dialog.set_detail(f"Animation exported to:\n{path}")
            dialog.set_buttons(["OK"])
            dialog.show(self)
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
    
    def _process_loop(self) -> None:
        """Main processing loop (runs in separate thread)."""
        frame_count = 0
        last_hands_data = None
        
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
            
            result = self._pose_processor.process(frame, timestamp)
            pose_2d = result.pose_2d if result else None
            pose_3d = result.pose_3d if result else None
            
            hands_data = None
            if self._enable_hands and self._hand_tracker:
                if frame_count % 2 == 0:
                    hands_data = self._hand_tracker.process(frame, timestamp)
                    last_hands_data = hands_data
                else:
                    hands_data = last_hands_data
            
            skeleton_pose = None
            if pose_3d and pose_3d.is_valid:
                self._current_pose_3d = pose_3d
                
                if self._enable_floor:
                    self._floor_detector.update(pose_3d)
                
                floor = self._floor_detector.current_plane
                root_transform = self._root_motion.process(pose_3d, floor)
                skeleton_pose = self._skeleton_solver.solve(pose_3d, root_transform)
                
                if self._enable_ik:
                    skeleton_pose = self._ik_solver.process(
                        skeleton_pose, pose_3d,
                        root_transform.left_foot_grounded,
                        root_transform.right_foot_grounded
                    )
                
                if self._is_recording and skeleton_pose:
                    self._recorded_poses.append(skeleton_pose)
            
            display_frame = frame.copy()
            if self._show_skeleton and pose_2d:
                display_frame = self._pose_processor.draw_pose(display_frame, pose_2d)
            
            if self._show_hands and hands_data and self._hand_tracker:
                display_frame = self._hand_tracker.draw_hands(display_frame, hands_data)
            
            self._frame_timer.stop()
            frame_count += 1
            
            GLib.idle_add(self._update_display, display_frame, pose_2d)
        
        self.logger.info("Processing loop ended")
    
    def _update_display(self, frame: np.ndarray, pose_2d) -> bool:
        """Update display (called from main thread)."""
        h, w = frame.shape[:2]
        display_w, display_h = 960, 540
        scale = min(display_w / w, display_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) if frame_resized.shape[2] == 3 else frame_resized
        
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            frame_rgb.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False, 8,
            new_w, new_h,
            new_w * 3
        )
        texture = Gdk.Texture.new_for_pixbuf(pixbuf)
        self._video_picture.set_paintable(texture)
        
        self._fps_label.set_label(f"FPS: {self._frame_timer.fps:.1f}")
        
        if pose_2d:
            self._pose_label.set_label(f"Pose: {pose_2d.num_visible_joints}/33 joints")
        else:
            self._pose_label.set_label("Pose: Not detected")
        
        if self._is_recording:
            self._frames_label.set_label(f"Recording: {len(self._recorded_poses)} frames")
        
        if self._show_3d_view and self._skeleton_viewer and self._current_pose_3d:
            skeleton_3d = self._skeleton_viewer.render(self._current_pose_3d)
            skeleton_rgb = cv2.cvtColor(skeleton_3d, cv2.COLOR_BGR2RGB) if skeleton_3d.shape[2] == 3 else skeleton_3d
            
            pixbuf_3d = GdkPixbuf.Pixbuf.new_from_data(
                skeleton_rgb.tobytes(),
                GdkPixbuf.Colorspace.RGB,
                False, 8,
                skeleton_3d.shape[1], skeleton_3d.shape[0],
                skeleton_3d.shape[1] * 3
            )
            texture_3d = Gdk.Texture.new_for_pixbuf(pixbuf_3d)
            self._3d_picture.set_paintable(texture_3d)
        
        if self._video_capture and not self._video_capture.is_webcam:
            progress = self._video_capture.progress
            current = self._video_capture.current_time
            total = self._video_capture.duration
            self._progress_bar.set_fraction(progress)
            self._progress_bar.set_text(f"{current:.1f}s / {total:.1f}s")
        else:
            self._progress_bar.set_text("Live")
        
        return False
    
    def do_close_request(self) -> bool:
        """Handle window close."""
        self._stop_processing.set()
        
        if self._process_thread:
            self._process_thread.join(timeout=1.0)
        
        if self._video_capture:
            self._video_capture.close()
        
        if self._pose_processor:
            self._pose_processor.close()
        
        if self._hand_tracker:
            self._hand_tracker.close()
        
        self.logger.info("Window closed, resources cleaned up")
        return False


def run_gtk4_app(config: Config) -> int:
    """Run the GTK4 application."""
    app = MoCapApplication(config)
    return app.run(None)
