"""Video capture module - webcam and file input with timestamps"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple
import threading
from queue import Queue, Empty

import cv2
import numpy as np

from src.core import get_logger, Config


class CaptureState(Enum):
    """Video capture state."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    STEPPING = auto()


@dataclass
class FrameResult:
    """Result of a frame capture operation."""
    frame: np.ndarray  # RGB frame
    frame_number: int
    timestamp: float  # Seconds from start
    capture_time: float  # System time
    width: int
    height: int
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.frame.shape
    
    @property
    def is_valid(self) -> bool:
        return self.frame is not None and self.frame.size > 0


class VideoCapture:
    """
    Video capture from webcam or file with playback controls.
    
    Features:
    - Webcam and video file support
    - Frame timestamps
    - Pause, resume, step frame-by-frame
    - Thread-safe frame buffer
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("video.capture")
        self.config = config or Config()
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._state = CaptureState.STOPPED
        self._frame_count = 0
        self._start_time = 0.0
        self._pause_time = 0.0
        self._total_pause_duration = 0.0
        
        self._source: Optional[str] = None
        self._is_webcam = False
        self._fps = 30.0
        self._width = 0
        self._height = 0
        self._total_frames = 0
        
        self._frame_buffer: Queue[FrameResult] = Queue(maxsize=5)
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._lock = threading.Lock()
    
    def open(self, source: Optional[str] = None) -> bool:
        """
        Open video source.
        
        Args:
            source: "webcam", camera index, or path to video file.
                   If None, uses config.
        
        Returns:
            True if opened successfully
        """
        if source is None:
            source = self.config.get("video.source", "webcam")
        
        self.close()
        
        if source == "webcam" or (isinstance(source, int)):
            camera_id = source if isinstance(source, int) else self.config.get("video.camera_id", 0)
            self._cap = cv2.VideoCapture(camera_id)
            self._is_webcam = True
            self._source = f"webcam:{camera_id}"
            
            width = self.config.get("video.width", 1280)
            height = self.config.get("video.height", 720)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
        else:
            path = Path(source)
            if not path.exists():
                self.logger.error(f"Video file not found: {source}")
                return False
            
            self._cap = cv2.VideoCapture(str(path))
            self._is_webcam = False
            self._source = str(path)
        
        if not self._cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            return False
        
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self._is_webcam else 0
        
        self.logger.info(f"Opened: {self._source}")
        self.logger.info(f"Resolution: {self._width}x{self._height} @ {self._fps:.1f} FPS")
        if self._total_frames > 0:
            duration = self._total_frames / self._fps
            self.logger.info(f"Duration: {duration:.1f}s ({self._total_frames} frames)")
        
        self._state = CaptureState.STOPPED
        self._frame_count = 0
        
        return True
    
    def close(self) -> None:
        """Close video source and cleanup."""
        self.stop()
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        self._state = CaptureState.STOPPED
        self._source = None
        
        while not self._frame_buffer.empty():
            try:
                self._frame_buffer.get_nowait()
            except Empty:
                break
    
    def start(self, threaded: bool = False) -> None:
        """
        Start video capture.
        
        Args:
            threaded: If True, capture runs in background thread
        """
        if self._cap is None or not self._cap.isOpened():
            self.logger.error("No video source opened")
            return
        
        self._start_time = time.perf_counter()
        self._total_pause_duration = 0.0
        self._frame_count = 0
        self._state = CaptureState.PLAYING
        
        if threaded:
            self._stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            self.logger.info("Started threaded capture")
        else:
            self.logger.info("Started synchronous capture")
    
    def stop(self) -> None:
        """Stop video capture."""
        self._stop_event.set()
        
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        
        self._state = CaptureState.STOPPED
    
    def pause(self) -> None:
        """Pause video capture."""
        if self._state == CaptureState.PLAYING:
            self._pause_time = time.perf_counter()
            self._state = CaptureState.PAUSED
            self.logger.debug("Paused")
    
    def resume(self) -> None:
        """Resume video capture."""
        if self._state == CaptureState.PAUSED:
            self._total_pause_duration += time.perf_counter() - self._pause_time
            self._state = CaptureState.PLAYING
            self.logger.debug("Resumed")
    
    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns True if now paused."""
        if self._state == CaptureState.PLAYING:
            self.pause()
            return True
        elif self._state == CaptureState.PAUSED:
            self.resume()
            return False
        return False
    
    def step(self) -> Optional[FrameResult]:
        """
        Step one frame forward (when paused).
        
        Returns:
            FrameResult or None if not paused/no frame
        """
        if self._state != CaptureState.PAUSED:
            return None
        
        self._state = CaptureState.STEPPING
        result = self.read()
        self._state = CaptureState.PAUSED
        
        return result
    
    def read(self) -> Optional[FrameResult]:
        """
        Read next frame synchronously.
        
        Returns:
            FrameResult or None if no frame available
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        
        if self._state == CaptureState.PAUSED:
            return None
        
        with self._lock:
            ret, frame = self._cap.read()
        
        if not ret or frame is None:
            if not self._is_webcam:
                self.logger.info("End of video file")
                self._state = CaptureState.STOPPED
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        capture_time = time.perf_counter()
        timestamp = capture_time - self._start_time - self._total_pause_duration
        
        result = FrameResult(
            frame=frame_rgb,
            frame_number=self._frame_count,
            timestamp=timestamp,
            capture_time=capture_time,
            width=self._width,
            height=self._height
        )
        
        self._frame_count += 1
        
        return result
    
    def read_buffered(self, timeout: float = 0.1) -> Optional[FrameResult]:
        """
        Read frame from buffer (when using threaded capture).
        
        Args:
            timeout: Max time to wait for frame
        
        Returns:
            FrameResult or None
        """
        try:
            return self._frame_buffer.get(timeout=timeout)
        except Empty:
            return None
    
    def _capture_loop(self) -> None:
        """Background capture thread."""
        while not self._stop_event.is_set():
            if self._state != CaptureState.PLAYING:
                time.sleep(0.01)
                continue
            
            result = self.read()
            if result is None:
                if self._state == CaptureState.STOPPED:
                    break
                continue
            
            try:
                self._frame_buffer.put(result, timeout=0.1)
            except:
                pass
            
            frame_duration = 1.0 / self._fps
            time.sleep(frame_duration * 0.5)
    
    def seek(self, frame_number: int) -> bool:
        """
        Seek to specific frame (video files only).
        
        Args:
            frame_number: Target frame number
        
        Returns:
            True if seek successful
        """
        if self._is_webcam or self._cap is None:
            return False
        
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._frame_count = frame_number
        
        self.logger.debug(f"Seeked to frame {frame_number}")
        return True
    
    def seek_time(self, seconds: float) -> bool:
        """
        Seek to specific time (video files only).
        
        Args:
            seconds: Target time in seconds
        
        Returns:
            True if seek successful
        """
        if self._is_webcam:
            return False
        
        frame_number = int(seconds * self._fps)
        return self.seek(frame_number)
    
    @property
    def state(self) -> CaptureState:
        return self._state
    
    @property
    def is_playing(self) -> bool:
        return self._state == CaptureState.PLAYING
    
    @property
    def is_paused(self) -> bool:
        return self._state == CaptureState.PAUSED
    
    @property
    def is_webcam(self) -> bool:
        return self._is_webcam
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def total_frames(self) -> int:
        return self._total_frames
    
    @property
    def duration(self) -> float:
        """Video duration in seconds (0 for webcam)."""
        if self._is_webcam or self._total_frames == 0:
            return 0.0
        return self._total_frames / self._fps
    
    @property
    def current_time(self) -> float:
        """Current playback time in seconds."""
        return self._frame_count / self._fps
    
    @property
    def progress(self) -> float:
        """Playback progress 0.0-1.0 (0 for webcam)."""
        if self._total_frames == 0:
            return 0.0
        return self._frame_count / self._total_frames
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
