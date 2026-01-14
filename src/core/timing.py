"""Frame timing and synchronization system"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class FrameData:
    """Container for frame timing information."""
    frame_number: int
    timestamp: float  # Seconds since start
    capture_time: float  # System time when captured
    processing_time: float = 0.0  # Time spent processing this frame
    
    @property
    def age(self) -> float:
        """Time since frame was captured."""
        return time.perf_counter() - self.capture_time


class FrameTimer:
    """Measures and tracks frame processing times."""
    
    def __init__(self, window_size: int = 60):
        self._window_size = window_size
        self._frame_times: Deque[float] = deque(maxlen=window_size)
        self._start_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None
    
    def start(self) -> None:
        """Start timing a frame."""
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time."""
        if self._start_time is None:
            return 0.0
        
        elapsed = time.perf_counter() - self._start_time
        self._frame_times.append(elapsed)
        self._last_frame_time = elapsed
        self._start_time = None
        return elapsed
    
    @property
    def last_frame_time(self) -> float:
        """Last frame processing time in seconds."""
        return self._last_frame_time or 0.0
    
    @property
    def average_frame_time(self) -> float:
        """Average frame processing time over window."""
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)
    
    @property
    def fps(self) -> float:
        """Current frames per second based on processing time."""
        avg = self.average_frame_time
        return 1.0 / avg if avg > 0 else 0.0
    
    @property
    def min_frame_time(self) -> float:
        """Minimum frame time in window."""
        return min(self._frame_times) if self._frame_times else 0.0
    
    @property
    def max_frame_time(self) -> float:
        """Maximum frame time in window."""
        return max(self._frame_times) if self._frame_times else 0.0
    
    def reset(self) -> None:
        """Reset all timing data."""
        self._frame_times.clear()
        self._start_time = None
        self._last_frame_time = None


@dataclass
class FrameClock:
    """
    Manages frame timing and synchronization for the pipeline.
    
    Provides consistent timestamps across all pipeline stages.
    """
    target_fps: float = 30.0
    _frame_count: int = field(default=0, init=False)
    _start_time: float = field(default=0.0, init=False)
    _last_frame_time: float = field(default=0.0, init=False)
    _paused: bool = field(default=False, init=False)
    _pause_time: float = field(default=0.0, init=False)
    _total_pause_duration: float = field(default=0.0, init=False)
    
    def start(self) -> None:
        """Start the frame clock."""
        self._start_time = time.perf_counter()
        self._last_frame_time = self._start_time
        self._frame_count = 0
        self._paused = False
        self._total_pause_duration = 0.0
    
    def tick(self) -> FrameData:
        """
        Advance to next frame and return frame data.
        
        Returns:
            FrameData with timing information
        """
        if self._paused:
            raise RuntimeError("Cannot tick while paused")
        
        current_time = time.perf_counter()
        timestamp = current_time - self._start_time - self._total_pause_duration
        
        frame_data = FrameData(
            frame_number=self._frame_count,
            timestamp=timestamp,
            capture_time=current_time
        )
        
        self._last_frame_time = current_time
        self._frame_count += 1
        
        return frame_data
    
    def pause(self) -> None:
        """Pause the frame clock."""
        if not self._paused:
            self._paused = True
            self._pause_time = time.perf_counter()
    
    def resume(self) -> None:
        """Resume the frame clock."""
        if self._paused:
            self._total_pause_duration += time.perf_counter() - self._pause_time
            self._paused = False
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def elapsed_time(self) -> float:
        """Total elapsed time excluding pauses."""
        if self._start_time == 0:
            return 0.0
        current = self._pause_time if self._paused else time.perf_counter()
        return current - self._start_time - self._total_pause_duration
    
    @property
    def target_frame_duration(self) -> float:
        """Target duration per frame in seconds."""
        return 1.0 / self.target_fps
    
    def wait_for_next_frame(self) -> float:
        """
        Wait until it's time for the next frame.
        
        Returns:
            Actual time waited in seconds
        """
        if self._paused:
            return 0.0
        
        elapsed_since_last = time.perf_counter() - self._last_frame_time
        wait_time = self.target_frame_duration - elapsed_since_last
        
        if wait_time > 0:
            time.sleep(wait_time)
            return wait_time
        
        return 0.0
    
    def reset(self) -> None:
        """Reset the frame clock."""
        self._frame_count = 0
        self._start_time = 0.0
        self._last_frame_time = 0.0
        self._paused = False
        self._pause_time = 0.0
        self._total_pause_duration = 0.0
