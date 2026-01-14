"""
Pose smoothing using One Euro Filter.

The One Euro Filter is designed to reduce jitter while maintaining low latency.
It adapts the cutoff frequency based on the speed of movement:
- When moving slowly: more smoothing (reduces jitter)
- When moving fast: less smoothing (reduces lag)

Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from src.core import get_logger


@dataclass
class OneEuroFilterParams:
    """Parameters for One Euro Filter."""
    min_cutoff: float = 1.0      # Minimum cutoff frequency (Hz) - lower = more smoothing
    beta: float = 0.007          # Speed coefficient - higher = less lag when moving
    d_cutoff: float = 1.0        # Derivative cutoff frequency
    
    # Thresholds for dead zone (position locked if movement below this)
    position_threshold: float = 0.002  # Normalized coordinates (0-1)
    velocity_threshold: float = 0.005  # Movement per frame threshold


class LowPassFilter:
    """Simple low-pass filter."""
    
    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha
        self._y: Optional[float] = None
        self._initialized = False
    
    def reset(self):
        self._y = None
        self._initialized = False
    
    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self._alpha = alpha
        
        if not self._initialized:
            self._y = value
            self._initialized = True
        else:
            self._y = self._alpha * value + (1.0 - self._alpha) * self._y
        
        return self._y
    
    @property
    def last_value(self) -> Optional[float]:
        return self._y


class OneEuroFilter:
    """
    One Euro Filter for a single value.
    
    Reduces jitter when stationary while maintaining responsiveness during movement.
    """
    
    def __init__(self, params: OneEuroFilterParams = None):
        self.params = params or OneEuroFilterParams()
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: Optional[float] = None
        self._last_value: Optional[float] = None
    
    def reset(self):
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None
        self._last_value = None
    
    def _compute_alpha(self, cutoff: float, dt: float) -> float:
        """Compute smoothing factor alpha from cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def filter(self, value: float, timestamp: float) -> float:
        """
        Filter a value.
        
        Args:
            value: Current raw value
            timestamp: Current timestamp in seconds
        
        Returns:
            Smoothed value
        """
        if self._last_time is None:
            # First sample
            self._last_time = timestamp
            self._last_value = value
            self._x_filter.filter(value)
            self._dx_filter.filter(0.0)
            return value
        
        # Compute time delta
        dt = timestamp - self._last_time
        if dt <= 0:
            dt = 1.0 / 30.0  # Default to 30 FPS
        
        self._last_time = timestamp
        
        # Estimate derivative (velocity)
        dx = (value - self._last_value) / dt if self._last_value is not None else 0.0
        
        # Filter the derivative
        edx = self._dx_filter.filter(dx, self._compute_alpha(self.params.d_cutoff, dt))
        
        # Compute adaptive cutoff based on speed
        cutoff = self.params.min_cutoff + self.params.beta * abs(edx)
        
        # Filter the value
        result = self._x_filter.filter(value, self._compute_alpha(cutoff, dt))
        
        self._last_value = value
        
        return result


class KeypointSmoother:
    """
    Smooths a single keypoint (x, y, z) using One Euro Filters.
    Also implements dead zone to lock position when nearly stationary.
    """
    
    def __init__(self, params: OneEuroFilterParams = None):
        self.params = params or OneEuroFilterParams()
        self._x_filter = OneEuroFilter(params)
        self._y_filter = OneEuroFilter(params)
        self._z_filter = OneEuroFilter(params)
        self._last_pos: Optional[Tuple[float, float, float]] = None
        self._locked_pos: Optional[Tuple[float, float, float]] = None
        self._frames_stationary = 0
    
    def reset(self):
        self._x_filter.reset()
        self._y_filter.reset()
        self._z_filter.reset()
        self._last_pos = None
        self._locked_pos = None
        self._frames_stationary = 0
    
    def smooth(
        self, 
        x: float, 
        y: float, 
        z: float, 
        timestamp: float
    ) -> Tuple[float, float, float]:
        """
        Smooth a keypoint position.
        
        Args:
            x, y, z: Raw keypoint coordinates (normalized 0-1)
            timestamp: Current timestamp
        
        Returns:
            Smoothed (x, y, z) coordinates
        """
        # Apply One Euro Filter
        sx = self._x_filter.filter(x, timestamp)
        sy = self._y_filter.filter(y, timestamp)
        sz = self._z_filter.filter(z, timestamp)
        
        # Dead zone check - lock position if movement is tiny
        if self._last_pos is not None:
            dx = abs(sx - self._last_pos[0])
            dy = abs(sy - self._last_pos[1])
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < self.params.position_threshold:
                self._frames_stationary += 1
                
                # After being stationary for a few frames, lock position
                if self._frames_stationary > 3:
                    if self._locked_pos is None:
                        self._locked_pos = (sx, sy, sz)
                    return self._locked_pos
            else:
                # Movement detected, unlock
                self._frames_stationary = 0
                self._locked_pos = None
        
        self._last_pos = (sx, sy, sz)
        return (sx, sy, sz)


class PoseSmoother:
    """
    Smooths all keypoints in a pose using individual One Euro Filters.
    """
    
    def __init__(self, params: OneEuroFilterParams = None):
        self.logger = get_logger("pose.smoother")
        self.params = params or OneEuroFilterParams()
        self._keypoint_smoothers: Dict[int, KeypointSmoother] = {}
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        if not value:
            self.reset()
    
    def reset(self):
        """Reset all filters."""
        for smoother in self._keypoint_smoothers.values():
            smoother.reset()
        self._keypoint_smoothers.clear()
    
    def _get_smoother(self, keypoint_idx: int) -> KeypointSmoother:
        """Get or create a smoother for a keypoint."""
        if keypoint_idx not in self._keypoint_smoothers:
            self._keypoint_smoothers[keypoint_idx] = KeypointSmoother(self.params)
        return self._keypoint_smoothers[keypoint_idx]
    
    def smooth_keypoint(
        self, 
        keypoint_idx: int,
        x: float, 
        y: float, 
        z: float, 
        timestamp: float
    ) -> Tuple[float, float, float]:
        """
        Smooth a single keypoint.
        
        Args:
            keypoint_idx: Index of the keypoint
            x, y, z: Raw coordinates
            timestamp: Current timestamp
        
        Returns:
            Smoothed (x, y, z)
        """
        if not self._enabled:
            return (x, y, z)
        
        smoother = self._get_smoother(keypoint_idx)
        return smoother.smooth(x, y, z, timestamp)
    
    def smooth_pose(self, keypoints: Dict, timestamp: float) -> Dict:
        """
        Smooth all keypoints in a pose.
        
        Args:
            keypoints: Dict of keypoint index -> HalpeKeypoint
            timestamp: Current timestamp
        
        Returns:
            Dict with smoothed keypoints
        """
        if not self._enabled:
            return keypoints
        
        smoothed = {}
        for idx, kp in keypoints.items():
            sx, sy, sz = self.smooth_keypoint(
                idx, kp.x, kp.y, kp.z, timestamp
            )
            # Create new keypoint with smoothed values
            from src.pose.halpe_estimator import HalpeKeypoint
            smoothed[idx] = HalpeKeypoint(
                index=kp.index,
                name=kp.name,
                x=sx,
                y=sy,
                z=sz,
                confidence=kp.confidence
            )
        
        return smoothed


# Preset configurations for different use cases
SMOOTH_PRESETS = {
    "default": OneEuroFilterParams(
        min_cutoff=1.0,
        beta=0.007,
        d_cutoff=1.0,
        position_threshold=0.002,
        velocity_threshold=0.005
    ),
    "stable": OneEuroFilterParams(
        min_cutoff=0.5,       # More smoothing
        beta=0.005,           # Less responsive
        d_cutoff=1.0,
        position_threshold=0.003,  # Larger dead zone
        velocity_threshold=0.008
    ),
    "responsive": OneEuroFilterParams(
        min_cutoff=2.0,       # Less smoothing
        beta=0.01,            # More responsive
        d_cutoff=1.0,
        position_threshold=0.001,  # Smaller dead zone
        velocity_threshold=0.003
    ),
    "animation": OneEuroFilterParams(
        min_cutoff=0.8,       # Good balance for character animation
        beta=0.006,
        d_cutoff=1.0,
        position_threshold=0.0025,
        velocity_threshold=0.006
    )
}
