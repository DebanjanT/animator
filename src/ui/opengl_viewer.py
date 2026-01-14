"""Python wrapper for C++ OpenGL viewer integration"""

import sys
import subprocess
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

from src.core import get_logger
from src.motion.skeleton_solver import SkeletonPose

# Path to the standalone viewer executable
VIEWER_EXECUTABLE = Path(__file__).parent.parent.parent / "viewer" / "build" / "mocap_viewer"


class OpenGLViewer:
    """High-performance OpenGL viewer for 3D model animation preview.
    
    Uses subprocess to launch the C++ viewer for stability.
    """
    
    def __init__(self, width: int = 1280, height: int = 720, title: str = "MoCap 3D Preview"):
        self.logger = get_logger("ui.opengl")
        self.width = width
        self.height = height
        self.title = title
        self._process: Optional[subprocess.Popen] = None
        self._model_path: Optional[str] = None
    
    @property
    def is_available(self) -> bool:
        return VIEWER_EXECUTABLE.exists()
    
    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None
    
    def launch(self, model_path: Optional[str] = None) -> bool:
        """Launch the OpenGL viewer as a separate process."""
        if not self.is_available:
            self.logger.error(f"Viewer executable not found: {VIEWER_EXECUTABLE}")
            return False
        
        if self.is_running:
            self.logger.warning("Viewer already running")
            return True
        
        try:
            cmd = [str(VIEWER_EXECUTABLE)]
            if model_path:
                cmd.append(str(model_path))
                self._model_path = model_path
            
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.logger.info(f"Launched OpenGL viewer (PID: {self._process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to launch viewer: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the viewer process."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self.logger.info("OpenGL viewer stopped")


def launch_viewer(model_path: Optional[str] = None) -> OpenGLViewer:
    """Launch the OpenGL viewer with optional model."""
    viewer = OpenGLViewer()
    
    if not viewer.is_available:
        print(f"Viewer not available. Build it first:")
        print(f"  cd viewer && mkdir build && cd build && cmake .. && make")
        return None
    
    if viewer.launch(model_path):
        print("OpenGL viewer launched!")
        print("Controls:")
        print("  WASD - Move camera")
        print("  QE - Move up/down")
        print("  Right Mouse - Look around")
        print("  Middle Mouse - Orbit")
        print("  Scroll - Zoom")
        print("  ESC - Quit")
        return viewer
    
    return None


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = launch_viewer(model)
    if viewer:
        try:
            while viewer.is_running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            viewer.stop()
