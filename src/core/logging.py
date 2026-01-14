"""Logging system with colored output and file logging"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored log output for terminal."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"\033[34m{record.name}{self.RESET}"  # Blue for module name
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup application-wide logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name
        log_dir: Directory for log files
    
    Returns:
        Root logger instance
    """
    root_logger = logging.getLogger("mocap")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    if root_logger.handlers:
        return root_logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColoredFormatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"{log_file}_{timestamp}.log"
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {file_path}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger under the mocap namespace.
    
    Args:
        name: Module name (e.g., "video", "pose", "skeleton")
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"mocap.{name}")
