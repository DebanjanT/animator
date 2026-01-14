#!/usr/bin/env python3
"""
MoCap to UE5 Animation - Main Entry Point

Converts monocular video into Unreal Engine 5 Mannequin-compatible
skeletal animation exported as FBX.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core import Config, setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert video to UE5 skeletal animation"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input video file (overrides config)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def main() -> int:
    """Main application entry point."""
    args = parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    try:
        config = Config(str(config_path))
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    log_level = "DEBUG" if args.debug else config.get("app.log_level", "INFO")
    setup_logging(level=log_level, log_file="mocap")
    logger = get_logger("main")
    
    logger.info("=" * 50)
    logger.info(f"MoCap to UE5 Animation v{config.get('app.version', '0.1.0')}")
    logger.info("=" * 50)
    
    if args.input:
        config.set("video.source", args.input)
        logger.info(f"Input override: {args.input}")
    
    if args.output:
        config.set("export.output_dir", args.output)
        logger.info(f"Output override: {args.output}")
    
    output_dir = Path(config.get("export.output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.headless:
        logger.info("Running in headless mode")
        return run_headless(config)
    else:
        logger.info("Starting GTK application")
        return run_gui(config)


def run_headless(config: Config) -> int:
    """Run pipeline without GUI."""
    logger = get_logger("main")
    logger.info("Headless mode not yet implemented")
    logger.info("Pipeline stages will be added incrementally")
    return 0


def run_gui(config: Config) -> int:
    """Run with GTK GUI, fallback to OpenCV if GTK unavailable."""
    logger = get_logger("main")
    
    try:
        import gi
        gi.require_version("Gtk", "4.0")
        from gi.repository import Gtk
        
        from src.ui.gtk4_window import run_gtk4_app
        
        logger.info("Starting GTK4 application")
        return run_gtk4_app(config)
        
    except (ImportError, Exception) as e:
        logger.warning(f"GTK4 not available: {e}")
        logger.info("Falling back to OpenCV-based UI")
        return run_opencv_gui(config)


def run_opencv_gui(config: Config) -> int:
    """Run with OpenCV-based GUI (cross-platform fallback)."""
    logger = get_logger("main")
    
    try:
        from src.ui.cv_window import run_cv_app
        
        logger.info("Starting OpenCV-based UI")
        return run_cv_app(config)
        
    except Exception as e:
        logger.error(f"Failed to start OpenCV UI: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
