# MoCap to UE5 Animation

A desktop application that converts monocular video into Unreal Engine 5 Mannequin-compatible skeletal animation.

## Features

- **Video Input**: Webcam or video file support
- **2D Pose Estimation**: MediaPipe BlazePose
- **3D Reconstruction**: Single-camera depth inference
- **Floor Detection**: RANSAC-based ground plane detection
- **Root Motion**: Pelvis-based locomotion with foot slide prevention
- **Skeleton Solver**: UE5 Mannequin bone rotation mapping
- **IK System**: Two-bone IK for legs, foot locking
- **FBX Export**: Animation-only export for UE5

## Requirements

- Python 3.10+
- GTK 3.0+ (for UI)
- OpenCV
- MediaPipe

## Installation

```bash
# Install system dependencies (macOS)
brew install gtk+3 pygobject3

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Or with a specific config:

```bash
python main.py --config custom_config.yaml
```

## Project Structure

```
animator/
├── main.py                 # Application entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── core/               # Core systems
│   │   ├── __init__.py
│   │   ├── config.py       # Configuration loader
│   │   ├── logging.py      # Logging system
│   │   └── timing.py       # Frame timing
│   ├── video/              # Video input
│   │   ├── __init__.py
│   │   └── capture.py
│   ├── pose/               # Pose estimation
│   │   ├── __init__.py
│   │   ├── estimator_2d.py
│   │   └── reconstructor_3d.py
│   ├── motion/             # Motion processing
│   │   ├── __init__.py
│   │   ├── floor_detector.py
│   │   ├── root_motion.py
│   │   └── skeleton_solver.py
│   ├── ik/                 # Inverse kinematics
│   │   ├── __init__.py
│   │   └── solver.py
│   ├── export/             # Animation export
│   │   ├── __init__.py
│   │   └── fbx_exporter.py
│   └── ui/                 # GTK UI
│       ├── __init__.py
│       └── main_window.py
└── output/                 # Exported animations
```

## Pipeline

```
Video → 2D Pose → 3D Pose → Floor → Root Motion → Skeleton → IK → FBX → UE5
```

## License

MIT
