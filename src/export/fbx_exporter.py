"""FBX Animation Exporter for UE5 Mannequin skeleton"""

import json
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core import get_logger, Config
from src.motion.skeleton_solver import SkeletonPose, UE5Bone, quaternion_to_euler


UE5_MANNEQUIN_HIERARCHY = {
    "root": None,
    "pelvis": "root",
    "spine_01": "pelvis",
    "spine_02": "spine_01",
    "spine_03": "spine_02",
    "neck_01": "spine_03",
    "head": "neck_01",
    "clavicle_l": "spine_03",
    "upperarm_l": "clavicle_l",
    "lowerarm_l": "upperarm_l",
    "hand_l": "lowerarm_l",
    "clavicle_r": "spine_03",
    "upperarm_r": "clavicle_r",
    "lowerarm_r": "upperarm_r",
    "hand_r": "lowerarm_r",
    "thigh_l": "pelvis",
    "calf_l": "thigh_l",
    "foot_l": "calf_l",
    "ball_l": "foot_l",
    "thigh_r": "pelvis",
    "calf_r": "thigh_r",
    "foot_r": "calf_r",
    "ball_r": "foot_r",
}

UE5_REST_POSE = {
    "root": {"pos": [0, 0, 0], "rot": [0, 0, 0, 1]},
    "pelvis": {"pos": [0, 0, 98.0], "rot": [0, 0, 0, 1]},
    "spine_01": {"pos": [0, 0, 10.0], "rot": [0, 0, 0, 1]},
    "spine_02": {"pos": [0, 0, 12.0], "rot": [0, 0, 0, 1]},
    "spine_03": {"pos": [0, 0, 12.0], "rot": [0, 0, 0, 1]},
    "neck_01": {"pos": [0, 0, 15.0], "rot": [0, 0, 0, 1]},
    "head": {"pos": [0, 0, 10.0], "rot": [0, 0, 0, 1]},
    "clavicle_l": {"pos": [3.0, 0, 12.0], "rot": [0, 0, 0, 1]},
    "upperarm_l": {"pos": [15.0, 0, 0], "rot": [0, 0, 0, 1]},
    "lowerarm_l": {"pos": [28.0, 0, 0], "rot": [0, 0, 0, 1]},
    "hand_l": {"pos": [25.0, 0, 0], "rot": [0, 0, 0, 1]},
    "clavicle_r": {"pos": [-3.0, 0, 12.0], "rot": [0, 0, 0, 1]},
    "upperarm_r": {"pos": [-15.0, 0, 0], "rot": [0, 0, 0, 1]},
    "lowerarm_r": {"pos": [-28.0, 0, 0], "rot": [0, 0, 0, 1]},
    "hand_r": {"pos": [-25.0, 0, 0], "rot": [0, 0, 0, 1]},
    "thigh_l": {"pos": [10.0, 0, -5.0], "rot": [0, 0, 0, 1]},
    "calf_l": {"pos": [0, 0, -42.0], "rot": [0, 0, 0, 1]},
    "foot_l": {"pos": [0, 0, -40.0], "rot": [0, 0, 0, 1]},
    "ball_l": {"pos": [0, 15.0, -5.0], "rot": [0, 0, 0, 1]},
    "thigh_r": {"pos": [-10.0, 0, -5.0], "rot": [0, 0, 0, 1]},
    "calf_r": {"pos": [0, 0, -42.0], "rot": [0, 0, 0, 1]},
    "foot_r": {"pos": [0, 0, -40.0], "rot": [0, 0, 0, 1]},
    "ball_r": {"pos": [0, 15.0, -5.0], "rot": [0, 0, 0, 1]},
}


@dataclass
class AnimationClip:
    """Container for animation data."""
    name: str
    fps: float
    frame_count: int
    duration: float
    poses: List[SkeletonPose] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return len(self.poses) > 0


@dataclass
class BoneKeyframe:
    """Single keyframe for a bone."""
    time: float
    rotation: np.ndarray  # Quaternion (w, x, y, z)
    position: Optional[np.ndarray] = None  # Only for root


class FBXExporter:
    """
    Export animation to FBX format for UE5 Mannequin.
    
    Uses ASCII FBX format for compatibility.
    Exports animation-only (no mesh).
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.logger = get_logger("export.fbx")
        self.config = config or Config()
        
        export_config = self.config.export
        
        self._output_dir = Path(export_config.get("output_dir", "./output"))
        self._fps = export_config.get("fps", 30)
        self._format = export_config.get("format", "fbx")
        
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized FBX exporter (fps={self._fps})")
    
    def export(
        self,
        poses: List[SkeletonPose],
        filename: str,
        clip_name: Optional[str] = None
    ) -> Path:
        """
        Export animation to FBX file.
        
        Args:
            poses: List of skeleton poses
            filename: Output filename (without extension)
            clip_name: Optional animation clip name
        
        Returns:
            Path to exported file
        """
        if not poses:
            raise ValueError("No poses to export")
        
        clip = AnimationClip(
            name=clip_name or filename,
            fps=self._fps,
            frame_count=len(poses),
            duration=len(poses) / self._fps,
            poses=poses
        )
        
        output_path = self._output_dir / f"{filename}.fbx"
        
        fbx_content = self._generate_fbx_ascii(clip)
        
        with open(output_path, "w") as f:
            f.write(fbx_content)
        
        self.logger.info(f"Exported animation to {output_path}")
        self.logger.info(f"  Frames: {clip.frame_count}, Duration: {clip.duration:.2f}s")
        
        json_path = self._output_dir / f"{filename}.json"
        self._export_json(clip, json_path)
        
        return output_path
    
    def _generate_fbx_ascii(self, clip: AnimationClip) -> str:
        """Generate ASCII FBX file content."""
        lines = []
        
        lines.extend(self._fbx_header())
        lines.extend(self._fbx_definitions())
        lines.extend(self._fbx_objects(clip))
        lines.extend(self._fbx_connections())
        lines.extend(self._fbx_takes(clip))
        
        return "\n".join(lines)
    
    def _fbx_header(self) -> List[str]:
        """Generate FBX header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return [
            "; FBX 7.4.0 project file",
            "; Generated by MoCap to UE5 Animation",
            f"; Timestamp: {timestamp}",
            "",
            "FBXHeaderExtension:  {",
            "    FBXHeaderVersion: 1003",
            "    FBXVersion: 7400",
            "    Creator: \"MoCap to UE5 Animation\"",
            "}",
            "",
            "GlobalSettings:  {",
            "    Version: 1000",
            "    Properties70:  {",
            "        P: \"UpAxis\", \"int\", \"Integer\", \"\", 2",
            "        P: \"UpAxisSign\", \"int\", \"Integer\", \"\", 1",
            "        P: \"FrontAxis\", \"int\", \"Integer\", \"\", 1",
            "        P: \"FrontAxisSign\", \"int\", \"Integer\", \"\", -1",
            "        P: \"CoordAxis\", \"int\", \"Integer\", \"\", 0",
            "        P: \"CoordAxisSign\", \"int\", \"Integer\", \"\", 1",
            "        P: \"OriginalUpAxis\", \"int\", \"Integer\", \"\", 2",
            "        P: \"OriginalUpAxisSign\", \"int\", \"Integer\", \"\", 1",
            f"        P: \"UnitScaleFactor\", \"double\", \"Number\", \"\", 1",
            f"        P: \"CustomFrameRate\", \"double\", \"Number\", \"\", {self._fps}",
            "    }",
            "}",
            "",
        ]
    
    def _fbx_definitions(self) -> List[str]:
        """Generate FBX definitions section."""
        bone_count = len(UE5_MANNEQUIN_HIERARCHY)
        
        return [
            "Definitions:  {",
            "    Version: 100",
            f"    Count: {bone_count + 2}",
            "",
            "    ObjectType: \"Model\" {",
            f"        Count: {bone_count}",
            "    }",
            "",
            "    ObjectType: \"AnimationStack\" {",
            "        Count: 1",
            "    }",
            "",
            "    ObjectType: \"AnimationLayer\" {",
            "        Count: 1",
            "    }",
            "}",
            "",
        ]
    
    def _fbx_objects(self, clip: AnimationClip) -> List[str]:
        """Generate FBX objects section with skeleton and animation."""
        lines = ["Objects:  {", ""]
        
        bone_ids = {}
        base_id = 1000000000
        
        for i, bone_name in enumerate(UE5_MANNEQUIN_HIERARCHY.keys()):
            bone_id = base_id + i
            bone_ids[bone_name] = bone_id
            
            rest = UE5_REST_POSE.get(bone_name, {"pos": [0, 0, 0], "rot": [0, 0, 0, 1]})
            pos = rest["pos"]
            
            lines.extend([
                f"    Model: {bone_id}, \"Model::{bone_name}\", \"LimbNode\" {{",
                "        Version: 232",
                "        Properties70:  {",
                f"            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\", {pos[0]}, {pos[1]}, {pos[2]}",
                "            P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\", 0, 0, 0",
                "            P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\", 1, 1, 1",
                "        }",
                "        Shading: Y",
                "        Culling: \"CullingOff\"",
                "    }",
                "",
            ])
        
        anim_stack_id = base_id + 1000
        anim_layer_id = base_id + 1001
        
        lines.extend([
            f"    AnimationStack: {anim_stack_id}, \"AnimStack::{clip.name}\", \"\" {{",
            "        Properties70:  {",
            f"            P: \"LocalStop\", \"KTime\", \"Time\", \"\", {int(clip.duration * 46186158000)}",
            "        }",
            "    }",
            "",
            f"    AnimationLayer: {anim_layer_id}, \"AnimLayer::BaseLayer\", \"\" {{",
            "    }",
            "",
        ])
        
        curve_id = base_id + 2000
        for bone_name in UE5_MANNEQUIN_HIERARCHY.keys():
            bone_id = bone_ids[bone_name]
            
            keyframes = self._extract_bone_keyframes(clip, bone_name)
            
            for channel, axis in [("R", "X"), ("R", "Y"), ("R", "Z")]:
                curve_id += 1
                
                times = []
                values = []
                
                for kf in keyframes:
                    time_fbx = int(kf.time * 46186158000)
                    times.append(time_fbx)
                    
                    euler = np.degrees(quaternion_to_euler(kf.rotation))
                    if axis == "X":
                        values.append(euler[0])
                    elif axis == "Y":
                        values.append(euler[1])
                    else:
                        values.append(euler[2])
                
                if times:
                    times_str = ", ".join(str(t) for t in times)
                    values_str = ", ".join(f"{v:.6f}" for v in values)
                    
                    lines.extend([
                        f"    AnimationCurve: {curve_id}, \"AnimCurve::{bone_name}_{channel}{axis}\", \"\" {{",
                        "        Default: 0",
                        f"        KeyVer: 4009",
                        f"        KeyTime: *{len(times)} {{",
                        f"            a: {times_str}",
                        "        }",
                        f"        KeyValueFloat: *{len(values)} {{",
                        f"            a: {values_str}",
                        "        }",
                        "    }",
                        "",
                    ])
            
            if bone_name == "root":
                for axis in ["X", "Y", "Z"]:
                    curve_id += 1
                    
                    times = []
                    values = []
                    
                    for kf in keyframes:
                        time_fbx = int(kf.time * 46186158000)
                        times.append(time_fbx)
                        
                        if kf.position is not None:
                            if axis == "X":
                                values.append(kf.position[0] * 100)
                            elif axis == "Y":
                                values.append(kf.position[2] * 100)
                            else:
                                values.append(kf.position[1] * 100)
                        else:
                            values.append(0)
                    
                    if times:
                        times_str = ", ".join(str(t) for t in times)
                        values_str = ", ".join(f"{v:.6f}" for v in values)
                        
                        lines.extend([
                            f"    AnimationCurve: {curve_id}, \"AnimCurve::{bone_name}_T{axis}\", \"\" {{",
                            "        Default: 0",
                            f"        KeyVer: 4009",
                            f"        KeyTime: *{len(times)} {{",
                            f"            a: {times_str}",
                            "        }",
                            f"        KeyValueFloat: *{len(values)} {{",
                            f"            a: {values_str}",
                            "        }",
                            "        }",
                            "",
                        ])
        
        lines.append("}")
        lines.append("")
        
        return lines
    
    def _fbx_connections(self) -> List[str]:
        """Generate FBX connections section."""
        lines = ["Connections:  {", ""]
        
        base_id = 1000000000
        bone_ids = {name: base_id + i for i, name in enumerate(UE5_MANNEQUIN_HIERARCHY.keys())}
        
        for bone_name, parent_name in UE5_MANNEQUIN_HIERARCHY.items():
            bone_id = bone_ids[bone_name]
            
            if parent_name is None:
                lines.append(f"    C: \"OO\", {bone_id}, 0")
            else:
                parent_id = bone_ids[parent_name]
                lines.append(f"    C: \"OO\", {bone_id}, {parent_id}")
        
        anim_stack_id = base_id + 1000
        anim_layer_id = base_id + 1001
        
        lines.append(f"    C: \"OO\", {anim_layer_id}, {anim_stack_id}")
        
        lines.append("}")
        lines.append("")
        
        return lines
    
    def _fbx_takes(self, clip: AnimationClip) -> List[str]:
        """Generate FBX takes section."""
        duration_fbx = int(clip.duration * 46186158000)
        
        return [
            "Takes:  {",
            f"    Current: \"{clip.name}\"",
            f"    Take: \"{clip.name}\" {{",
            f"        FileName: \"{clip.name}.tak\"",
            f"        LocalTime: 0, {duration_fbx}",
            f"        ReferenceTime: 0, {duration_fbx}",
            "    }",
            "}",
            "",
        ]
    
    def _extract_bone_keyframes(
        self,
        clip: AnimationClip,
        bone_name: str
    ) -> List[BoneKeyframe]:
        """Extract keyframes for a specific bone."""
        keyframes = []
        
        try:
            bone_enum = UE5Bone(bone_name)
        except ValueError:
            identity = np.array([1, 0, 0, 0], dtype=np.float32)
            for i, pose in enumerate(clip.poses):
                keyframes.append(BoneKeyframe(
                    time=i / clip.fps,
                    rotation=identity,
                    position=pose.root_position if bone_name == "root" else None
                ))
            return keyframes
        
        for i, pose in enumerate(clip.poses):
            time = i / clip.fps
            
            if bone_name == "root":
                keyframes.append(BoneKeyframe(
                    time=time,
                    rotation=pose.root_rotation,
                    position=pose.root_position
                ))
            else:
                transform = pose.bone_transforms.get(bone_enum)
                if transform:
                    keyframes.append(BoneKeyframe(
                        time=time,
                        rotation=transform.local_rotation
                    ))
                else:
                    keyframes.append(BoneKeyframe(
                        time=time,
                        rotation=np.array([1, 0, 0, 0], dtype=np.float32)
                    ))
        
        return keyframes
    
    def _export_json(self, clip: AnimationClip, path: Path) -> None:
        """Export animation data as JSON for debugging/validation."""
        data = {
            "name": clip.name,
            "fps": clip.fps,
            "frame_count": clip.frame_count,
            "duration": clip.duration,
            "skeleton": list(UE5_MANNEQUIN_HIERARCHY.keys()),
            "frames": []
        }
        
        for pose in clip.poses:
            frame_data = {
                "frame": pose.frame_number,
                "timestamp": pose.timestamp,
                "root_position": pose.root_position.tolist(),
                "root_rotation": pose.root_rotation.tolist(),
                "bones": {}
            }
            
            for bone, transform in pose.bone_transforms.items():
                frame_data["bones"][bone.value] = {
                    "rotation": transform.rotation.tolist(),
                    "local_rotation": transform.local_rotation.tolist()
                }
            
            data["frames"].append(frame_data)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        self.logger.debug(f"Exported JSON to {path}")
    
    def export_for_blender(
        self,
        poses: List[SkeletonPose],
        filename: str
    ) -> Path:
        """
        Export as Python script for Blender import.
        
        This can be run in Blender to create the animation.
        """
        output_path = self._output_dir / f"{filename}_blender.py"
        
        script = self._generate_blender_script(poses, filename)
        
        with open(output_path, "w") as f:
            f.write(script)
        
        self.logger.info(f"Exported Blender script to {output_path}")
        return output_path
    
    def _generate_blender_script(
        self,
        poses: List[SkeletonPose],
        clip_name: str
    ) -> str:
        """Generate Blender Python script for animation import."""
        lines = [
            "# Blender Animation Import Script",
            "# Generated by MoCap to UE5 Animation",
            "# Run this script in Blender with an armature selected",
            "",
            "import bpy",
            "import mathutils",
            "from math import radians",
            "",
            f"CLIP_NAME = \"{clip_name}\"",
            f"FPS = {self._fps}",
            "",
            "# Bone mapping from our skeleton to Blender armature",
            "BONE_MAPPING = {",
        ]
        
        for bone_name in UE5_MANNEQUIN_HIERARCHY.keys():
            lines.append(f"    \"{bone_name}\": \"{bone_name}\",")
        
        lines.extend([
            "}",
            "",
            "# Animation data",
            "ANIMATION_DATA = [",
        ])
        
        for pose in poses:
            frame_data = {
                "frame": pose.frame_number,
                "root_pos": pose.root_position.tolist(),
                "root_rot": pose.root_rotation.tolist(),
                "bones": {
                    bone.value: transform.local_rotation.tolist()
                    for bone, transform in pose.bone_transforms.items()
                }
            }
            lines.append(f"    {frame_data},")
        
        lines.extend([
            "]",
            "",
            "def apply_animation():",
            "    obj = bpy.context.active_object",
            "    if obj is None or obj.type != 'ARMATURE':",
            "        print('Please select an armature')",
            "        return",
            "    ",
            "    bpy.context.scene.render.fps = FPS",
            "    ",
            "    for frame_data in ANIMATION_DATA:",
            "        frame = frame_data['frame']",
            "        bpy.context.scene.frame_set(frame)",
            "        ",
            "        for bone_name, rotation in frame_data['bones'].items():",
            "            mapped_name = BONE_MAPPING.get(bone_name, bone_name)",
            "            if mapped_name in obj.pose.bones:",
            "                bone = obj.pose.bones[mapped_name]",
            "                bone.rotation_quaternion = mathutils.Quaternion(rotation)",
            "                bone.keyframe_insert(data_path='rotation_quaternion', frame=frame)",
            "    ",
            "    print(f'Applied {len(ANIMATION_DATA)} frames')",
            "",
            "if __name__ == '__main__':",
            "    apply_animation()",
        ])
        
        return "\n".join(lines)
    
    def set_output_dir(self, path: str) -> None:
        """Set output directory."""
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def output_dir(self) -> Path:
        return self._output_dir
