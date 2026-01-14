"""Configuration management system"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml


class Config:
    """Centralized configuration manager with dot-notation access."""
    
    _instance: Optional["Config"] = None
    _config: dict = {}
    
    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._initialized and config_path is None:
            return
            
        if config_path is None:
            config_path = self._find_config()
        
        self._load(config_path)
        self._initialized = True
    
    def _find_config(self) -> str:
        """Find config.yaml in project root."""
        current = Path(__file__).parent
        for _ in range(5):
            config_file = current / "config.yaml"
            if config_file.exists():
                return str(config_file)
            current = current.parent
        
        raise FileNotFoundError("config.yaml not found")
    
    def _load(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
        self._config_path = config_path
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load(self._config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Example:
            config.get("video.fps", 30)
            config.get("pose_estimation.model")
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set config value using dot notation (runtime only, not persisted)."""
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self._config_path
        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    @property
    def video(self) -> dict:
        return self._config.get("video", {})
    
    @property
    def pose_estimation(self) -> dict:
        return self._config.get("pose_estimation", {})
    
    @property
    def pose_3d(self) -> dict:
        return self._config.get("pose_3d", {})
    
    @property
    def floor_detection(self) -> dict:
        return self._config.get("floor_detection", {})
    
    @property
    def root_motion(self) -> dict:
        return self._config.get("root_motion", {})
    
    @property
    def skeleton(self) -> dict:
        return self._config.get("skeleton", {})
    
    @property
    def ik(self) -> dict:
        return self._config.get("ik", {})
    
    @property
    def export(self) -> dict:
        return self._config.get("export", {})
    
    @property
    def visualization(self) -> dict:
        return self._config.get("visualization", {})
    
    def __repr__(self) -> str:
        return f"Config({self._config_path})"
