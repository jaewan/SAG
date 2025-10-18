"""Configuration management"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration container"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getattr__(self, key: str):
        """Allow dot notation access"""
        value = self._config.get(key)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str):
        """Allow dict-style access"""
        return self._config[key]

    def get(self, key: str, default=None):
        """Safe get with default"""
        return self._config.get(key, default)

    def to_dict(self) -> Dict:
        """Convert back to dict"""
        return self._config


def load_config(config_path: str = "config/phase1_config.yaml") -> Config:
    """Load configuration from YAML file"""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)
