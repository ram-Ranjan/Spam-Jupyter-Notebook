"""
Configuration loader utility
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'model.rf_params.n_estimators')
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> config = {'model': {'rf_params': {'n_estimators': 100}}}
        >>> get_config_value(config, 'model.rf_params.n_estimators')
        100
    """
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
