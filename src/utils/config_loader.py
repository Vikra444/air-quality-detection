"""
Configuration management utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from .exceptions import ConfigurationError


def load_env_file(env_file: str = ".env") -> None:
    """Load environment variables from .env file."""
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try .env.example as fallback
        example_path = Path(".env.example")
        if example_path.exists():
            load_dotenv(example_path)


def get_env(key: str, default: Optional[Any] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with validation."""
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ConfigurationError(f"Required environment variable {key} is not set")
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        raise ConfigurationError(f"Environment variable {key} must be an integer")


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        raise ConfigurationError(f"Environment variable {key} must be a float")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_keys = ["database_url", "redis_url"]
    
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required configuration key: {key}")
    
    return True

