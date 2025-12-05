"""
Configuration management for the VapourSynth Image Upscaler.

Handles loading and saving of user settings to a JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .constants import (
    CONFIG_PATH,
    DEFAULT_ONNX_PATH,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_MODEL_SCALE,
)


@dataclass
class Config:
    """
    Application configuration dataclass.

    All settings are stored here and can be loaded/saved to JSON.
    """

    # Model settings
    onnx_path: str = DEFAULT_ONNX_PATH
    tile_w_limit: int = DEFAULT_TILE_WIDTH
    tile_h_limit: int = DEFAULT_TILE_HEIGHT
    model_scale: int = DEFAULT_MODEL_SCALE

    # Precision flags (for vsmlrt Backend.TRT)
    use_fp16: bool = False
    use_bf16: bool = True
    use_tf32: bool = False

    # Output options
    same_dir: bool = False
    same_dir_suffix: str = "_upscaled"
    append_model_suffix: bool = False
    overwrite: bool = True
    use_alpha: bool = False

    # Custom resolution settings
    custom_res_enabled: bool = False
    custom_res_width: int = 0
    custom_res_height: int = 0
    custom_res_maintain_ar: bool = True

    # Secondary output settings
    secondary_enabled: bool = False
    secondary_mode: str = "width"  # "width", "height", or "2x"
    secondary_width: int = 1920
    secondary_height: int = 1080

    # Last used input path (for convenience on restart)
    input_path: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """
        Load configuration from a JSON file.

        Args:
            path: Path to the config file. Defaults to CONFIG_PATH.

        Returns:
            Loaded Config instance, or default Config if file doesn't exist.
        """
        config_path = path or CONFIG_PATH

        if not config_path.exists():
            return cls()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls._from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            return cls()

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        """Create a Config from a dictionary, handling missing/extra keys."""
        config = cls()

        # Model settings
        config.onnx_path = str(data.get("onnx_path", config.onnx_path))
        config.tile_w_limit = _parse_int(data.get("tile_w_limit"), config.tile_w_limit)
        config.tile_h_limit = _parse_int(data.get("tile_h_limit"), config.tile_h_limit)
        config.model_scale = _parse_model_scale(data.get("model_scale"), config.model_scale)

        # Precision flags
        config.use_fp16 = bool(data.get("use_fp16", config.use_fp16))
        config.use_bf16 = bool(data.get("use_bf16", config.use_bf16))
        config.use_tf32 = bool(data.get("use_tf32", config.use_tf32))

        # Output options
        config.same_dir = bool(data.get("same_dir", config.same_dir))
        config.same_dir_suffix = str(data.get("same_dir_suffix", config.same_dir_suffix))
        config.append_model_suffix = bool(data.get("append_model_suffix", config.append_model_suffix))
        config.overwrite = bool(data.get("overwrite", config.overwrite))
        config.use_alpha = bool(data.get("use_alpha", config.use_alpha))

        # Custom resolution
        config.custom_res_enabled = bool(data.get("custom_res_enabled", config.custom_res_enabled))
        config.custom_res_width = _parse_int(data.get("custom_res_width"), config.custom_res_width)
        config.custom_res_height = _parse_int(data.get("custom_res_height"), config.custom_res_height)
        config.custom_res_maintain_ar = bool(data.get("custom_res_maintain_ar", config.custom_res_maintain_ar))

        # Secondary output
        config.secondary_enabled = bool(data.get("secondary_enabled", config.secondary_enabled))
        config.secondary_mode = str(data.get("secondary_mode", config.secondary_mode))
        config.secondary_width = _parse_int(data.get("secondary_width"), config.secondary_width)
        config.secondary_height = _parse_int(data.get("secondary_height"), config.secondary_height)

        # Last input
        config.input_path = str(data.get("input_path", config.input_path))

        return config

    def save(self, path: Path | None = None) -> None:
        """
        Save configuration to a JSON file.

        Args:
            path: Path to the config file. Defaults to CONFIG_PATH.
        """
        config_path = path or CONFIG_PATH

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except OSError as e:
            print(f"Warning: Failed to save config to {config_path}: {e}")


def _parse_int(value: Any, default: int) -> int:
    """Parse an integer value with a default fallback."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _parse_model_scale(value: Any, default: int) -> int:
    """Parse model scale, ensuring it's a valid value (1, 2, 4, or 8)."""
    from .constants import VALID_MODEL_SCALES

    parsed = _parse_int(value, default)
    if parsed in VALID_MODEL_SCALES:
        return parsed
    return default
