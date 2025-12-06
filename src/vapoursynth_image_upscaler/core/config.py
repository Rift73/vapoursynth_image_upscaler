"""
Configuration management for the VapourSynth Image Upscaler.

Handles loading and saving of user settings to the Windows Registry.
Settings are stored under HKEY_CURRENT_USER\\Software\\VapourSynthImageUpscaler.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from .constants import (
    DEFAULT_ONNX_PATH,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_MODEL_SCALE,
)

# Windows Registry key path
REGISTRY_KEY = r"Software\VapourSynthImageUpscaler"

# Registry access (Windows only)
if sys.platform == "win32":
    import winreg


@dataclass
class Config:
    """
    Application configuration dataclass.

    All settings are stored here and can be loaded/saved to the Windows Registry.
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
    num_streams: int = 1  # num_streams for Backend.TRT

    # Output options
    same_dir: bool = False
    same_dir_suffix: str = "_upscaled"
    manga_folder: bool = False
    append_model_suffix: bool = False
    overwrite: bool = True
    use_alpha: bool = False
    batch_mode: bool = False

    # Custom resolution settings
    custom_res_enabled: bool = False
    custom_res_width: int = 0
    custom_res_height: int = 0
    custom_res_maintain_ar: bool = True
    custom_res_mode: str = "width"  # "width", "height", or "2x"
    custom_res_kernel: str = "lanczos"  # "lanczos" or "hermite"

    # Secondary output settings
    secondary_enabled: bool = False
    secondary_mode: str = "width"  # "width", "height", or "2x"
    secondary_width: int = 1920
    secondary_height: int = 1080
    secondary_kernel: str = "lanczos"  # "lanczos" or "hermite"

    # Pre-scaling settings (downscale before upscaling)
    prescale_enabled: bool = False
    prescale_mode: str = "width"  # "width", "height", or "2x"
    prescale_width: int = 1920
    prescale_height: int = 1080
    prescale_kernel: str = "lanczos"  # "lanczos" or "hermite"

    # Sharpening settings (CAS - Contrast Adaptive Sharpening)
    sharpen_enabled: bool = False
    sharpen_value: float = 0.5  # 0.0 to 1.0

    # Last used input path (for convenience on restart)
    input_path: str = ""

    # Last used ONNX browse directory (for file browser persistence)
    last_onnx_browse_dir: str = ""

    @classmethod
    def load(cls) -> Config:
        """
        Load configuration from the Windows Registry.

        Returns:
            Loaded Config instance, or default Config if registry key doesn't exist.
        """
        if sys.platform != "win32":
            return cls()

        config = cls()

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY) as key:
                config.onnx_path = _read_reg_str(key, "onnx_path", config.onnx_path)
                config.tile_w_limit = _read_reg_int(key, "tile_w_limit", config.tile_w_limit)
                config.tile_h_limit = _read_reg_int(key, "tile_h_limit", config.tile_h_limit)
                config.model_scale = _parse_model_scale(
                    _read_reg_int(key, "model_scale", config.model_scale),
                    config.model_scale
                )

                config.use_fp16 = _read_reg_bool(key, "use_fp16", config.use_fp16)
                config.use_bf16 = _read_reg_bool(key, "use_bf16", config.use_bf16)
                config.use_tf32 = _read_reg_bool(key, "use_tf32", config.use_tf32)
                config.num_streams = _read_reg_int(key, "num_streams", config.num_streams)

                config.same_dir = _read_reg_bool(key, "same_dir", config.same_dir)
                config.same_dir_suffix = _read_reg_str(key, "same_dir_suffix", config.same_dir_suffix)
                config.manga_folder = _read_reg_bool(key, "manga_folder", config.manga_folder)
                config.append_model_suffix = _read_reg_bool(key, "append_model_suffix", config.append_model_suffix)
                config.overwrite = _read_reg_bool(key, "overwrite", config.overwrite)
                config.use_alpha = _read_reg_bool(key, "use_alpha", config.use_alpha)
                config.batch_mode = _read_reg_bool(key, "batch_mode", config.batch_mode)

                config.custom_res_enabled = _read_reg_bool(key, "custom_res_enabled", config.custom_res_enabled)
                config.custom_res_width = _read_reg_int(key, "custom_res_width", config.custom_res_width)
                config.custom_res_height = _read_reg_int(key, "custom_res_height", config.custom_res_height)
                config.custom_res_maintain_ar = _read_reg_bool(key, "custom_res_maintain_ar", config.custom_res_maintain_ar)
                config.custom_res_mode = _read_reg_str(key, "custom_res_mode", config.custom_res_mode)
                config.custom_res_kernel = _read_reg_str(key, "custom_res_kernel", config.custom_res_kernel)

                config.secondary_enabled = _read_reg_bool(key, "secondary_enabled", config.secondary_enabled)
                config.secondary_mode = _read_reg_str(key, "secondary_mode", config.secondary_mode)
                config.secondary_width = _read_reg_int(key, "secondary_width", config.secondary_width)
                config.secondary_height = _read_reg_int(key, "secondary_height", config.secondary_height)
                config.secondary_kernel = _read_reg_str(key, "secondary_kernel", config.secondary_kernel)

                config.prescale_enabled = _read_reg_bool(key, "prescale_enabled", config.prescale_enabled)
                config.prescale_mode = _read_reg_str(key, "prescale_mode", config.prescale_mode)
                config.prescale_width = _read_reg_int(key, "prescale_width", config.prescale_width)
                config.prescale_height = _read_reg_int(key, "prescale_height", config.prescale_height)
                config.prescale_kernel = _read_reg_str(key, "prescale_kernel", config.prescale_kernel)

                config.sharpen_enabled = _read_reg_bool(key, "sharpen_enabled", config.sharpen_enabled)
                config.sharpen_value = _read_reg_float(key, "sharpen_value", config.sharpen_value)

                config.input_path = _read_reg_str(key, "input_path", config.input_path)
                config.last_onnx_browse_dir = _read_reg_str(key, "last_onnx_browse_dir", config.last_onnx_browse_dir)

        except FileNotFoundError:
            # Registry key doesn't exist yet, use defaults
            pass
        except OSError:
            # Other registry errors, use defaults
            pass

        return config

    def save(self) -> None:
        """Save configuration to the Windows Registry."""
        if sys.platform != "win32":
            return

        try:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY) as key:
                _write_reg_str(key, "onnx_path", self.onnx_path)
                _write_reg_int(key, "tile_w_limit", self.tile_w_limit)
                _write_reg_int(key, "tile_h_limit", self.tile_h_limit)
                _write_reg_int(key, "model_scale", self.model_scale)

                _write_reg_bool(key, "use_fp16", self.use_fp16)
                _write_reg_bool(key, "use_bf16", self.use_bf16)
                _write_reg_bool(key, "use_tf32", self.use_tf32)
                _write_reg_int(key, "num_streams", self.num_streams)

                _write_reg_bool(key, "same_dir", self.same_dir)
                _write_reg_str(key, "same_dir_suffix", self.same_dir_suffix)
                _write_reg_bool(key, "manga_folder", self.manga_folder)
                _write_reg_bool(key, "append_model_suffix", self.append_model_suffix)
                _write_reg_bool(key, "overwrite", self.overwrite)
                _write_reg_bool(key, "use_alpha", self.use_alpha)
                _write_reg_bool(key, "batch_mode", self.batch_mode)

                _write_reg_bool(key, "custom_res_enabled", self.custom_res_enabled)
                _write_reg_int(key, "custom_res_width", self.custom_res_width)
                _write_reg_int(key, "custom_res_height", self.custom_res_height)
                _write_reg_bool(key, "custom_res_maintain_ar", self.custom_res_maintain_ar)
                _write_reg_str(key, "custom_res_mode", self.custom_res_mode)
                _write_reg_str(key, "custom_res_kernel", self.custom_res_kernel)

                _write_reg_bool(key, "secondary_enabled", self.secondary_enabled)
                _write_reg_str(key, "secondary_mode", self.secondary_mode)
                _write_reg_int(key, "secondary_width", self.secondary_width)
                _write_reg_int(key, "secondary_height", self.secondary_height)
                _write_reg_str(key, "secondary_kernel", self.secondary_kernel)

                _write_reg_bool(key, "prescale_enabled", self.prescale_enabled)
                _write_reg_str(key, "prescale_mode", self.prescale_mode)
                _write_reg_int(key, "prescale_width", self.prescale_width)
                _write_reg_int(key, "prescale_height", self.prescale_height)
                _write_reg_str(key, "prescale_kernel", self.prescale_kernel)

                _write_reg_bool(key, "sharpen_enabled", self.sharpen_enabled)
                _write_reg_float(key, "sharpen_value", self.sharpen_value)

                _write_reg_str(key, "input_path", self.input_path)
                _write_reg_str(key, "last_onnx_browse_dir", self.last_onnx_browse_dir)

        except OSError:
            # Failed to write to registry, silently ignore
            pass


def _read_reg_str(key, name: str, default: str) -> str:
    """Read a string value from the registry."""
    try:
        value, _ = winreg.QueryValueEx(key, name)
        return str(value) if value is not None else default
    except FileNotFoundError:
        return default


def _read_reg_int(key, name: str, default: int) -> int:
    """Read an integer value from the registry."""
    try:
        value, _ = winreg.QueryValueEx(key, name)
        return int(value) if value is not None else default
    except (FileNotFoundError, ValueError, TypeError):
        return default


def _read_reg_bool(key, name: str, default: bool) -> bool:
    """Read a boolean value from the registry (stored as DWORD 0/1)."""
    try:
        value, _ = winreg.QueryValueEx(key, name)
        return bool(value)
    except FileNotFoundError:
        return default


def _read_reg_float(key, name: str, default: float) -> float:
    """Read a float value from the registry (stored as string)."""
    try:
        value, _ = winreg.QueryValueEx(key, name)
        return float(value) if value is not None else default
    except (FileNotFoundError, ValueError, TypeError):
        return default


def _write_reg_str(key, name: str, value: str) -> None:
    """Write a string value to the registry."""
    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def _write_reg_int(key, name: str, value: int) -> None:
    """Write an integer value to the registry."""
    winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)


def _write_reg_bool(key, name: str, value: bool) -> None:
    """Write a boolean value to the registry (as DWORD 0/1)."""
    winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, 1 if value else 0)


def _write_reg_float(key, name: str, value: float) -> None:
    """Write a float value to the registry (as string)."""
    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, str(value))


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
