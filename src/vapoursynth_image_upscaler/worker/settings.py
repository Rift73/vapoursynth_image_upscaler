"""
Worker settings loaded from environment variables.

These settings are set by the GUI before spawning worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.utils import get_env_int, get_env_bool, get_env_str
from ..core.constants import (
    DEFAULT_ONNX_PATH,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_MODEL_SCALE,
)


@dataclass(frozen=True)
class WorkerSettings:
    """
    Settings for the worker process, loaded from environment variables.

    These are set by the GUI before spawning the worker subprocess.
    """

    # Model path
    onnx_path: str

    # Tile sizes for vsmlrt inference
    tile_w_limit: int
    tile_h_limit: int

    # Super-resolution model scale (1, 2, 4, or 8)
    model_scale: int

    # Precision flags for vsmlrt Backend.TRT
    use_fp16: bool
    use_bf16: bool
    use_tf32: bool

    # Secondary output settings
    use_secondary_output: bool
    secondary_mode: str  # "width", "height", or "2x"
    secondary_width: int
    secondary_height: int

    # Same-directory output
    use_same_dir_output: bool
    same_dir_suffix: str

    # Overwrite behavior
    overwrite_output: bool

    # Custom main resolution (downscaled from SR output)
    custom_res_enabled: bool
    custom_width: int
    custom_height: int

    # Alpha processing flag
    use_alpha: bool

    # Append model suffix to output filename
    append_model_suffix: bool

    @classmethod
    def from_environment(cls) -> WorkerSettings:
        """Create WorkerSettings by reading environment variables."""
        onnx_path = get_env_str("ONNX_PATH", DEFAULT_ONNX_PATH)
        tile_w = get_env_int("TILE_W_LIMIT", DEFAULT_TILE_WIDTH)
        tile_h = get_env_int("TILE_H_LIMIT", DEFAULT_TILE_HEIGHT)

        model_scale = get_env_int("MODEL_SCALE", DEFAULT_MODEL_SCALE)
        if model_scale <= 0:
            model_scale = DEFAULT_MODEL_SCALE

        return cls(
            onnx_path=onnx_path,
            tile_w_limit=tile_w,
            tile_h_limit=tile_h,
            model_scale=model_scale,
            use_fp16=get_env_bool("USE_FP16", False),
            use_bf16=get_env_bool("USE_BF16", True),
            use_tf32=get_env_bool("USE_TF32", False),
            use_secondary_output=get_env_bool("USE_SECONDARY_OUTPUT", False),
            secondary_mode=get_env_str("SECONDARY_MODE", "width"),
            secondary_width=get_env_int("SECONDARY_WIDTH", 1920),
            secondary_height=get_env_int("SECONDARY_HEIGHT", 1080),
            use_same_dir_output=get_env_bool("USE_SAME_DIR_OUTPUT", False),
            same_dir_suffix=get_env_str("SAME_DIR_SUFFIX", "_upscaled"),
            overwrite_output=get_env_bool("OVERWRITE_OUTPUT", True),
            custom_res_enabled=get_env_bool("CUSTOM_RES_ENABLED", False),
            custom_width=get_env_int("CUSTOM_WIDTH", 0),
            custom_height=get_env_int("CUSTOM_HEIGHT", 0),
            use_alpha=get_env_bool("USE_ALPHA", False),
            append_model_suffix=get_env_bool("APPEND_MODEL_SUFFIX", False),
        )

    def get_model_suffix(self) -> str:
        """Get the model suffix to append to filenames, if enabled."""
        if not self.append_model_suffix or not self.onnx_path:
            return ""
        try:
            return "_" + Path(self.onnx_path).stem
        except Exception:
            return ""
