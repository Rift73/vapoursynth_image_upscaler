"""
Worker settings loaded from environment variables.

These settings are set by the GUI before spawning worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.utils import get_env_int, get_env_bool, get_env_str, get_env_float
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
    num_streams: int  # num_streams for Backend.TRT

    # Secondary output settings
    use_secondary_output: bool
    secondary_mode: str  # "width", "height", or "2x"
    secondary_width: int
    secondary_height: int
    secondary_kernel: str  # "lanczos" or "hermite"

    # Same-directory output
    use_same_dir_output: bool
    same_dir_suffix: str

    # Manga folder mode (Parent Folder_suffix/Subfolder/.../Image.png)
    manga_folder_enabled: bool

    # Overwrite behavior
    overwrite_output: bool

    # Custom main resolution (downscaled from SR output)
    custom_res_enabled: bool
    custom_res_mode: str  # "width", "height", or "2x"
    custom_width: int
    custom_height: int
    custom_res_kernel: str  # "lanczos" or "hermite"

    # Alpha processing flag
    use_alpha: bool

    # Append model suffix to output filename
    append_model_suffix: bool

    # Pre-scaling settings (downscale before upscaling)
    prescale_enabled: bool
    prescale_mode: str  # "width", "height", or "2x"
    prescale_width: int
    prescale_height: int
    prescale_kernel: str  # "lanczos" or "hermite"

    # Sharpening settings (CAS - Contrast Adaptive Sharpening)
    sharpen_enabled: bool
    sharpen_value: float  # 0.0 to 1.0

    # Input file extension (for output format detection)
    input_extension: str  # e.g., ".png", ".gif", ".mp4"

    # Input video duration in seconds (for GIF output decision)
    input_duration: float  # 0.0 for images

    # Input frame rate (for animated GIF output)
    input_fps: float  # frames per second, 0.0 for images

    # Animated output settings
    animated_output_format: str  # "GIF", "WebP", "AVIF", or "APNG"
    # GIF settings (gifski)
    gif_quality: int  # 1-100 (quality, lower = smaller file)
    gif_fast: bool  # --fast mode (50% faster, 10% worse quality)
    # WebP settings (FFmpeg libwebp)
    webp_quality: int  # 0-100
    webp_lossless: bool
    webp_preset: str  # "none", "default", "picture", etc.
    # AVIF settings (avifenc)
    avif_quality: int  # 0-100 (color quality)
    avif_quality_alpha: int  # 0-100 (alpha quality)
    avif_speed: int  # 0-10 (0=slowest/best, 10=fastest)
    avif_lossless: bool
    # APNG settings (FFmpeg apng encoder)
    apng_pred: str  # "none", "sub", "up", "avg", "paeth", "mixed"

    # Upscale toggle - when False, skips SR upscaling and only applies resolution/alpha
    upscale_enabled: bool

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
            num_streams=get_env_int("NUM_STREAMS", 1),
            use_secondary_output=get_env_bool("USE_SECONDARY_OUTPUT", False),
            secondary_mode=get_env_str("SECONDARY_MODE", "width"),
            secondary_width=get_env_int("SECONDARY_WIDTH", 1920),
            secondary_height=get_env_int("SECONDARY_HEIGHT", 1080),
            secondary_kernel=get_env_str("SECONDARY_KERNEL", "lanczos"),
            use_same_dir_output=get_env_bool("USE_SAME_DIR_OUTPUT", False),
            same_dir_suffix=get_env_str("SAME_DIR_SUFFIX", "_upscaled"),
            manga_folder_enabled=get_env_bool("MANGA_FOLDER_ENABLED", False),
            overwrite_output=get_env_bool("OVERWRITE_OUTPUT", True),
            custom_res_enabled=get_env_bool("CUSTOM_RES_ENABLED", False),
            custom_res_mode=get_env_str("CUSTOM_RES_MODE", "width"),
            custom_width=get_env_int("CUSTOM_WIDTH", 0),
            custom_height=get_env_int("CUSTOM_HEIGHT", 0),
            custom_res_kernel=get_env_str("CUSTOM_RES_KERNEL", "lanczos"),
            use_alpha=get_env_bool("USE_ALPHA", False),
            append_model_suffix=get_env_bool("APPEND_MODEL_SUFFIX", False),
            prescale_enabled=get_env_bool("PRESCALE_ENABLED", False),
            prescale_mode=get_env_str("PRESCALE_MODE", "width"),
            prescale_width=get_env_int("PRESCALE_WIDTH", 1920),
            prescale_height=get_env_int("PRESCALE_HEIGHT", 1080),
            prescale_kernel=get_env_str("PRESCALE_KERNEL", "lanczos"),
            sharpen_enabled=get_env_bool("SHARPEN_ENABLED", False),
            sharpen_value=get_env_float("SHARPEN_VALUE", 0.5),
            input_extension=get_env_str("INPUT_EXTENSION", ".png"),
            input_duration=get_env_float("INPUT_DURATION", 0.0),
            input_fps=get_env_float("INPUT_FPS", 0.0),
            animated_output_format=get_env_str("ANIMATED_OUTPUT_FORMAT", "GIF"),
            gif_quality=get_env_int("GIF_QUALITY", 90),
            gif_fast=get_env_bool("GIF_FAST", False),
            webp_quality=get_env_int("WEBP_QUALITY", 90),
            webp_lossless=get_env_bool("WEBP_LOSSLESS", True),
            webp_preset=get_env_str("WEBP_PRESET", "none"),
            avif_quality=get_env_int("AVIF_QUALITY", 80),
            avif_quality_alpha=get_env_int("AVIF_QUALITY_ALPHA", 90),
            avif_speed=get_env_int("AVIF_SPEED", 6),
            avif_lossless=get_env_bool("AVIF_LOSSLESS", False),
            apng_pred=get_env_str("APNG_PRED", "mixed"),
            upscale_enabled=get_env_bool("UPSCALE_ENABLED", True),
        )

    def get_model_suffix(self) -> str:
        """Get the model suffix to append to filenames, if enabled."""
        if not self.append_model_suffix or not self.onnx_path:
            return ""
        try:
            return "_" + Path(self.onnx_path).stem
        except Exception:
            return ""

    def compute_dest_stem_and_dir(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> tuple[str, Path]:
        """
        Compute destination filename stem and directory based on settings.

        Args:
            input_path: Path to the input file.
            output_dir: Default output directory.

        Returns:
            Tuple of (dest_stem, dest_dir) where:
            - dest_stem: The filename without extension (e.g., "image_upscaled_ModelName")
            - dest_dir: The directory to write the output to
        """
        base_name = input_path.stem
        model_suffix = self.get_model_suffix()

        if self.manga_folder_enabled:
            # Manga folder mode: suffix is in the parent folder name, not the file
            dest_stem = f"{base_name}{model_suffix}"
            dest_dir = output_dir
        elif self.use_same_dir_output:
            suffix = self.same_dir_suffix or ""
            dest_stem = f"{base_name}{suffix}{model_suffix}"
            dest_dir = input_path.parent
        else:
            dest_stem = f"{base_name}{model_suffix}"
            dest_dir = output_dir

        return dest_stem, dest_dir
